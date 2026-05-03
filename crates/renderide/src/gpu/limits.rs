//! Centralized GPU capability snapshot from [`wgpu::Device::limits`], [`wgpu::Device::features`], and
//! [`wgpu::Adapter::get_downlevel_capabilities`].
//!
//! Construct once after [`wgpu::Device`] creation via [`GpuLimits::try_new`] and pass [`std::sync::Arc`]
//! through upload paths and frame resources instead of calling [`wgpu::Device::limits`] ad hoc.

use std::sync::Arc;

use thiserror::Error;

mod validation;

/// Number of array layers used for a GPU cubemap (six faces).
pub const CUBEMAP_ARRAY_LAYERS: u32 = 6;

/// Product cap (texels per edge) for host-reported max texture size when the GPU is not yet
/// available, and upper bound used by [`GpuLimits::clamp_render_texture_edge`] together with
/// [`wgpu::Limits::max_texture_dimension_2d`] (see [`crate::config::RendererSettings::reported_max_texture_dimension_for_host`]).
pub const REPORTED_MAX_TEXTURE_SIZE_FALLBACK_EDGE: u32 = 8192;

/// Renderer-specific GPU limits and feature flags (immutable after construction).
#[derive(Clone, Debug)]
pub struct GpuLimits {
    /// Full wgpu limits for the active device (post-`request_device` effective caps).
    pub wgpu: wgpu::Limits,
    /// Whether merged mesh draws may use non-zero `first_instance` ([`wgpu::DownlevelCapabilities::is_webgpu_compliant`]).
    pub supports_base_instance: bool,
    /// Whether [`wgpu::Features::MULTIVIEW`] was enabled on the device.
    pub supports_multiview: bool,
    /// Whether [`wgpu::Features::FLOAT32_FILTERABLE`] is present (embedded materials / filterable float).
    pub supports_float32_filterable: bool,
    /// BC / ETC2 / ASTC bits that were requested and enabled (for diagnostics).
    pub texture_compression_features: wgpu::Features,
    /// Maximum rows in the mesh-forward `@group(2)` storage slab (`max_storage_buffer_binding_size / 256`).
    pub max_per_draw_slab_slots: usize,
}

/// Minimum requirements not met for running the default render graph.
#[derive(Debug, Error)]
pub enum GpuLimitsError {
    /// Field-specific validation failure.
    #[error("GPU limits insufficient for Renderide: {0}")]
    Requirement(&'static str),
}

impl GpuLimits {
    /// Builds a snapshot from the device and adapter (downlevel flags from `adapter`).
    ///
    /// Fails when core WebGPU-style minimums for this codebase are not met (bind groups, storage
    /// binding size for the per-draw slab, texture dimensions).
    pub fn try_new(
        device: &wgpu::Device,
        adapter: &wgpu::Adapter,
    ) -> Result<Arc<Self>, GpuLimitsError> {
        validation::try_new(device, adapter)
    }

    /// `min_storage_buffer_offset_alignment` for dynamic storage offsets (e.g. per-draw slab).
    #[inline]
    pub fn min_storage_buffer_offset_alignment(&self) -> u32 {
        self.wgpu.min_storage_buffer_offset_alignment
    }

    /// `min_uniform_buffer_offset_alignment` for dynamic uniform offsets.
    #[inline]
    pub fn min_uniform_buffer_offset_alignment(&self) -> u32 {
        self.wgpu.min_uniform_buffer_offset_alignment
    }

    /// Rounds `n` up to a multiple of [`Self::min_storage_buffer_offset_alignment`].
    #[inline]
    pub fn align_storage_offset(&self, n: u64) -> u64 {
        let align = u64::from(self.wgpu.min_storage_buffer_offset_alignment).max(1);
        n.div_ceil(align) * align
    }

    /// Rounds `n` up to a multiple of [`Self::min_uniform_buffer_offset_alignment`].
    #[inline]
    pub fn align_uniform_offset(&self, n: u64) -> u64 {
        let align = u64::from(self.wgpu.min_uniform_buffer_offset_alignment).max(1);
        n.div_ceil(align) * align
    }

    /// `max_buffer_size` for the device.
    #[inline]
    pub fn max_buffer_size(&self) -> u64 {
        self.wgpu.max_buffer_size
    }

    /// `max_storage_buffer_binding_size` for the device.
    #[inline]
    pub fn max_storage_buffer_binding_size(&self) -> u64 {
        self.wgpu.max_storage_buffer_binding_size
    }

    /// `max_uniform_buffer_binding_size` for the device.
    #[inline]
    pub fn max_uniform_buffer_binding_size(&self) -> u64 {
        self.wgpu.max_uniform_buffer_binding_size
    }

    /// Returns `true` if `bytes` fits in [`Self::max_buffer_size`].
    #[must_use]
    #[inline]
    pub fn buffer_size_fits(&self, bytes: u64) -> bool {
        bytes <= self.wgpu.max_buffer_size
    }

    /// Returns `true` if `bytes` fits in [`Self::max_storage_buffer_binding_size`].
    #[must_use]
    #[inline]
    pub fn storage_binding_fits(&self, bytes: u64) -> bool {
        bytes <= self.wgpu.max_storage_buffer_binding_size
    }

    /// Returns `true` if `bytes` fits in [`Self::max_uniform_buffer_binding_size`].
    #[must_use]
    #[inline]
    pub fn uniform_binding_fits(&self, bytes: u64) -> bool {
        bytes <= self.wgpu.max_uniform_buffer_binding_size
    }

    /// `max_texture_dimension_2d` for the device.
    #[inline]
    pub fn max_texture_dimension_2d(&self) -> u32 {
        self.wgpu.max_texture_dimension_2d
    }

    /// `max_texture_dimension_3d` for the device.
    #[inline]
    pub fn max_texture_dimension_3d(&self) -> u32 {
        self.wgpu.max_texture_dimension_3d
    }

    /// `max_texture_array_layers` for the device (cubemaps use [`CUBEMAP_ARRAY_LAYERS`]).
    #[inline]
    pub fn max_texture_array_layers(&self) -> u32 {
        self.wgpu.max_texture_array_layers
    }

    /// Returns `true` if `(w, h)` fits in [`Self::max_texture_dimension_2d`].
    #[must_use]
    #[inline]
    pub fn texture_2d_fits(&self, w: u32, h: u32) -> bool {
        let m = self.wgpu.max_texture_dimension_2d;
        w <= m && h <= m
    }

    /// Returns `true` if `(w, h, d)` fits in [`Self::max_texture_dimension_3d`].
    #[must_use]
    #[inline]
    pub fn texture_3d_fits(&self, w: u32, h: u32, d: u32) -> bool {
        let m = self.wgpu.max_texture_dimension_3d;
        w <= m && h <= m && d <= m
    }

    /// Returns `true` if `layers` fits in [`Self::max_texture_array_layers`].
    #[must_use]
    #[inline]
    pub fn array_layers_fit(&self, layers: u32) -> bool {
        layers <= self.wgpu.max_texture_array_layers
    }

    /// Returns `true` when [`Self::max_texture_array_layers`] is at least [`CUBEMAP_ARRAY_LAYERS`].
    #[must_use]
    #[inline]
    pub fn cubemap_fits_texture_array_layers(&self) -> bool {
        self.wgpu.max_texture_array_layers >= CUBEMAP_ARRAY_LAYERS
    }

    /// `max_compute_workgroups_per_dimension` for dispatch validation.
    #[inline]
    pub fn max_compute_workgroups_per_dimension(&self) -> u32 {
        self.wgpu.max_compute_workgroups_per_dimension
    }

    /// `max_compute_invocations_per_workgroup` for shader workgroup-size validation.
    #[inline]
    pub fn max_compute_invocations_per_workgroup(&self) -> u32 {
        self.wgpu.max_compute_invocations_per_workgroup
    }

    /// `max_compute_workgroup_size_x` for shader workgroup-size validation.
    #[inline]
    pub fn max_compute_workgroup_size_x(&self) -> u32 {
        self.wgpu.max_compute_workgroup_size_x
    }

    /// `max_compute_workgroup_size_y` for shader workgroup-size validation.
    #[inline]
    pub fn max_compute_workgroup_size_y(&self) -> u32 {
        self.wgpu.max_compute_workgroup_size_y
    }

    /// `max_compute_workgroup_size_z` for shader workgroup-size validation.
    #[inline]
    pub fn max_compute_workgroup_size_z(&self) -> u32 {
        self.wgpu.max_compute_workgroup_size_z
    }

    /// Returns `true` if `(x,y,z)` dispatch dimensions are within per-axis limits.
    #[must_use]
    #[inline]
    pub fn compute_dispatch_fits(&self, x: u32, y: u32, z: u32) -> bool {
        let m = self.wgpu.max_compute_workgroups_per_dimension;
        x <= m && y <= m && z <= m
    }

    /// Returns `true` if a `@workgroup_size(x, y, z)` declaration fits the device's per-axis caps
    /// and total invocation cap.
    #[must_use]
    #[inline]
    pub fn workgroup_size_fits(&self, x: u32, y: u32, z: u32) -> bool {
        x <= self.wgpu.max_compute_workgroup_size_x
            && y <= self.wgpu.max_compute_workgroup_size_y
            && z <= self.wgpu.max_compute_workgroup_size_z
            && u64::from(x) * u64::from(y) * u64::from(z)
                <= u64::from(self.wgpu.max_compute_invocations_per_workgroup)
    }

    /// `max_bind_groups` for the device.
    #[inline]
    pub fn max_bind_groups(&self) -> u32 {
        self.wgpu.max_bind_groups
    }

    /// `max_bindings_per_bind_group` for the device.
    #[inline]
    pub fn max_bindings_per_bind_group(&self) -> u32 {
        self.wgpu.max_bindings_per_bind_group
    }

    /// `max_samplers_per_shader_stage` for the device.
    #[inline]
    pub fn max_samplers_per_shader_stage(&self) -> u32 {
        self.wgpu.max_samplers_per_shader_stage
    }

    /// `max_sampled_textures_per_shader_stage` for the device.
    #[inline]
    pub fn max_sampled_textures_per_shader_stage(&self) -> u32 {
        self.wgpu.max_sampled_textures_per_shader_stage
    }

    /// `max_storage_textures_per_shader_stage` for the device.
    #[inline]
    pub fn max_storage_textures_per_shader_stage(&self) -> u32 {
        self.wgpu.max_storage_textures_per_shader_stage
    }

    /// `max_storage_buffers_per_shader_stage` for the device.
    #[inline]
    pub fn max_storage_buffers_per_shader_stage(&self) -> u32 {
        self.wgpu.max_storage_buffers_per_shader_stage
    }

    /// `max_uniform_buffers_per_shader_stage` for the device.
    #[inline]
    pub fn max_uniform_buffers_per_shader_stage(&self) -> u32 {
        self.wgpu.max_uniform_buffers_per_shader_stage
    }

    /// `max_color_attachments` for the device.
    #[inline]
    pub fn max_color_attachments(&self) -> u32 {
        self.wgpu.max_color_attachments
    }

    /// `max_vertex_buffers` for the device.
    #[inline]
    pub fn max_vertex_buffers(&self) -> u32 {
        self.wgpu.max_vertex_buffers
    }

    /// `max_vertex_attributes` for the device.
    #[inline]
    pub fn max_vertex_attributes(&self) -> u32 {
        self.wgpu.max_vertex_attributes
    }

    /// Clamps host edge length for render textures: `[4, min(REPORTED_MAX_TEXTURE_SIZE_FALLBACK_EDGE, max_texture_dimension_2d)]`.
    #[inline]
    pub fn clamp_render_texture_edge(&self, edge: i32) -> u32 {
        let cap = self
            .wgpu
            .max_texture_dimension_2d
            .min(REPORTED_MAX_TEXTURE_SIZE_FALLBACK_EDGE);
        edge.clamp(4, cap as i32) as u32
    }

    /// Clamps `edge` to `[1, max_texture_dimension_2d]`. Returns `None` when `edge == 0`.
    #[must_use]
    #[inline]
    pub fn clamp_texture_2d_edge(&self, edge: u32) -> Option<u32> {
        if edge == 0 {
            return None;
        }
        Some(edge.min(self.wgpu.max_texture_dimension_2d))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_dispatch_fits_respects_max_per_axis() {
        let l = wgpu::Limits {
            max_compute_workgroups_per_dimension: 256,
            ..Default::default()
        };
        let gl = GpuLimits {
            wgpu: l,
            supports_base_instance: true,
            supports_multiview: false,
            supports_float32_filterable: false,
            texture_compression_features: wgpu::Features::empty(),
            max_per_draw_slab_slots: 1024,
        };
        assert!(gl.compute_dispatch_fits(256, 256, 24));
        assert!(!gl.compute_dispatch_fits(257, 1, 1));
    }

    fn synthetic_limits(max_tex_2d: u32) -> GpuLimits {
        GpuLimits {
            wgpu: wgpu::Limits {
                max_texture_dimension_2d: max_tex_2d,
                ..Default::default()
            },
            supports_base_instance: true,
            supports_multiview: false,
            supports_float32_filterable: false,
            texture_compression_features: wgpu::Features::empty(),
            max_per_draw_slab_slots: 1024,
        }
    }

    fn synthetic_limits_layers(max_tex_2d: u32, max_array_layers: u32) -> GpuLimits {
        GpuLimits {
            wgpu: wgpu::Limits {
                max_texture_dimension_2d: max_tex_2d,
                max_texture_array_layers: max_array_layers,
                ..Default::default()
            },
            supports_base_instance: true,
            supports_multiview: false,
            supports_float32_filterable: false,
            texture_compression_features: wgpu::Features::empty(),
            max_per_draw_slab_slots: 1024,
        }
    }

    #[test]
    fn cubemap_fits_requires_six_array_layers() {
        assert!(!synthetic_limits_layers(4096, 4).cubemap_fits_texture_array_layers());
        assert!(synthetic_limits_layers(4096, 6).cubemap_fits_texture_array_layers());
    }

    #[test]
    fn clamp_render_texture_edge_clamps_min_to_four() {
        let gl = synthetic_limits(8192);
        assert_eq!(gl.clamp_render_texture_edge(0), 4);
        assert_eq!(gl.clamp_render_texture_edge(-100), 4);
        assert_eq!(gl.clamp_render_texture_edge(3), 4);
        assert_eq!(gl.clamp_render_texture_edge(4), 4);
    }

    #[test]
    fn clamp_render_texture_edge_caps_at_min_of_fallback_edge_and_gpu_max() {
        let gl_small = synthetic_limits(512);
        assert_eq!(gl_small.clamp_render_texture_edge(10_000), 512);

        let gl_large = synthetic_limits(16384);
        assert_eq!(
            gl_large.clamp_render_texture_edge(100_000),
            REPORTED_MAX_TEXTURE_SIZE_FALLBACK_EDGE
        );
        assert_eq!(gl_large.clamp_render_texture_edge(4096), 4096);
    }

    fn limits_with(wgpu_limits: wgpu::Limits) -> GpuLimits {
        let max_per_draw_slab_slots = (wgpu_limits.max_storage_buffer_binding_size / 256) as usize;
        GpuLimits {
            wgpu: wgpu_limits,
            supports_base_instance: true,
            supports_multiview: false,
            supports_float32_filterable: false,
            texture_compression_features: wgpu::Features::empty(),
            max_per_draw_slab_slots,
        }
    }

    #[test]
    fn texture_2d_fits_checks_both_axes() {
        let gl = synthetic_limits(4096);
        assert!(gl.texture_2d_fits(4096, 4096));
        assert!(!gl.texture_2d_fits(4097, 4096));
        assert!(!gl.texture_2d_fits(4096, 4097));
    }

    #[test]
    fn texture_3d_fits_checks_all_axes() {
        let gl = limits_with(wgpu::Limits {
            max_texture_dimension_3d: 256,
            ..Default::default()
        });
        assert!(gl.texture_3d_fits(256, 256, 256));
        assert!(!gl.texture_3d_fits(257, 256, 256));
        assert!(!gl.texture_3d_fits(256, 257, 256));
        assert!(!gl.texture_3d_fits(256, 256, 257));
    }

    #[test]
    fn array_layers_fit_respects_limit() {
        let gl = limits_with(wgpu::Limits {
            max_texture_array_layers: 256,
            ..Default::default()
        });
        assert!(gl.array_layers_fit(256));
        assert!(!gl.array_layers_fit(257));
    }

    #[test]
    fn buffer_size_fits_respects_max_buffer_size() {
        let gl = limits_with(wgpu::Limits {
            max_buffer_size: 1024,
            ..Default::default()
        });
        assert!(gl.buffer_size_fits(1024));
        assert!(!gl.buffer_size_fits(1025));
    }

    #[test]
    fn storage_binding_fits_respects_max_storage_binding_size() {
        let gl = limits_with(wgpu::Limits {
            max_storage_buffer_binding_size: 65_536,
            ..Default::default()
        });
        assert!(gl.storage_binding_fits(65_536));
        assert!(!gl.storage_binding_fits(65_537));
    }

    #[test]
    fn uniform_binding_fits_respects_max_uniform_binding_size() {
        let gl = limits_with(wgpu::Limits {
            max_uniform_buffer_binding_size: 16_384,
            ..Default::default()
        });
        assert!(gl.uniform_binding_fits(16_384));
        assert!(!gl.uniform_binding_fits(16_385));
    }

    #[test]
    fn align_storage_offset_rounds_up_to_min_storage_alignment() {
        let gl = limits_with(wgpu::Limits {
            min_storage_buffer_offset_alignment: 64,
            ..Default::default()
        });
        assert_eq!(gl.align_storage_offset(0), 0);
        assert_eq!(gl.align_storage_offset(1), 64);
        assert_eq!(gl.align_storage_offset(64), 64);
        assert_eq!(gl.align_storage_offset(65), 128);
    }

    #[test]
    fn workgroup_size_fits_per_axis_and_total() {
        let gl = limits_with(wgpu::Limits {
            max_compute_workgroup_size_x: 256,
            max_compute_workgroup_size_y: 256,
            max_compute_workgroup_size_z: 64,
            max_compute_invocations_per_workgroup: 256,
            ..Default::default()
        });
        assert!(gl.workgroup_size_fits(16, 16, 1));
        assert!(gl.workgroup_size_fits(256, 1, 1));
        assert!(!gl.workgroup_size_fits(257, 1, 1));
        assert!(!gl.workgroup_size_fits(1, 1, 65));
        assert!(!gl.workgroup_size_fits(16, 16, 2));
    }

    #[test]
    fn clamp_texture_2d_edge_returns_none_for_zero() {
        let gl = synthetic_limits(4096);
        assert_eq!(gl.clamp_texture_2d_edge(0), None);
        assert_eq!(gl.clamp_texture_2d_edge(1), Some(1));
        assert_eq!(gl.clamp_texture_2d_edge(8192), Some(4096));
    }

    #[test]
    fn align_uniform_offset_rounds_up_and_zero_alignment_is_safe() {
        let gl = limits_with(wgpu::Limits {
            min_uniform_buffer_offset_alignment: 128,
            ..Default::default()
        });
        assert_eq!(gl.align_uniform_offset(0), 0);
        assert_eq!(gl.align_uniform_offset(1), 128);
        assert_eq!(gl.align_uniform_offset(128), 128);
        assert_eq!(gl.align_uniform_offset(129), 256);

        let zero = limits_with(wgpu::Limits {
            min_uniform_buffer_offset_alignment: 0,
            min_storage_buffer_offset_alignment: 0,
            ..Default::default()
        });
        assert_eq!(zero.align_uniform_offset(13), 13);
        assert_eq!(zero.align_storage_offset(13), 13);
    }

    #[test]
    fn public_limit_getters_mirror_wgpu_limits() {
        let gl = limits_with(wgpu::Limits {
            max_buffer_size: 1_000,
            max_storage_buffer_binding_size: 2_000,
            max_uniform_buffer_binding_size: 3_000,
            max_texture_dimension_2d: 4_000,
            max_texture_dimension_3d: 500,
            max_texture_array_layers: 64,
            max_compute_workgroups_per_dimension: 128,
            max_compute_invocations_per_workgroup: 256,
            max_compute_workgroup_size_x: 16,
            max_compute_workgroup_size_y: 32,
            max_compute_workgroup_size_z: 8,
            max_bind_groups: 4,
            max_bindings_per_bind_group: 32,
            max_samplers_per_shader_stage: 8,
            max_sampled_textures_per_shader_stage: 16,
            max_storage_textures_per_shader_stage: 4,
            max_storage_buffers_per_shader_stage: 6,
            max_uniform_buffers_per_shader_stage: 10,
            max_color_attachments: 4,
            max_vertex_buffers: 12,
            max_vertex_attributes: 24,
            ..Default::default()
        });

        assert_eq!(gl.max_buffer_size(), 1_000);
        assert_eq!(gl.max_storage_buffer_binding_size(), 2_000);
        assert_eq!(gl.max_uniform_buffer_binding_size(), 3_000);
        assert_eq!(gl.max_texture_dimension_2d(), 4_000);
        assert_eq!(gl.max_texture_dimension_3d(), 500);
        assert_eq!(gl.max_texture_array_layers(), 64);
        assert_eq!(gl.max_compute_workgroups_per_dimension(), 128);
        assert_eq!(gl.max_compute_invocations_per_workgroup(), 256);
        assert_eq!(gl.max_compute_workgroup_size_x(), 16);
        assert_eq!(gl.max_compute_workgroup_size_y(), 32);
        assert_eq!(gl.max_compute_workgroup_size_z(), 8);
        assert_eq!(gl.max_bind_groups(), 4);
        assert_eq!(gl.max_bindings_per_bind_group(), 32);
        assert_eq!(gl.max_samplers_per_shader_stage(), 8);
        assert_eq!(gl.max_sampled_textures_per_shader_stage(), 16);
        assert_eq!(gl.max_storage_textures_per_shader_stage(), 4);
        assert_eq!(gl.max_storage_buffers_per_shader_stage(), 6);
        assert_eq!(gl.max_uniform_buffers_per_shader_stage(), 10);
        assert_eq!(gl.max_color_attachments(), 4);
        assert_eq!(gl.max_vertex_buffers(), 12);
        assert_eq!(gl.max_vertex_attributes(), 24);
    }
}
