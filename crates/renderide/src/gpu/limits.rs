//! Centralized GPU capability snapshot from [`wgpu::Device::limits`], [`wgpu::Device::features`], and
//! [`wgpu::Adapter::get_downlevel_capabilities`].
//!
//! Construct once after [`wgpu::Device`] creation via [`GpuLimits::try_new`] and pass [`std::sync::Arc`]
//! through upload paths and frame resources instead of calling [`wgpu::Device::limits`] ad hoc.

use std::sync::Arc;

use thiserror::Error;

/// Per-draw row size in bytes; must match [`crate::backend::mesh_deform::PER_DRAW_UNIFORM_STRIDE`].
const PER_DRAW_UNIFORM_STRIDE: usize = 256;
/// Initial slab row count; must match [`crate::backend::mesh_deform::INITIAL_PER_DRAW_UNIFORM_SLOTS`].
const INITIAL_PER_DRAW_UNIFORM_SLOTS: usize = 256;

/// Number of array layers used for a GPU cubemap (six faces).
pub const CUBEMAP_ARRAY_LAYERS: u32 = 6;

/// Product cap (texels per edge) for host-reported max texture size when the GPU is not yet
/// available, and upper bound used by [`GpuLimits::clamp_render_texture_edge`] together with
/// [`wgpu::Limits::max_texture_dimension_2d`] (see [`crate::config::RendererSettings::reported_max_texture_dimension_for_host`]).
pub const REPORTED_MAX_TEXTURE_SIZE_FALLBACK_EDGE: u32 = 8192;

/// Renderer-specific GPU limits and feature flags (immutable after construction).
#[derive(Clone, Debug)]
pub struct GpuLimits {
    /// Full wgpu limits for the active device (post–`request_device` effective caps).
    pub wgpu: wgpu::Limits,
    /// Whether merged mesh draws may use non-zero `first_instance` (`wgpu::DownlevelCapabilities::is_webgpu_compliant`).
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
        let wgpu_limits = device.limits();
        let features = device.features();
        let down = adapter.get_downlevel_capabilities();

        Self::validate_wgpu_minimums(&wgpu_limits)?;

        // Non–WebGPU-compliant stacks (e.g. some GLES/WebGL paths) may not implement `first_instance`
        // for `draw_indexed` batching the same way; disable merged instance batches there.
        let supports_base_instance = down.is_webgpu_compliant();
        let supports_multiview = features.contains(wgpu::Features::MULTIVIEW);
        let supports_float32_filterable = features.contains(wgpu::Features::FLOAT32_FILTERABLE);
        let texture_compression_features = features
            & (wgpu::Features::TEXTURE_COMPRESSION_BC
                | wgpu::Features::TEXTURE_COMPRESSION_ETC2
                | wgpu::Features::TEXTURE_COMPRESSION_ASTC);

        let max_binding = wgpu_limits.max_storage_buffer_binding_size;
        let stride = PER_DRAW_UNIFORM_STRIDE as u64;
        let max_per_draw_slab_slots = (max_binding / stride) as usize;

        if max_per_draw_slab_slots < INITIAL_PER_DRAW_UNIFORM_SLOTS {
            return Err(GpuLimitsError::Requirement(
                "max_storage_buffer_binding_size too small for initial per-draw slab (256×256 B rows)",
            ));
        }

        let limits = Self {
            wgpu: wgpu_limits,
            supports_base_instance,
            supports_multiview,
            supports_float32_filterable,
            texture_compression_features,
            max_per_draw_slab_slots,
        };

        logger::info!(
            "GPU limits: max_texture_2d={} max_buffer={} max_storage_binding={} max_compute_wg_per_dim={} max_samplers_stage={} max_sampled_textures_stage={} base_instance={} multiview={}",
            limits.wgpu.max_texture_dimension_2d,
            limits.wgpu.max_buffer_size,
            limits.wgpu.max_storage_buffer_binding_size,
            limits.wgpu.max_compute_workgroups_per_dimension,
            limits.wgpu.max_samplers_per_shader_stage,
            limits.wgpu.max_sampled_textures_per_shader_stage,
            supports_base_instance,
            supports_multiview
        );

        Ok(Arc::new(limits))
    }

    fn validate_wgpu_minimums(l: &wgpu::Limits) -> Result<(), GpuLimitsError> {
        if l.max_bind_groups < 4 {
            return Err(GpuLimitsError::Requirement(
                "max_bind_groups must be at least 4 (frame / material / per-draw / …)",
            ));
        }
        if l.max_texture_dimension_2d < 1024 {
            return Err(GpuLimitsError::Requirement(
                "max_texture_dimension_2d must be at least 1024",
            ));
        }
        let min_slab = (INITIAL_PER_DRAW_UNIFORM_SLOTS * PER_DRAW_UNIFORM_STRIDE) as u64;
        if l.max_storage_buffer_binding_size < min_slab {
            return Err(GpuLimitsError::Requirement(
                "max_storage_buffer_binding_size must fit initial per-draw slab (65536 bytes)",
            ));
        }
        Ok(())
    }

    /// `min_storage_buffer_offset_alignment` for dynamic storage offsets (e.g. per-draw slab).
    #[inline]
    pub fn min_storage_buffer_offset_alignment(&self) -> u32 {
        self.wgpu.min_storage_buffer_offset_alignment
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

    /// Returns `true` if `(x,y,z)` dispatch dimensions are within per-axis limits.
    #[must_use]
    #[inline]
    pub fn compute_dispatch_fits(&self, x: u32, y: u32, z: u32) -> bool {
        let m = self.wgpu.max_compute_workgroups_per_dimension;
        x <= m && y <= m && z <= m
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_per_draw_slots_formula() {
        // Synthetic limits struct for math only (not a real device).
        let l = wgpu::Limits {
            max_storage_buffer_binding_size: 256 * 1024, // 256 KiB
            ..Default::default()
        };
        let max_binding = l.max_storage_buffer_binding_size;
        let stride = PER_DRAW_UNIFORM_STRIDE as u64;
        let slots = (max_binding / stride) as usize;
        assert_eq!(slots, 1024);
    }

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
}
