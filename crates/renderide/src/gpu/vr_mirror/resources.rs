//! Persistent VR mirror state: lazy staging texture, shared 16-byte UV uniform, and
//! per-format surface-pipeline cache.
//!
//! Per-frame blit logic for HMD final-copy and staging->surface lives in the sibling
//! [`super::eye_blit`] / [`super::surface_blit`] modules.

use crate::gpu::blit_kit::pipeline::{ColorBlitPipelineSlot, UvUniformBuffer};

use super::HMD_MIRROR_SOURCE_FORMAT;
use super::pipelines::surface_pipeline;

/// GPU resources for VR mirror blit (staging texture + pipelines).
pub struct VrMirrorBlitResources {
    staging_texture: Option<wgpu::Texture>,
    staging_extent: (u32, u32),
    /// `true` after a successful owned-eye to staging copy this session.
    staging_valid: bool,
    surface_uniform: UvUniformBuffer,
    surface_pipeline: ColorBlitPipelineSlot,
}

impl Default for VrMirrorBlitResources {
    fn default() -> Self {
        Self::new()
    }
}

impl VrMirrorBlitResources {
    /// Empty resources; staging is allocated on first successful HMD frame.
    pub fn new() -> Self {
        Self {
            staging_texture: None,
            staging_extent: (0, 0),
            staging_valid: false,
            surface_uniform: UvUniformBuffer::new(),
            surface_pipeline: ColorBlitPipelineSlot::new(),
        }
    }

    /// `true` after [`Self::encode_owned_hmd_to_openxr_and_staging`] has copied at least one HMD
    /// eye into the staging texture this session.
    pub fn staging_valid(&self) -> bool {
        self.staging_valid
    }

    pub(super) fn mark_staging_valid(&mut self) {
        self.staging_valid = true;
    }

    pub(super) fn staging_texture(&self) -> Option<&wgpu::Texture> {
        self.staging_texture.as_ref()
    }

    pub(super) fn staging_extent(&self) -> (u32, u32) {
        self.staging_extent
    }

    pub(super) fn surface_uniform(&self) -> &UvUniformBuffer {
        &self.surface_uniform
    }

    pub(super) fn ensure_staging(
        &mut self,
        device: &wgpu::Device,
        limits: &crate::gpu::GpuLimits,
        extent: (u32, u32),
    ) {
        if self.staging_extent == extent && self.staging_texture.is_some() {
            return;
        }
        let req_w = extent.0.max(1);
        let req_h = extent.1.max(1);
        let max_dim = limits.max_texture_dimension_2d();
        let w = req_w.min(max_dim);
        let h = req_h.min(max_dim);
        if (w, h) != (req_w, req_h) {
            logger::warn!(
                "vr_mirror staging: {req_w}x{req_h} exceeds max_texture_dimension_2d={max_dim}; clamped to {w}x{h}",
            );
        }
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vr_mirror_staging"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HMD_MIRROR_SOURCE_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.staging_texture = Some(tex);
        self.staging_extent = (w, h);
    }

    pub(super) fn ensure_surface_uniform(&mut self, device: &wgpu::Device) {
        self.surface_uniform.ensure(device, "vr_mirror_surface_uv");
    }

    pub(super) fn surface_pipeline_for_format(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
    ) -> &wgpu::RenderPipeline {
        self.surface_pipeline
            .get_or_build(format, |format| surface_pipeline(device, format))
    }
}
