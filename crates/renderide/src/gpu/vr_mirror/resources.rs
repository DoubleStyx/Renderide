//! Persistent VR mirror state: lazy staging texture, surface uniform buffer, and
//! per-format surface pipeline cache.
//!
//! Per-frame blit logic for eye->staging and staging->surface lives in the sibling
//! [`super::eye_blit`] / [`super::surface_blit`] modules.

use super::HMD_MIRROR_SOURCE_FORMAT;
use super::pipelines::surface_pipeline;

/// GPU resources for VR mirror blit (staging texture + pipelines).
pub struct VrMirrorBlitResources {
    staging_texture: Option<wgpu::Texture>,
    staging_extent: (u32, u32),
    /// `true` after a successful eye->staging copy this session.
    staging_valid: bool,
    surface_uniform: Option<wgpu::Buffer>,
    surface_pipeline: Option<(wgpu::TextureFormat, wgpu::RenderPipeline)>,
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
            surface_uniform: None,
            surface_pipeline: None,
        }
    }

    /// `true` after [`Self::submit_eye_to_staging`] has copied at least one HMD eye image
    /// into the staging texture this session.
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

    pub(super) fn surface_uniform_buffer(&self) -> Option<&wgpu::Buffer> {
        self.surface_uniform.as_ref()
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
        if self.surface_uniform.is_some() {
            return;
        }
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vr_mirror_surface_uv"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.surface_uniform = Some(buf);
    }

    pub(super) fn surface_pipeline_for_format(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
    ) -> &wgpu::RenderPipeline {
        let entry = self
            .surface_pipeline
            .get_or_insert_with(|| (format, surface_pipeline(device, format)));
        if entry.0 != format {
            *entry = (format, surface_pipeline(device, format));
        }
        &entry.1
    }
}
