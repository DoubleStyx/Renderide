//! Persistent display-blit state: per-format pipeline cache and a shared 16-byte UV uniform.
//!
//! Per-frame blit logic lives in the sibling [`super::surface_blit`] module.

use crate::gpu::blit_kit::pipeline::{ColorBlitPipelineSlot, UvUniformBuffer};

use super::pipelines::{overlay_pipeline, surface_pipeline};

/// GPU resources for the desktop `BlitToDisplay` pass.
///
/// Shared across frames; the only per-format reconfigure is the surface pipeline when the
/// swapchain format changes (rare, e.g. window-move HDR transition).
#[derive(Debug, Default)]
pub struct DisplayBlitResources {
    uniform: UvUniformBuffer,
    overlay_uniform: UvUniformBuffer,
    pipeline: ColorBlitPipelineSlot,
    overlay_pipeline: ColorBlitPipelineSlot,
}

impl DisplayBlitResources {
    /// Empty resources; the GPU buffer and pipeline are lazily created on first blit.
    pub fn new() -> Self {
        Self::default()
    }

    pub(super) fn uniform(&self) -> &UvUniformBuffer {
        &self.uniform
    }

    pub(super) fn overlay_uniform(&self) -> &UvUniformBuffer {
        &self.overlay_uniform
    }

    pub(super) fn ensure_uniform(&mut self, device: &wgpu::Device) {
        self.uniform.ensure(device, "display_blit_uv");
    }

    pub(super) fn ensure_overlay_uniform(&mut self, device: &wgpu::Device) {
        self.overlay_uniform
            .ensure(device, "display_blit_overlay_uv");
    }

    pub(super) fn pipeline_for_format(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
    ) -> &wgpu::RenderPipeline {
        self.pipeline
            .get_or_build(format, |format| surface_pipeline(device, format))
    }

    pub(super) fn overlay_pipeline_for_format(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
    ) -> &wgpu::RenderPipeline {
        self.overlay_pipeline
            .get_or_build(format, |format| overlay_pipeline(device, format))
    }
}
