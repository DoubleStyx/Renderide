//! Persistent display-blit state: per-format pipeline cache and a shared 16-byte UV uniform.
//!
//! Per-frame blit logic lives in the sibling [`super::surface_blit`] module.

use crate::gpu::blit_kit::layout::sampled_2d_filtered_uv_layout;
use crate::gpu::blit_kit::pipeline::{ColorBlitPipelineSlot, UvUniformBuffer};
use crate::gpu::resource_cache::SingleResourceSlot;

use super::pipelines::surface_pipeline;

/// GPU resources for the desktop `BlitToDisplay` pass.
///
/// Shared across frames; the only per-format reconfigure is the surface pipeline when the
/// swapchain format changes (rare, e.g. window-move HDR transition).
#[derive(Debug, Default)]
pub struct DisplayBlitResources {
    uniform: UvUniformBuffer,
    pipeline: ColorBlitPipelineSlot,
    bind_group: SingleResourceSlot<wgpu::TextureView, wgpu::BindGroup>,
}

impl DisplayBlitResources {
    /// Empty resources; the GPU buffer and pipeline are lazily created on first blit.
    pub fn new() -> Self {
        Self::default()
    }

    pub(super) fn uniform(&self) -> &UvUniformBuffer {
        &self.uniform
    }

    pub(super) fn ensure_uniform(&mut self, device: &wgpu::Device) {
        self.uniform.ensure(device, "display_blit_uv");
    }

    pub(super) fn pipeline_for_format(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
    ) -> &wgpu::RenderPipeline {
        self.pipeline
            .get_or_build(format, |format| surface_pipeline(device, format))
    }

    /// Returns a sampled-source bind group for the current blit source, rebuilding on view change.
    pub(super) fn bind_group_for_source(
        &mut self,
        device: &wgpu::Device,
        label: &'static str,
        view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        uniform_buf: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.bind_group.get_or_build(view.clone(), || {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: sampled_2d_filtered_uv_layout(device),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: uniform_buf.as_entire_binding(),
                    },
                ],
            });
            crate::profiling::note_resource_churn!(BindGroup, "gpu::display_blit_bind_group");
            bind_group
        })
    }
}
