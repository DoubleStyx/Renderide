//! Cached shader + per-format pipeline construction for the display-blit pass.

use crate::embedded_shaders::embedded_wgsl;
use crate::gpu::blit_kit::layout::sampled_2d_filtered_uv_layout;
use crate::gpu::blit_kit::pipeline::color_blit_pipeline;

/// Builds the fragment-output-format-specific pipeline for the display blit pass.
pub(super) fn surface_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("display_blit"),
        source: wgpu::ShaderSource::Wgsl(embedded_wgsl!("display_blit").into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("display_blit"),
        bind_group_layouts: &[Some(sampled_2d_filtered_uv_layout(device))],
        immediate_size: 0,
    });
    let pipeline = color_blit_pipeline(device, &shader, &layout, "display_blit", format);
    crate::profiling::note_resource_churn!(RenderPipeline, "gpu::display_blit_pipeline");
    pipeline
}
