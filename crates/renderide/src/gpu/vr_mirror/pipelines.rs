//! Pipeline builders for owned-HMD-to-OpenXR and VR mirror staging-to-surface blits.
//!
//! Bind-group layouts and the linear-clamp sampler are shared with [`crate::gpu::display_blit`]
//! via [`crate::gpu::blit_kit`]. HMD-copy pipelines target [`HMD_MIRROR_SOURCE_FORMAT`]; the
//! surface pipeline targets the active desktop swapchain format.

use std::num::NonZeroU32;
use std::sync::OnceLock;

use crate::embedded_shaders::embedded_wgsl;
use crate::gpu::blit_kit::layout::{
    sampled_2d_array_filtered_layout, sampled_2d_filtered_layout, sampled_2d_filtered_uv_layout,
};
use crate::gpu::blit_kit::pipeline::{
    color_blit_pipeline, color_blit_pipeline_with_multiview_mask,
};

use super::HMD_MIRROR_SOURCE_FORMAT;

/// Pipeline that samples owned stereo HMD color into the OpenXR swapchain array.
pub(super) fn openxr_multiview_pipeline(device: &wgpu::Device) -> &'static wgpu::RenderPipeline {
    static PIPE: OnceLock<wgpu::RenderPipeline> = OnceLock::new();
    PIPE.get_or_init(|| {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vr_mirror_eye_to_openxr_multiview"),
            source: wgpu::ShaderSource::Wgsl(
                embedded_wgsl!("vr_mirror_eye_to_openxr_multiview").into(),
            ),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vr_mirror_eye_to_openxr_multiview"),
            bind_group_layouts: &[Some(sampled_2d_array_filtered_layout(device))],
            immediate_size: 0,
        });
        let pipeline = color_blit_pipeline_with_multiview_mask(
            device,
            &shader,
            &layout,
            "vr_mirror_eye_to_openxr_multiview",
            HMD_MIRROR_SOURCE_FORMAT,
            NonZeroU32::new(3),
        );
        crate::profiling::note_resource_churn!(
            RenderPipeline,
            "gpu::vr_mirror_openxr_multiview_pipeline"
        );
        pipeline
    })
}

/// Pipeline that samples the owned left-eye view into the desktop mirror staging texture.
pub(super) fn eye_pipeline(device: &wgpu::Device) -> &'static wgpu::RenderPipeline {
    static PIPE: OnceLock<wgpu::RenderPipeline> = OnceLock::new();
    PIPE.get_or_init(|| {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vr_mirror_eye_to_staging"),
            source: wgpu::ShaderSource::Wgsl(embedded_wgsl!("vr_mirror_eye_to_staging").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vr_mirror_eye_to_staging"),
            bind_group_layouts: &[Some(sampled_2d_filtered_layout(device))],
            immediate_size: 0,
        });
        let pipeline = color_blit_pipeline(
            device,
            &shader,
            &layout,
            "vr_mirror_eye_to_staging",
            HMD_MIRROR_SOURCE_FORMAT,
        );
        crate::profiling::note_resource_churn!(RenderPipeline, "gpu::vr_mirror_eye_pipeline");
        pipeline
    })
}

/// Pipeline that copies mirror staging into the active desktop surface format.
pub(super) fn surface_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("vr_mirror_surface"),
        source: wgpu::ShaderSource::Wgsl(embedded_wgsl!("vr_mirror_surface").into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("vr_mirror_surface"),
        bind_group_layouts: &[Some(sampled_2d_filtered_uv_layout(device))],
        immediate_size: 0,
    });
    let pipeline = color_blit_pipeline(device, &shader, &layout, "vr_mirror_surface", format);
    crate::profiling::note_resource_churn!(RenderPipeline, "gpu::vr_mirror_surface_pipeline");
    pipeline
}
