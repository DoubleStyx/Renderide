//! Lazy compute pipeline creation for IBL bake passes.

use std::borrow::Cow;

use crate::embedded_shaders;

use super::SkyboxIblBakeError;
use super::resources::IBL_CUBE_FORMAT;

/// Compute pipeline + bind-group layout pair built lazily from an embedded shader stem.
pub(super) struct ComputePipeline {
    /// Compute pipeline.
    pub(super) pipeline: wgpu::ComputePipeline,
    /// Bind-group layout for this pipeline.
    pub(super) layout: wgpu::BindGroupLayout,
}

/// Storage texture entry for one cubemap mip.
pub(super) fn storage_texture_layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: IBL_CUBE_FORMAT,
            view_dimension: wgpu::TextureViewDimension::D2Array,
        },
        count: None,
    }
}

/// Bind-group layout entries for the analytic mip-0 producer.
pub(super) fn analytic_layout_entries() -> [wgpu::BindGroupLayoutEntry; 2] {
    [
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        storage_texture_layout_entry(1),
    ]
}

/// Bind-group layout entries for the cube/equirect/convolve passes that read a sampled texture.
pub(super) fn mip0_input_layout_entries(
    input_dim: wgpu::TextureViewDimension,
) -> [wgpu::BindGroupLayoutEntry; 4] {
    [
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: input_dim,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
        storage_texture_layout_entry(3),
    ]
}

/// Lazily creates and caches a compute pipeline from an embedded shader stem.
pub(super) fn ensure_pipeline<'a>(
    slot: &'a mut Option<ComputePipeline>,
    device: &wgpu::Device,
    stem: &'static str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> Result<&'a ComputePipeline, SkyboxIblBakeError> {
    if slot.is_none() {
        profiling::scope!("skybox_ibl::create_pipeline", stem);
        let source = embedded_shaders::embedded_target_wgsl(stem)
            .ok_or(SkyboxIblBakeError::MissingShader(stem))?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(stem),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{stem} bind group layout")),
            entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{stem} pipeline layout")),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(stem),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        *slot = Some(ComputePipeline { pipeline, layout });
    }
    slot.as_ref().ok_or(SkyboxIblBakeError::MissingShader(stem))
}
