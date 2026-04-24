//! Compute shaders for mesh skinning and blendshape deformation (frame-begin preprocess hook).
//!
//! Pipelines compile when [`crate::backend::RenderBackend::attach`] runs (via
//! [`crate::runtime::RendererRuntime::attach_gpu`]). Full vertex layout binding
//! to [`crate::assets::mesh::GpuMesh`] interleaved buffers is a follow-up; this module establishes
//! bind group contracts and valid WGSL entry points.

use std::num::NonZeroU64;

use wgpu::util::DeviceExt;

use crate::embedded_shaders::embedded_target_wgsl;

/// Embedded shader stem for the mesh skinning compute pass (single-variant).
const SKINNING_STEM: &str = "mesh_skinning";
/// Embedded shader stem for the blendshape compute pass (single-variant).
const BLENDSHAPE_STEM: &str = "mesh_blendshape";

fn load_embedded(stem: &str) -> &'static str {
    #[expect(
        clippy::expect_used,
        reason = "embedded shader is required; absence is a build script regression"
    )]
    embedded_target_wgsl(stem)
        .expect("mesh_preprocess: embedded shader missing (build script regression)")
}

fn storage_buffer_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn skinning_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mesh_skinning"),
        entries: &[
            storage_buffer_entry(0, true),
            storage_buffer_entry(1, true),
            storage_buffer_entry(2, true),
            storage_buffer_entry(3, true),
            storage_buffer_entry(4, false),
            storage_buffer_entry(5, true),
            storage_buffer_entry(6, false),
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(32),
                },
                count: None,
            },
        ],
    })
}

fn blendshape_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mesh_blendshape_scatter"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(32),
                },
                count: None,
            },
            storage_buffer_entry(1, true),
            storage_buffer_entry(2, true),
            storage_buffer_entry(3, false),
        ],
    })
}

/// Built compute pipelines for mesh deformation (skinning + blendshapes).
pub struct MeshPreprocessPipelines {
    /// [`Self::skinning_bind_group_layout`] for bind group 0.
    pub skinning_bind_group_layout: wgpu::BindGroupLayout,
    /// Skinning LBS compute dispatch pipeline.
    pub skinning_pipeline: wgpu::ComputePipeline,
    /// Bind group layout for blendshape compute (`@group(0)`).
    pub blendshape_bind_group_layout: wgpu::BindGroupLayout,
    /// Blendshape delta apply compute pipeline.
    pub blendshape_pipeline: wgpu::ComputePipeline,
}

impl MeshPreprocessPipelines {
    /// Compiles WGSL and creates compute pipelines. Fails if shader validation errors.
    pub fn new(device: &wgpu::Device) -> Result<Self, String> {
        let skin_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh_skinning"),
            source: wgpu::ShaderSource::Wgsl(load_embedded(SKINNING_STEM).into()),
        });
        let blend_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh_blendshape"),
            source: wgpu::ShaderSource::Wgsl(load_embedded(BLENDSHAPE_STEM).into()),
        });

        let skin_bgl = skinning_bind_group_layout(device);
        let skin_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh_skinning_layout"),
            bind_group_layouts: &[Some(&skin_bgl)],
            immediate_size: 0,
        });
        let skinning_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mesh_skinning_pipeline"),
            layout: Some(&skin_layout),
            module: &skin_shader,
            entry_point: Some("skin_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let blend_bgl = blendshape_bind_group_layout(device);
        let blend_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh_blendshape_layout"),
            bind_group_layouts: &[Some(&blend_bgl)],
            immediate_size: 0,
        });
        let blendshape_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("mesh_blendshape_scatter_pipeline"),
                layout: Some(&blend_layout),
                module: &blend_shader,
                entry_point: Some("blendshape_scatter_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Ok(Self {
            skinning_bind_group_layout: skin_bgl,
            skinning_pipeline,
            blendshape_bind_group_layout: blend_bgl,
            blendshape_pipeline,
        })
    }

    /// Records no drawing; reserved for future per-frame preprocess encode from [`crate::scene`].
    pub fn placeholder_uniform_slice(device: &wgpu::Device) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_preprocess_placeholder_uniform"),
            contents: &[0u8; 32],
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }
}
