//! Native WGSL world `Shader "Unlit"` ([`crate::assets::CANONICAL_UNITY_WORLD_UNLIT`]): uniform ring (group 0)
//! and material uniforms + `_Tex` / `_MaskTex` (group 1). Uses [`crate::gpu::mesh::VertexPosNormalUv`].

use std::mem::size_of;

use wgpu::util::DeviceExt;

use crate::assets::WorldUnlitMaterialUniform;

use super::super::mesh::{GpuMeshBuffers, VertexPosNormalUv};
use super::builder;
use super::core::{NonSkinnedUniformUpload, RenderPipeline};
use super::ring_buffer::UniformRingBuffer;
use super::ui_unlit_native::fallback_white;

const WORLD_UNLIT_WGSL: &str = include_str!(concat!(env!("OUT_DIR"), "/world_unlit.wgsl"));

/// Resonite world `Shader "Unlit"` forward mesh pipeline (non-UI, depth-tested).
pub struct WorldUnlitPipeline {
    pipeline: wgpu::RenderPipeline,
    uniform_ring: UniformRingBuffer,
    ring_bind_group: wgpu::BindGroup,
    material_uniform: wgpu::Buffer,
    material_bgl: wgpu::BindGroupLayout,
    linear_sampler: wgpu::Sampler,
    /// Placeholder bind group so the pipeline layout validates; per-draw binds come from [`crate::gpu::NativeUiMaterialBindCache::write_world_unlit_material_bind`].
    _material_bind_group: wgpu::BindGroup,
}

impl WorldUnlitPipeline {
    /// Builds the world-unlit pipeline for the swapchain format.
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("world unlit native shader"),
            source: wgpu::ShaderSource::Wgsl(WORLD_UNLIT_WGSL.into()),
        });
        let ring_bgl = builder::uniform_ring_bind_group_layout(device, "world unlit ring BGL");
        let material_bgl = create_world_unlit_material_bind_group_layout(device);
        let white = fallback_white(device);
        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("world unlit linear sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let initial = WorldUnlitMaterialUniform {
            color: [1.0, 1.0, 1.0, 1.0],
            tex_st: [1.0, 1.0, 0.0, 0.0],
            mask_tex_st: [1.0, 1.0, 0.0, 0.0],
            cutoff: 0.5,
            flags: 0,
            pad_tail: [0; 2],
        };
        let material_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("world unlit material uniform"),
            contents: bytemuck::bytes_of(&initial),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let material_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("world unlit material BG placeholder"),
            layout: &material_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: material_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(white),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(white),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&linear_sampler),
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("world unlit native PL"),
            bind_group_layouts: &[&ring_bgl, &material_bgl],
            immediate_size: 0,
        });
        let vb_layout = wgpu::VertexBufferLayout {
            array_stride: size_of::<VertexPosNormalUv>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &builder::POS_NORMAL_UV_ATTRIBS,
        };
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("world unlit native RP"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vb_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(builder::standard_color_target(config.format))],
                compilation_options: Default::default(),
            }),
            primitive: builder::standard_primitive_state(),
            depth_stencil: Some(builder::depth_stencil_opaque()),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let uniform_ring = UniformRingBuffer::new(device, "world unlit native ring");
        let ring_bind_group = builder::uniform_ring_bind_group(
            device,
            "world unlit native ring BG",
            &ring_bgl,
            &uniform_ring.buffer,
        );
        Self {
            pipeline,
            uniform_ring,
            ring_bind_group,
            material_uniform,
            material_bgl,
            linear_sampler,
            _material_bind_group: material_bind_group,
        }
    }

    /// Bind group layout for material uniforms + textures (group 1).
    pub fn material_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.material_bgl
    }

    /// Linear sampler for `_Tex` / `_MaskTex`.
    pub fn linear_sampler(&self) -> &wgpu::Sampler {
        &self.linear_sampler
    }

    /// Uniform buffer written each draw for material properties.
    pub fn material_uniform_buffer(&self) -> &wgpu::Buffer {
        &self.material_uniform
    }
}

fn create_world_unlit_material_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("world unlit material BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<
                        WorldUnlitMaterialUniform,
                    >() as u64),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

impl RenderPipeline for WorldUnlitPipeline {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn bind_pipeline(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(&self.pipeline);
    }

    fn bind_draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        batch_index: Option<u32>,
        frame_index: u64,
        _draw_bind_group: Option<&wgpu::BindGroup>,
    ) {
        let dynamic_offset = batch_index
            .map(|i| self.uniform_ring.dynamic_offset(i, frame_index))
            .unwrap_or(0);
        pass.set_bind_group(0, &self.ring_bind_group, &[dynamic_offset]);
    }

    fn set_mesh_buffers(&self, pass: &mut wgpu::RenderPass<'_>, buffers: &GpuMeshBuffers) {
        let (vb, ib) = buffers.pos_normal_uv_buffers().expect(
            "create_mesh_buffers always uploads pos+normal+uv (default uv when mesh has no UV0)",
        );
        pass.set_vertex_buffer(0, vb.slice(..));
        pass.set_index_buffer(ib.slice(..), buffers.index_format);
    }

    fn draw_mesh_indexed(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        buffers: &GpuMeshBuffers,
        index_range_override: Option<(u32, u32)>,
    ) {
        for &(index_start, index_count) in &buffers.effective_draw_ranges(index_range_override) {
            pass.draw_indexed(index_start..index_start + index_count, 0, 0..1);
        }
    }

    fn supports_instancing(&self) -> bool {
        true
    }

    fn draw_mesh_indexed_instanced(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        buffers: &GpuMeshBuffers,
        instance_count: u32,
        index_range_override: Option<(u32, u32)>,
    ) {
        for &(index_start, index_count) in &buffers.effective_draw_ranges(index_range_override) {
            pass.draw_indexed(index_start..index_start + index_count, 0, 0..instance_count);
        }
    }

    fn upload_batch(
        &self,
        queue: &wgpu::Queue,
        draws: &[NonSkinnedUniformUpload],
        frame_index: u64,
    ) {
        self.uniform_ring.upload(queue, draws, frame_index);
    }
}
