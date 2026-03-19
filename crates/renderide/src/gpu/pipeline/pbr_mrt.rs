//! PBR MRT pipeline: PBR shader with G-buffer output (color, position, normal) for RTAO.
//!
//! Same bind groups and vertex layout as [`super::pbr::PbrPipeline`], but fragment outputs
//! to three color attachments matching [`super::mrt::NormalDebugMRTPipeline`].

use std::mem::size_of;

use nalgebra::Matrix4;

use super::super::mesh::{GpuMeshBuffers, VertexPosNormal};
use super::core::{MAX_INSTANCE_RUN, RenderPipeline, UNIFORM_ALIGNMENT, UniformData};
use super::mrt::{MRT_NORMAL_FORMAT, MRT_POSITION_FORMAT};
use super::ring_buffer::UniformRingBuffer;
use super::shaders::PBR_MRT_SHADER_SRC;
use super::uniforms::SceneUniforms;

/// PBR MRT pipeline: PBR lighting with G-buffer output for RTAO.
pub struct PbrMRTPipeline {
    pipeline: wgpu::RenderPipeline,
    uniform_ring: UniformRingBuffer,
    bind_group: wgpu::BindGroup,
    scene_bind_group_layout: wgpu::BindGroupLayout,
    scene_uniform_buffer: wgpu::Buffer,
}

impl PbrMRTPipeline {
    /// Creates a PBR MRT pipeline with three color attachments.
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PBR MRT shader"),
            source: wgpu::ShaderSource::Wgsl(PBR_MRT_SHADER_SRC.into()),
        });

        let bind_group_layout_0 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("PBR MRT bind group 0 layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: std::num::NonZeroU64::new(
                            (MAX_INSTANCE_RUN as u64) * UNIFORM_ALIGNMENT,
                        ),
                    },
                    count: None,
                }],
            });

        let scene_uniform_size = size_of::<SceneUniforms>() as u64;
        let scene_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("PBR MRT bind group 1 layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: std::num::NonZeroU64::new(scene_uniform_size),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PBR MRT pipeline layout"),
            bind_group_layouts: &[&bind_group_layout_0, &scene_bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("PBR MRT pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<VertexPosNormal>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset: 12,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: MRT_POSITION_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: MRT_NORMAL_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Cw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::GreaterEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let uniform_ring = UniformRingBuffer::new(device, "PBR MRT uniform ring buffer");
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PBR MRT bind group 0"),
            layout: &bind_group_layout_0,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_ring.buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new((MAX_INSTANCE_RUN as u64) * UNIFORM_ALIGNMENT),
                }),
            }],
        });

        let scene_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PBR MRT scene uniform buffer"),
            size: scene_uniform_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            uniform_ring,
            bind_group,
            scene_bind_group_layout,
            scene_uniform_buffer,
        }
    }

    fn create_scene_bind_group_inner(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &SceneUniforms,
        light_buffer: &wgpu::Buffer,
        cluster_light_counts: &wgpu::Buffer,
        cluster_light_indices: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        queue.write_buffer(&self.scene_uniform_buffer, 0, bytemuck::bytes_of(scene));
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PBR MRT scene bind group"),
            layout: &self.scene_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.scene_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: light_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cluster_light_counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: cluster_light_indices.as_entire_binding(),
                },
            ],
        })
    }
}

impl RenderPipeline for PbrMRTPipeline {
    fn bind_pipeline(&self, pass: &mut wgpu::RenderPass) {
        pass.set_pipeline(&self.pipeline);
    }

    fn bind_draw(
        &self,
        pass: &mut wgpu::RenderPass,
        batch_index: Option<u32>,
        frame_index: u64,
        _draw_bind_group: Option<&wgpu::BindGroup>,
    ) {
        let dynamic_offset = batch_index
            .map(|i| self.uniform_ring.dynamic_offset(i, frame_index))
            .unwrap_or(0);
        pass.set_bind_group(0, &self.bind_group, &[dynamic_offset]);
    }

    fn bind_scene(&self, pass: &mut wgpu::RenderPass, scene_bind_group: Option<&wgpu::BindGroup>) {
        if let Some(bg) = scene_bind_group {
            pass.set_bind_group(1, bg, &[]);
        }
    }

    fn draw_mesh(
        &self,
        pass: &mut wgpu::RenderPass,
        buffers: &GpuMeshBuffers,
        _uniforms: &UniformData<'_>,
    ) {
        self.set_mesh_buffers(pass, buffers);
        self.draw_mesh_indexed(pass, buffers);
    }

    fn set_mesh_buffers(&self, pass: &mut wgpu::RenderPass, buffers: &GpuMeshBuffers) {
        let (vb, ib) = buffers.normal_buffers();
        pass.set_vertex_buffer(0, vb.slice(..));
        pass.set_index_buffer(ib.slice(..), buffers.index_format);
    }

    fn draw_mesh_indexed(&self, pass: &mut wgpu::RenderPass, buffers: &GpuMeshBuffers) {
        for &(index_start, index_count) in &buffers.draw_ranges() {
            pass.draw_indexed(index_start..index_start + index_count, 0, 0..1);
        }
    }

    fn supports_instancing(&self) -> bool {
        true
    }

    fn draw_mesh_indexed_instanced(
        &self,
        pass: &mut wgpu::RenderPass,
        buffers: &GpuMeshBuffers,
        instance_count: u32,
    ) {
        for &(index_start, index_count) in &buffers.draw_ranges() {
            pass.draw_indexed(index_start..index_start + index_count, 0, 0..instance_count);
        }
    }

    fn upload_batch(
        &self,
        queue: &wgpu::Queue,
        mvp_models: &[(Matrix4<f32>, Matrix4<f32>)],
        frame_index: u64,
    ) {
        self.uniform_ring.upload(queue, mvp_models, frame_index);
    }

    fn write_scene_uniform(&self, queue: &wgpu::Queue, scene: &[u8]) {
        queue.write_buffer(&self.scene_uniform_buffer, 0, scene);
    }

    fn create_scene_bind_group(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view_position: [f32; 3],
        cluster_count_x: u32,
        cluster_count_y: u32,
        light_count: u32,
        light_buffer: &wgpu::Buffer,
        cluster_light_counts: &wgpu::Buffer,
        cluster_light_indices: &wgpu::Buffer,
    ) -> Option<wgpu::BindGroup> {
        let scene = SceneUniforms {
            view_position,
            _pad0: 0.0,
            cluster_count_x,
            cluster_count_y,
            light_count,
            _pad1: 0,
        };
        Some(self.create_scene_bind_group_inner(
            device,
            queue,
            &scene,
            light_buffer,
            cluster_light_counts,
            cluster_light_indices,
        ))
    }
}
