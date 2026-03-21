//! PBR pipeline: PBS metallic shader with Cook-Torrance BRDF and clustered lighting.
//!
//! Uses `VertexPosNormal` vertex format (same as NormalDebug). Bind group 0 is the standard
//! dynamic uniform ring (MVP/model per draw); bind group 1 holds scene uniforms, the light
//! storage buffer, and the two cluster buffers.

use std::mem::size_of;

use nalgebra::Matrix4;

use super::super::mesh::{GpuMeshBuffers, VertexPosNormal};
use super::builder;
use super::core::{MAX_INSTANCE_RUN, RenderPipeline, UNIFORM_ALIGNMENT, UniformData};
use super::ring_buffer::UniformRingBuffer;
use super::shaders::PBR_SHADER_SRC;
use super::uniforms::SceneUniforms;

/// PBR pipeline with Cook-Torrance BRDF and clustered lighting (single color target).
pub struct PbrPipeline {
    pipeline: wgpu::RenderPipeline,
    uniform_ring: UniformRingBuffer,
    bind_group: wgpu::BindGroup,
    scene_bind_group_layout: wgpu::BindGroupLayout,
    scene_uniform_buffer: wgpu::Buffer,
}

impl PbrPipeline {
    /// Creates a PBR pipeline.
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PBR shader"),
            source: wgpu::ShaderSource::Wgsl(PBR_SHADER_SRC.into()),
        });

        let bgl0 = builder::uniform_ring_bind_group_layout(device, "PBR BGL 0");
        let (scene_bgl, scene_uniform_size) = Self::create_scene_bind_group_layout(device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PBR pipeline layout"),
            bind_group_layouts: &[&bgl0, &scene_bgl],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("PBR pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: size_of::<VertexPosNormal>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &builder::POS_NORMAL_ATTRIBS,
                }],
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

        let uniform_ring = UniformRingBuffer::new(device, "PBR uniform ring buffer");
        let bind_group =
            builder::uniform_ring_bind_group(device, "PBR BG 0", &bgl0, &uniform_ring.buffer);
        let scene_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PBR scene uniform buffer"),
            size: scene_uniform_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            uniform_ring,
            bind_group,
            scene_bind_group_layout: scene_bgl,
            scene_uniform_buffer,
        }
    }

    /// Creates the scene bind group layout (group 1) shared by PBR and PBR-MRT.
    ///
    /// Returns `(layout, scene_uniform_size_bytes)`.
    pub(crate) fn create_scene_bind_group_layout(
        device: &wgpu::Device,
    ) -> (wgpu::BindGroupLayout, u64) {
        let scene_uniform_size = size_of::<SceneUniforms>() as u64;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PBR scene BGL"),
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
        (layout, scene_uniform_size)
    }

    fn create_scene_bind_group_inner(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &SceneUniforms,
        light_buffer: &wgpu::Buffer,
        cluster_light_counts: &wgpu::Buffer,
        cluster_light_indices: &wgpu::Buffer,
        _acceleration_structure: Option<&wgpu::Tlas>,
    ) -> wgpu::BindGroup {
        queue.write_buffer(&self.scene_uniform_buffer, 0, bytemuck::bytes_of(scene));
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PBR scene BG"),
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

impl RenderPipeline for PbrPipeline {
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
        view_space_z_coeffs: [f32; 4],
        cluster_count_x: u32,
        cluster_count_y: u32,
        cluster_count_z: u32,
        near_clip: f32,
        far_clip: f32,
        light_count: u32,
        viewport_width: u32,
        viewport_height: u32,
        light_buffer: &wgpu::Buffer,
        cluster_light_counts: &wgpu::Buffer,
        cluster_light_indices: &wgpu::Buffer,
        acceleration_structure: Option<&wgpu::Tlas>,
        _rt_shadow: Option<super::rt_shadow_uniforms::RtShadowSceneBind<'_>>,
    ) -> Option<wgpu::BindGroup> {
        let scene = SceneUniforms {
            view_position,
            _pad0: 0.0,
            view_space_z_coeffs,
            cluster_count_x,
            cluster_count_y,
            cluster_count_z,
            near_clip,
            far_clip,
            light_count,
            viewport_width,
            viewport_height,
        };
        Some(self.create_scene_bind_group_inner(
            device,
            queue,
            &scene,
            light_buffer,
            cluster_light_counts,
            cluster_light_indices,
            acceleration_structure,
        ))
    }
}

// Ensure UNIFORM_ALIGNMENT / MAX_INSTANCE_RUN remain used via builder helpers.
const _: () = {
    let _ = MAX_INSTANCE_RUN;
    let _ = UNIFORM_ALIGNMENT;
};
