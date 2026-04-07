//! PBR MRT pipeline: PBR lighting with G-buffer output for RTAO.
//!
//! Same bind groups and vertex layout as [`super::pbr::PbrPipeline`]; the only difference
//! is the fragment outputs three color attachments (color, position, normal) instead of one.
//! Scene bind group layout is shared via [`PbrPipeline::create_scene_bind_group_layout`].

use std::mem::size_of;

use super::super::mesh::{GpuMeshBuffers, VertexPosNormal};
use super::builder;
use super::core::{NonSkinnedUniformUpload, RenderPipeline, UniformData};
use super::pbr::PbrPipeline;
use super::ring_buffer::UniformRingBuffer;
use super::shaders::PBR_MRT_SHADER_SRC;
use super::uniforms::SceneUniforms;

/// PBR MRT pipeline: PBR lighting with three-target G-buffer output for RTAO.
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

        let bgl0 = builder::uniform_ring_bind_group_layout(device, "PBR MRT BGL 0");
        let (scene_bgl, scene_uniform_size) = PbrPipeline::create_scene_bind_group_layout(device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PBR MRT pipeline layout"),
            bind_group_layouts: &[&bgl0, &scene_bgl],
            immediate_size: 0,
        });

        let mrt_targets = builder::mrt_color_targets(config.format);
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("PBR MRT pipeline"),
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
                targets: &mrt_targets,
                compilation_options: Default::default(),
            }),
            primitive: builder::standard_primitive_state(),
            depth_stencil: Some(builder::depth_stencil_opaque()),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let uniform_ring = UniformRingBuffer::new(device, "PBR MRT uniform ring buffer");
        let bind_group =
            builder::uniform_ring_bind_group(device, "PBR MRT BG 0", &bgl0, &uniform_ring.buffer);
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
            scene_bind_group_layout: scene_bgl,
            scene_uniform_buffer,
        }
    }

    #[allow(clippy::too_many_arguments)]
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
            label: Some("PBR MRT scene BG"),
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
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

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
        self.draw_mesh_indexed(pass, buffers, None);
    }

    fn set_mesh_buffers(&self, pass: &mut wgpu::RenderPass, buffers: &GpuMeshBuffers) {
        let (vb, ib) = buffers.normal_buffers();
        pass.set_vertex_buffer(0, vb.slice(..));
        pass.set_index_buffer(ib.slice(..), buffers.index_format);
    }

    fn draw_mesh_indexed(
        &self,
        pass: &mut wgpu::RenderPass,
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
        pass: &mut wgpu::RenderPass,
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
