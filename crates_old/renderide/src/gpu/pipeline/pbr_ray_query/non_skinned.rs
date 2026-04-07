//! Non-skinned PBR ray-query pipelines: single color target and MRT variants.

use std::mem::size_of;

use super::super::builder;
use super::super::core::{NonSkinnedUniformUpload, RenderPipeline, UniformData};
use super::super::ring_buffer::UniformRingBuffer;
use super::super::rt_shadow_uniforms::RtShadowSceneBind;
use super::super::shaders::{PBR_MRT_RAY_QUERY_SHADER_SRC, PBR_RAY_QUERY_SHADER_SRC};
use super::super::uniforms::SceneUniforms;
use super::scene::{
    create_pbr_scene_bind_group_with_accel, pbr_scene_bind_group_layout_with_accel,
};
use crate::gpu::mesh::{GpuMeshBuffers, VertexPosNormal};

/// Non-skinned PBR with ray-traced shadows (single color target).
pub struct PbrRayQueryPipeline {
    pipeline: wgpu::RenderPipeline,
    uniform_ring: UniformRingBuffer,
    bind_group: wgpu::BindGroup,
    scene_bind_group_layout: wgpu::BindGroupLayout,
    scene_uniform_buffer: wgpu::Buffer,
}

impl PbrRayQueryPipeline {
    /// Builds the pipeline. Requires a device created with [`wgpu::Features::EXPERIMENTAL_RAY_QUERY`].
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PBR ray query shader"),
            source: wgpu::ShaderSource::Wgsl(PBR_RAY_QUERY_SHADER_SRC.into()),
        });
        let bgl0 = builder::uniform_ring_bind_group_layout(device, "PBR ray query BGL 0");
        let (scene_bgl, scene_uniform_size) = pbr_scene_bind_group_layout_with_accel(device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PBR ray query pipeline layout"),
            bind_group_layouts: &[&bgl0, &scene_bgl],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("PBR ray query pipeline"),
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
        let uniform_ring = UniformRingBuffer::new(device, "PBR ray query uniform ring buffer");
        let bind_group = builder::uniform_ring_bind_group(
            device,
            "PBR ray query BG 0",
            &bgl0,
            &uniform_ring.buffer,
        );
        let scene_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PBR ray query scene uniform buffer"),
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
}

impl RenderPipeline for PbrRayQueryPipeline {
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
        rt_shadow: Option<RtShadowSceneBind<'_>>,
    ) -> Option<wgpu::BindGroup> {
        let tlas = acceleration_structure?;
        let rs = rt_shadow?;
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
        Some(create_pbr_scene_bind_group_with_accel(
            device,
            queue,
            &self.scene_bind_group_layout,
            "PBR ray query scene BG + TLAS + RT shadow",
            &self.scene_uniform_buffer,
            &scene,
            light_buffer,
            cluster_light_counts,
            cluster_light_indices,
            tlas,
            &rs,
        ))
    }
}

/// Non-skinned PBR MRT with ray-traced shadows.
pub struct PbrMrtRayQueryPipeline {
    pipeline: wgpu::RenderPipeline,
    uniform_ring: UniformRingBuffer,
    bind_group: wgpu::BindGroup,
    scene_bind_group_layout: wgpu::BindGroupLayout,
    scene_uniform_buffer: wgpu::Buffer,
}

impl PbrMrtRayQueryPipeline {
    /// Builds the pipeline. Requires [`wgpu::Features::EXPERIMENTAL_RAY_QUERY`].
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PBR MRT ray query shader"),
            source: wgpu::ShaderSource::Wgsl(PBR_MRT_RAY_QUERY_SHADER_SRC.into()),
        });
        let bgl0 = builder::uniform_ring_bind_group_layout(device, "PBR MRT ray query BGL 0");
        let (scene_bgl, scene_uniform_size) = pbr_scene_bind_group_layout_with_accel(device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PBR MRT ray query pipeline layout"),
            bind_group_layouts: &[&bgl0, &scene_bgl],
            immediate_size: 0,
        });
        let mrt_targets = builder::mrt_color_targets(config.format);
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("PBR MRT ray query pipeline"),
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
        let uniform_ring = UniformRingBuffer::new(device, "PBR MRT ray query uniform ring buffer");
        let bind_group = builder::uniform_ring_bind_group(
            device,
            "PBR MRT ray query BG 0",
            &bgl0,
            &uniform_ring.buffer,
        );
        let scene_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PBR MRT ray query scene uniform buffer"),
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
}

impl RenderPipeline for PbrMrtRayQueryPipeline {
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
        rt_shadow: Option<RtShadowSceneBind<'_>>,
    ) -> Option<wgpu::BindGroup> {
        let tlas = acceleration_structure?;
        let rs = rt_shadow?;
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
        Some(create_pbr_scene_bind_group_with_accel(
            device,
            queue,
            &self.scene_bind_group_layout,
            "PBR MRT ray query scene BG + TLAS + RT shadow",
            &self.scene_uniform_buffer,
            &scene,
            light_buffer,
            cluster_light_counts,
            cluster_light_indices,
            tlas,
            &rs,
        ))
    }
}
