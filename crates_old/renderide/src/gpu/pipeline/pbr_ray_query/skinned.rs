//! Skinned PBR ray-query pipelines: single color target and MRT variants, with per-mesh draw bind groups.

use std::mem::size_of;

use nalgebra::Matrix4;

use super::super::builder;
use super::super::core::{RenderPipeline, UniformData};
use super::super::ring_buffer::SkinnedUniformRingBuffer;
use super::super::rt_shadow_uniforms::RtShadowSceneBind;
use super::super::shaders::{
    SKINNED_PBR_MRT_RAY_QUERY_SHADER_SRC, SKINNED_PBR_RAY_QUERY_SHADER_SRC,
};
use super::super::skinned_pbr::{SkinnedPbrPipeline, create_skinned_draw_bg};
use super::super::uniforms::SceneUniforms;
use super::scene::{
    create_pbr_scene_bind_group_with_accel, pbr_scene_bind_group_layout_with_accel,
};
use crate::gpu::mesh::{GpuMeshBuffers, VertexSkinned};

fn skinned_pbr_ray_query_bind_group_layouts(
    device: &wgpu::Device,
) -> (wgpu::BindGroupLayout, wgpu::BindGroupLayout, u64) {
    let (draw_bgl, _, scene_uniform_size) = SkinnedPbrPipeline::create_bind_group_layouts(device);
    let (scene_bgl, _) = pbr_scene_bind_group_layout_with_accel(device);
    (draw_bgl, scene_bgl, scene_uniform_size)
}

/// Forwards to [`create_pbr_scene_bind_group_with_accel`] with skinned pipeline scene layout.
#[allow(clippy::too_many_arguments)]
fn create_skinned_scene_bind_group_with_accel(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    scene_uniform_buffer: &wgpu::Buffer,
    scene: &SceneUniforms,
    queue: &wgpu::Queue,
    light_buffer: &wgpu::Buffer,
    cluster_light_counts: &wgpu::Buffer,
    cluster_light_indices: &wgpu::Buffer,
    tlas: &wgpu::Tlas,
    rt_shadow: &RtShadowSceneBind<'_>,
) -> wgpu::BindGroup {
    create_pbr_scene_bind_group_with_accel(
        device,
        queue,
        layout,
        label,
        scene_uniform_buffer,
        scene,
        light_buffer,
        cluster_light_counts,
        cluster_light_indices,
        tlas,
        rt_shadow,
    )
}

/// Skinned PBR with ray-traced shadows (single color target).
pub struct SkinnedPbrRayQueryPipeline {
    pipeline: wgpu::RenderPipeline,
    uniform_ring: SkinnedUniformRingBuffer,
    bind_group_layout: wgpu::BindGroupLayout,
    dummy_blendshape_buffer: wgpu::Buffer,
    scene_bind_group_layout: wgpu::BindGroupLayout,
    scene_uniform_buffer: wgpu::Buffer,
}

impl SkinnedPbrRayQueryPipeline {
    /// Builds the pipeline. Requires [`wgpu::Features::EXPERIMENTAL_RAY_QUERY`].
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("skinned PBR ray query shader"),
            source: wgpu::ShaderSource::Wgsl(SKINNED_PBR_RAY_QUERY_SHADER_SRC.into()),
        });
        let (draw_bgl, scene_bgl, scene_uniform_size) =
            skinned_pbr_ray_query_bind_group_layouts(device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("skinned PBR ray query pipeline layout"),
            bind_group_layouts: &[&draw_bgl, &scene_bgl],
            immediate_size: 0,
        });
        let uniform_ring =
            SkinnedUniformRingBuffer::new(device, "skinned PBR ray query uniform ring buffer");
        let dummy_blendshape_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skinned PBR ray query dummy blendshape buffer"),
            size: 1,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let scene_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skinned PBR ray query scene uniform buffer"),
            size: scene_uniform_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("skinned PBR ray query pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: size_of::<VertexSkinned>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &builder::SKINNED_ATTRIBS,
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
        Self {
            pipeline,
            uniform_ring,
            bind_group_layout: draw_bgl,
            dummy_blendshape_buffer,
            scene_bind_group_layout: scene_bgl,
            scene_uniform_buffer,
        }
    }
}

impl RenderPipeline for SkinnedPbrRayQueryPipeline {
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
        draw_bind_group: Option<&wgpu::BindGroup>,
    ) {
        let dynamic_offset = batch_index
            .map(|i| self.uniform_ring.dynamic_offset(i, frame_index))
            .unwrap_or(0);
        let bg = draw_bind_group.expect("skinned PBR ray query requires draw_bind_group");
        pass.set_bind_group(0, bg, &[dynamic_offset]);
    }

    fn bind_scene(&self, pass: &mut wgpu::RenderPass, scene_bind_group: Option<&wgpu::BindGroup>) {
        if let Some(bg) = scene_bind_group {
            pass.set_bind_group(1, bg, &[]);
        }
    }

    fn create_skinned_draw_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuMeshBuffers,
    ) -> Option<wgpu::BindGroup> {
        let blendshape = buffers
            .blendshape_buffer
            .as_ref()
            .map(|b| b.as_ref())
            .unwrap_or(&self.dummy_blendshape_buffer);
        Some(create_skinned_draw_bg(
            device,
            "skinned PBR ray query draw BG",
            &self.bind_group_layout,
            &self.uniform_ring.buffer,
            blendshape,
        ))
    }

    fn draw_skinned(
        &self,
        pass: &mut wgpu::RenderPass,
        buffers: &GpuMeshBuffers,
        _uniforms: &UniformData<'_>,
    ) {
        self.set_skinned_buffers(pass, buffers);
        self.draw_skinned_indexed(pass, buffers, None);
    }

    fn set_skinned_buffers(&self, pass: &mut wgpu::RenderPass, buffers: &GpuMeshBuffers) {
        let Some((vb, ib)) = buffers.skinned_buffers() else {
            return;
        };
        pass.set_vertex_buffer(0, vb.slice(..));
        pass.set_index_buffer(ib.slice(..), buffers.index_format);
    }

    fn draw_skinned_indexed(
        &self,
        pass: &mut wgpu::RenderPass,
        buffers: &GpuMeshBuffers,
        index_range_override: Option<(u32, u32)>,
    ) {
        for &(index_start, index_count) in &buffers.effective_draw_ranges(index_range_override) {
            pass.draw_indexed(index_start..index_start + index_count, 0, 0..1);
        }
    }

    fn upload_skinned_batch(
        &self,
        queue: &wgpu::Queue,
        items: &[(Matrix4<f32>, &[[[f32; 4]; 4]], Option<&[f32]>, u32)],
        frame_index: u64,
    ) {
        self.uniform_ring.upload(queue, items, frame_index);
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
        Some(create_skinned_scene_bind_group_with_accel(
            device,
            "skinned PBR ray query scene BG + RT shadow",
            &self.scene_bind_group_layout,
            &self.scene_uniform_buffer,
            &scene,
            queue,
            light_buffer,
            cluster_light_counts,
            cluster_light_indices,
            tlas,
            &rs,
        ))
    }
}

/// Skinned PBR MRT with ray-traced shadows.
pub struct SkinnedPbrMrtRayQueryPipeline {
    pipeline: wgpu::RenderPipeline,
    uniform_ring: SkinnedUniformRingBuffer,
    bind_group_layout: wgpu::BindGroupLayout,
    dummy_blendshape_buffer: wgpu::Buffer,
    scene_bind_group_layout: wgpu::BindGroupLayout,
    scene_uniform_buffer: wgpu::Buffer,
}

impl SkinnedPbrMrtRayQueryPipeline {
    /// Builds the pipeline. Requires [`wgpu::Features::EXPERIMENTAL_RAY_QUERY`].
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("skinned PBR MRT ray query shader"),
            source: wgpu::ShaderSource::Wgsl(SKINNED_PBR_MRT_RAY_QUERY_SHADER_SRC.into()),
        });
        let (draw_bgl, scene_bgl, scene_uniform_size) =
            skinned_pbr_ray_query_bind_group_layouts(device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("skinned PBR MRT ray query pipeline layout"),
            bind_group_layouts: &[&draw_bgl, &scene_bgl],
            immediate_size: 0,
        });
        let uniform_ring =
            SkinnedUniformRingBuffer::new(device, "skinned PBR MRT ray query uniform ring buffer");
        let dummy_blendshape_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skinned PBR MRT ray query dummy blendshape buffer"),
            size: 1,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let scene_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skinned PBR MRT ray query scene uniform buffer"),
            size: scene_uniform_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mrt_targets = builder::mrt_color_targets(config.format);
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("skinned PBR MRT ray query pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: size_of::<VertexSkinned>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &builder::SKINNED_ATTRIBS,
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
        Self {
            pipeline,
            uniform_ring,
            bind_group_layout: draw_bgl,
            dummy_blendshape_buffer,
            scene_bind_group_layout: scene_bgl,
            scene_uniform_buffer,
        }
    }
}

impl RenderPipeline for SkinnedPbrMrtRayQueryPipeline {
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
        draw_bind_group: Option<&wgpu::BindGroup>,
    ) {
        let dynamic_offset = batch_index
            .map(|i| self.uniform_ring.dynamic_offset(i, frame_index))
            .unwrap_or(0);
        let bg = draw_bind_group.expect("skinned PBR MRT ray query requires draw_bind_group");
        pass.set_bind_group(0, bg, &[dynamic_offset]);
    }

    fn bind_scene(&self, pass: &mut wgpu::RenderPass, scene_bind_group: Option<&wgpu::BindGroup>) {
        if let Some(bg) = scene_bind_group {
            pass.set_bind_group(1, bg, &[]);
        }
    }

    fn create_skinned_draw_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuMeshBuffers,
    ) -> Option<wgpu::BindGroup> {
        let blendshape = buffers
            .blendshape_buffer
            .as_ref()
            .map(|b| b.as_ref())
            .unwrap_or(&self.dummy_blendshape_buffer);
        Some(create_skinned_draw_bg(
            device,
            "skinned PBR MRT ray query draw BG",
            &self.bind_group_layout,
            &self.uniform_ring.buffer,
            blendshape,
        ))
    }

    fn draw_skinned(
        &self,
        pass: &mut wgpu::RenderPass,
        buffers: &GpuMeshBuffers,
        _uniforms: &UniformData<'_>,
    ) {
        self.set_skinned_buffers(pass, buffers);
        self.draw_skinned_indexed(pass, buffers, None);
    }

    fn set_skinned_buffers(&self, pass: &mut wgpu::RenderPass, buffers: &GpuMeshBuffers) {
        let Some((vb, ib)) = buffers.skinned_buffers() else {
            return;
        };
        pass.set_vertex_buffer(0, vb.slice(..));
        pass.set_index_buffer(ib.slice(..), buffers.index_format);
    }

    fn draw_skinned_indexed(
        &self,
        pass: &mut wgpu::RenderPass,
        buffers: &GpuMeshBuffers,
        index_range_override: Option<(u32, u32)>,
    ) {
        for &(index_start, index_count) in &buffers.effective_draw_ranges(index_range_override) {
            pass.draw_indexed(index_start..index_start + index_count, 0, 0..1);
        }
    }

    fn upload_skinned_batch(
        &self,
        queue: &wgpu::Queue,
        items: &[(Matrix4<f32>, &[[[f32; 4]; 4]], Option<&[f32]>, u32)],
        frame_index: u64,
    ) {
        self.uniform_ring.upload(queue, items, frame_index);
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
        Some(create_skinned_scene_bind_group_with_accel(
            device,
            "skinned PBR MRT ray query scene BG + RT shadow",
            &self.scene_bind_group_layout,
            &self.scene_uniform_buffer,
            &scene,
            queue,
            light_buffer,
            cluster_light_counts,
            cluster_light_indices,
            tlas,
            &rs,
        ))
    }
}
