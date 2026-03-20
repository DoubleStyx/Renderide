//! UV debug pipeline: colors surfaces by UV coordinates using HSV color mapping.

use nalgebra::Matrix4;

use super::super::mesh::{GpuMeshBuffers, VertexWithUv};
use super::builder;
use super::core::{RenderPipeline, UniformData};
use super::ring_buffer::UniformRingBuffer;
use super::shaders::UV_DEBUG_SHADER_SRC;

/// UV debug pipeline: colors surfaces by UV coordinates.
pub struct UvDebugPipeline {
    pipeline: wgpu::RenderPipeline,
    uniform_ring: UniformRingBuffer,
    bind_group: wgpu::BindGroup,
}

impl UvDebugPipeline {
    /// Creates a UV debug pipeline. When `disable_depth_test` is true, uses
    /// `CompareFunction::Always` and disables depth write for screen-space overlay.
    pub fn new(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        disable_depth_test: bool,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("UV debug shader"),
            source: wgpu::ShaderSource::Wgsl(UV_DEBUG_SHADER_SRC.into()),
        });
        let bgl = builder::uniform_ring_bind_group_layout(device, "UV debug BGL");
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("UV debug pipeline layout"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("UV debug pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<VertexWithUv>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &builder::POS_UV_ATTRIBS,
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
            depth_stencil: Some(builder::depth_stencil_debug(disable_depth_test)),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let uniform_ring = UniformRingBuffer::new(device, "UV debug uniform ring buffer");
        let bind_group =
            builder::uniform_ring_bind_group(device, "UV debug BG", &bgl, &uniform_ring.buffer);
        Self {
            pipeline,
            uniform_ring,
            bind_group,
        }
    }
}

impl RenderPipeline for UvDebugPipeline {
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
        let (vb, ib) = buffers.uv_buffers();
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
}
