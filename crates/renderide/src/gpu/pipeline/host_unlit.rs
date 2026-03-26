//! Host-driven unlit mesh pipeline: uniform ring (group 0) plus a flat color (group 1).
//!
//! Used when the material property stream sets [`MaterialPropertyUpdateType::set_shader`](crate::shared::MaterialPropertyUpdateType::set_shader)
//! and [`RenderConfig::use_host_unlit_pilot`](crate::config::RenderConfig::use_host_unlit_pilot) is enabled.

use bytemuck::{Pod, Zeroable};
use nalgebra::Matrix4;
use wgpu::util::DeviceExt;

use super::super::mesh::{GpuMeshBuffers, VertexPosNormal};
use super::builder;
use super::core::{RenderPipeline, UniformData};
use super::ring_buffer::UniformRingBuffer;

const HOST_UNLIT_WGSL: &str = include_str!(concat!(env!("OUT_DIR"), "/host_unlit.wgsl"));

/// Uniform for [`HostUnlitPipeline`] fragment color (matches `HostUnlitParams` in WGSL).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HostUnlitColorPod {
    base_color: [f32; 4],
}

/// Non-skinned mesh pipeline for host-selected unlit materials.
pub struct HostUnlitPipeline {
    pipeline: wgpu::RenderPipeline,
    uniform_ring: UniformRingBuffer,
    ring_bind_group: wgpu::BindGroup,
    /// GPU uniform for fragment [`HostUnlitColorPod`]; kept for future host-driven color updates.
    _color_buffer: wgpu::Buffer,
    color_bind_group: wgpu::BindGroup,
}

impl HostUnlitPipeline {
    /// Creates the host-unlit pipeline for the given surface format.
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("host unlit shader"),
            source: wgpu::ShaderSource::Wgsl(HOST_UNLIT_WGSL.into()),
        });
        let ring_bgl = builder::uniform_ring_bind_group_layout(device, "host unlit ring BGL");
        let color_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("host unlit color BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZero::new(
                        std::mem::size_of::<HostUnlitColorPod>() as u64,
                    ),
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("host unlit pipeline layout"),
            bind_group_layouts: &[&ring_bgl, &color_bgl],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("host unlit pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<VertexPosNormal>() as u64,
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
        let uniform_ring = UniformRingBuffer::new(device, "host unlit uniform ring buffer");
        let ring_bind_group = builder::uniform_ring_bind_group(
            device,
            "host unlit ring BG",
            &ring_bgl,
            &uniform_ring.buffer,
        );
        let initial = HostUnlitColorPod {
            base_color: [1.0, 1.0, 1.0, 1.0],
        };
        let color_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("host unlit base color"),
            contents: bytemuck::bytes_of(&initial),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let color_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("host unlit color BG"),
            layout: &color_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: color_buffer.as_entire_binding(),
            }],
        });
        Self {
            pipeline,
            uniform_ring,
            ring_bind_group,
            _color_buffer: color_buffer,
            color_bind_group,
        }
    }
}

impl RenderPipeline for HostUnlitPipeline {
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
        pass.set_bind_group(0, &self.ring_bind_group, &[dynamic_offset]);
        pass.set_bind_group(1, &self.color_bind_group, &[]);
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
        mvp_models: &[(Matrix4<f32>, Matrix4<f32>)],
        frame_index: u64,
    ) {
        self.uniform_ring.upload(queue, mvp_models, frame_index);
    }
}
