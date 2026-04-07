//! Overlay stencil pipelines for GraphicsChunk masking.

use nalgebra::Matrix4;

use super::super::mesh::{GpuMeshBuffers, VertexWithUv};
use super::core::{RenderPipeline, UniformData};
use super::ring_buffer::OverlayStencilUniformRingBuffer;
use super::shaders::OVERLAY_STENCIL_SHADER_SRC;
use super::uniforms::OverlayStencilUniforms;

/// Stencil phase for GraphicsChunk RenderType flow.
pub(crate) enum OverlayStencilPhase {
    /// Equal, Keep, write_mask=0
    Content,
    /// Always, Replace, write_mask=0xFF
    MaskWrite,
    /// Always, Zero, write_mask=0xFF
    MaskClear,
}

fn create_overlay_stencil_pipeline(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
    phase: OverlayStencilPhase,
    label: &str,
) -> (
    wgpu::RenderPipeline,
    OverlayStencilUniformRingBuffer,
    wgpu::BindGroup,
) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("overlay stencil shader"),
        source: wgpu::ShaderSource::Wgsl(OVERLAY_STENCIL_SHADER_SRC.into()),
    });
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("overlay stencil bind group layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<
                    OverlayStencilUniforms,
                >() as u64),
            },
            count: None,
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("overlay stencil pipeline layout"),
        bind_group_layouts: &[&bind_group_layout],
        immediate_size: 0,
    });
    let (stencil_face, write_mask) = match phase {
        OverlayStencilPhase::Content => (
            wgpu::StencilFaceState {
                compare: wgpu::CompareFunction::Equal,
                fail_op: wgpu::StencilOperation::Keep,
                depth_fail_op: wgpu::StencilOperation::Keep,
                pass_op: wgpu::StencilOperation::Keep,
            },
            0u32,
        ),
        OverlayStencilPhase::MaskWrite => (
            wgpu::StencilFaceState {
                compare: wgpu::CompareFunction::Always,
                fail_op: wgpu::StencilOperation::Keep,
                depth_fail_op: wgpu::StencilOperation::Keep,
                pass_op: wgpu::StencilOperation::Replace,
            },
            0xFFu32,
        ),
        OverlayStencilPhase::MaskClear => (
            wgpu::StencilFaceState {
                compare: wgpu::CompareFunction::Always,
                fail_op: wgpu::StencilOperation::Keep,
                depth_fail_op: wgpu::StencilOperation::Keep,
                pass_op: wgpu::StencilOperation::Zero,
            },
            0xFFu32,
        ),
    };
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<VertexWithUv>() as u64,
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
                        format: wgpu::VertexFormat::Float32x2,
                    },
                ],
            }],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
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
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Always,
            stencil: wgpu::StencilState {
                front: stencil_face,
                back: stencil_face,
                read_mask: 0xFF,
                write_mask,
            },
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });
    let uniform_ring =
        OverlayStencilUniformRingBuffer::new(device, "overlay stencil uniform ring buffer");
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("overlay stencil bind group"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &uniform_ring.buffer,
                offset: 0,
                size: wgpu::BufferSize::new(std::mem::size_of::<OverlayStencilUniforms>() as u64),
            }),
        }],
    });
    (pipeline, uniform_ring, bind_group)
}

/// Overlay stencil pipeline for GraphicsChunk masking.
///
/// Implements the **Content** phase of the GraphicsChunk RenderType flow (see
/// [`crate::stencil`]): stencil compare=Equal, pass_op=Keep, read_mask=0xFF,
/// write_mask=0. Used when overlay draws have `stencil_state`. The pass must call
/// `set_stencil_reference` before each draw with the draw's `stencil_state.reference`.
///
/// MaskWrite and MaskClear use separate pipeline variants ([`OverlayStencilMaskWritePipeline`],
/// [`OverlayStencilMaskClearPipeline`]).
pub struct OverlayStencilPipeline {
    pipeline: wgpu::RenderPipeline,
    uniform_ring: OverlayStencilUniformRingBuffer,
    bind_group: wgpu::BindGroup,
}

impl OverlayStencilPipeline {
    /// Creates an overlay stencil pipeline for the Content phase.
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let (pipeline, uniform_ring, bind_group) = create_overlay_stencil_pipeline(
            device,
            config,
            OverlayStencilPhase::Content,
            "overlay stencil pipeline",
        );
        Self {
            pipeline,
            uniform_ring,
            bind_group,
        }
    }
}

/// Overlay stencil MaskWrite pipeline: compare=Always, pass_op=Replace, write_mask=0xFF.
/// Writes mask shape for GraphicsChunk masking.
pub struct OverlayStencilMaskWritePipeline {
    pipeline: wgpu::RenderPipeline,
    uniform_ring: OverlayStencilUniformRingBuffer,
    bind_group: wgpu::BindGroup,
}

impl OverlayStencilMaskWritePipeline {
    /// Creates an overlay stencil MaskWrite pipeline.
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let (pipeline, uniform_ring, bind_group) = create_overlay_stencil_pipeline(
            device,
            config,
            OverlayStencilPhase::MaskWrite,
            "overlay stencil mask write",
        );
        Self {
            pipeline,
            uniform_ring,
            bind_group,
        }
    }
}

/// Overlay stencil MaskClear pipeline: compare=Always, pass_op=Zero, write_mask=0xFF.
/// Clears mask for next GraphicsChunk.
pub struct OverlayStencilMaskClearPipeline {
    pipeline: wgpu::RenderPipeline,
    uniform_ring: OverlayStencilUniformRingBuffer,
    bind_group: wgpu::BindGroup,
}

impl OverlayStencilMaskClearPipeline {
    /// Creates an overlay stencil MaskClear pipeline.
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let (pipeline, uniform_ring, bind_group) = create_overlay_stencil_pipeline(
            device,
            config,
            OverlayStencilPhase::MaskClear,
            "overlay stencil mask clear",
        );
        Self {
            pipeline,
            uniform_ring,
            bind_group,
        }
    }
}

impl RenderPipeline for OverlayStencilMaskWritePipeline {
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

    fn upload_batch_overlay(
        &self,
        queue: &wgpu::Queue,
        items: &[(Matrix4<f32>, Matrix4<f32>, Option<[f32; 4]>)],
        frame_index: u64,
    ) {
        self.uniform_ring.upload(queue, items, frame_index);
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
        let (vb, ib) = buffers.uv_buffers();
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
}

impl RenderPipeline for OverlayStencilMaskClearPipeline {
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

    fn upload_batch_overlay(
        &self,
        queue: &wgpu::Queue,
        items: &[(Matrix4<f32>, Matrix4<f32>, Option<[f32; 4]>)],
        frame_index: u64,
    ) {
        self.uniform_ring.upload(queue, items, frame_index);
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
        let (vb, ib) = buffers.uv_buffers();
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
}

impl RenderPipeline for OverlayStencilPipeline {
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

    fn upload_batch_overlay(
        &self,
        queue: &wgpu::Queue,
        items: &[(Matrix4<f32>, Matrix4<f32>, Option<[f32; 4]>)],
        frame_index: u64,
    ) {
        self.uniform_ring.upload(queue, items, frame_index);
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
        let (vb, ib) = buffers.uv_buffers();
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

    fn upload_batch(
        &self,
        _queue: &wgpu::Queue,
        _draws: &[super::core::NonSkinnedUniformUpload],
        _frame_index: u64,
    ) {
    }
}
