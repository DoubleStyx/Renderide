//! Skinned mesh pipeline: transforms vertices by weighted bone matrices.

use nalgebra::Matrix4;

use super::super::mesh::{GpuMeshBuffers, VertexSkinned};
use super::builder;
use super::core::{RenderPipeline, UniformData};
use super::overlay_stencil::OverlayStencilPhase;
use super::ring_buffer::SkinnedUniformRingBuffer;
use super::shaders::SKINNED_SHADER_SRC;
use super::uniforms::SkinnedUniforms;

/// Skinned mesh pipeline: transforms vertices by weighted bone matrices.
pub struct SkinnedPipeline {
    pipeline: wgpu::RenderPipeline,
    uniform_ring: SkinnedUniformRingBuffer,
    bind_group_layout: wgpu::BindGroupLayout,
    dummy_blendshape_buffer: wgpu::Buffer,
}

impl SkinnedPipeline {
    /// Creates a skinned pipeline. When `stencil_phase` is `Some`, enables stencil for overlay
    /// masking (Content, MaskWrite, or MaskClear). When `disable_depth_test` is true, uses
    /// `CompareFunction::Always` and disables depth write for orthographic screen-space overlay.
    pub(crate) fn new(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        stencil_phase: Option<OverlayStencilPhase>,
        disable_depth_test: bool,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("skinned mesh shader"),
            source: wgpu::ShaderSource::Wgsl(SKINNED_SHADER_SRC.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("skinned mesh bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<
                            SkinnedUniforms,
                        >()
                            as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
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
            label: Some("skinned mesh pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });
        let uniform_ring =
            SkinnedUniformRingBuffer::new(device, "skinned mesh uniform ring buffer");
        let dummy_blendshape_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skinned mesh dummy blendshape buffer"),
            size: 1,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("skinned mesh pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<VertexSkinned>() as u64,
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
            depth_stencil: Some({
                let use_stencil = stencil_phase.is_some();
                let stencil = if let Some(phase) = stencil_phase {
                    let (face, write_mask) = match phase {
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
                    wgpu::StencilState {
                        front: face,
                        back: face,
                        read_mask: 0xFF,
                        write_mask,
                    }
                } else {
                    wgpu::StencilState::default()
                };
                wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24PlusStencil8,
                    depth_write_enabled: !use_stencil && !disable_depth_test,
                    depth_compare: if use_stencil || disable_depth_test {
                        wgpu::CompareFunction::Always
                    } else {
                        wgpu::CompareFunction::GreaterEqual
                    },
                    stencil,
                    bias: wgpu::DepthBiasState::default(),
                }
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        Self {
            pipeline,
            uniform_ring,
            bind_group_layout,
            dummy_blendshape_buffer,
        }
    }

    /// Creates a per-draw bind group with uniform buffer and mesh's blendshape buffer.
    fn create_draw_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuMeshBuffers,
    ) -> wgpu::BindGroup {
        let blendshape_buffer = buffers
            .blendshape_buffer
            .as_ref()
            .map(|b| b.as_ref())
            .unwrap_or(&self.dummy_blendshape_buffer);
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skinned mesh draw bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.uniform_ring.buffer,
                        offset: 0,
                        size: wgpu::BufferSize::new(std::mem::size_of::<SkinnedUniforms>() as u64),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: blendshape_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        })
    }
}

impl RenderPipeline for SkinnedPipeline {
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
        let bind_group = draw_bind_group.expect("skinned pipeline requires draw_bind_group");
        pass.set_bind_group(0, bind_group, &[dynamic_offset]);
    }

    fn create_skinned_draw_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuMeshBuffers,
    ) -> Option<wgpu::BindGroup> {
        Some(self.create_draw_bind_group(device, buffers))
    }

    fn draw_skinned(
        &self,
        pass: &mut wgpu::RenderPass,
        buffers: &GpuMeshBuffers,
        _uniforms: &UniformData<'_>,
    ) {
        self.set_skinned_buffers(pass, buffers);
        self.draw_skinned_indexed(pass, buffers);
    }

    fn set_skinned_buffers(&self, pass: &mut wgpu::RenderPass, buffers: &GpuMeshBuffers) {
        let Some((vb, ib)) = buffers.skinned_buffers() else {
            return;
        };
        pass.set_vertex_buffer(0, vb.slice(..));
        pass.set_index_buffer(ib.slice(..), buffers.index_format);
    }

    fn draw_skinned_indexed(&self, pass: &mut wgpu::RenderPass, buffers: &GpuMeshBuffers) {
        for &(index_start, index_count) in &buffers.draw_ranges() {
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
}
