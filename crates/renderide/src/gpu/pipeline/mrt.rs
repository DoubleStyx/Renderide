//! MRT pipelines: output color, world position, world normal for RTAO.
//!
//! [`NormalDebugMRTPipeline`] and [`UvDebugMRTPipeline`] share the same Rust-side implementation
//! (field layout, bind group setup, draw loop) and differ only in the vertex buffer method
//! they call on [`GpuMeshBuffers`]. The `impl_non_skinned_mrt_pipeline!` macro eliminates that
//! duplication without requiring a generic or trait object.

use nalgebra::Matrix4;

use super::super::mesh::{GpuMeshBuffers, VertexPosNormal, VertexSkinned, VertexWithUv};
use super::builder;
use super::core::{RenderPipeline, UNIFORM_ALIGNMENT, UniformData};
use super::ring_buffer::{SkinnedUniformRingBuffer, UniformRingBuffer};
use super::shaders::{
    NORMAL_DEBUG_MRT_SHADER_SRC, SKINNED_MRT_SHADER_SRC, UV_DEBUG_MRT_SHADER_SRC,
};
use super::uniforms::SkinnedUniforms;

// ─── Shared MRT G-buffer types ────────────────────────────────────────────────

/// Per-frame uniform uploaded to the MRT G-buffer origin buffer (group 1, binding 0).
///
/// The MRT position target stores `world - view_position` (camera-relative) so
/// `Rgba16Float` retains precision far from the world origin. This struct carries
/// `view_position` so the fragment shader can perform the subtraction.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct MrtGbufferOriginUniform {
    /// World-space camera position (= primary view translation for the main batch).
    pub view_position: [f32; 3],
    pub _pad: f32,
}

#[cfg(test)]
mod mrt_gbuffer_origin_uniform_tests {
    use super::MrtGbufferOriginUniform;
    use std::mem::size_of;

    #[test]
    fn mrt_gbuffer_origin_uniform_is_16_bytes() {
        assert_eq!(size_of::<MrtGbufferOriginUniform>(), 16);
        assert_eq!(size_of::<MrtGbufferOriginUniform>() % 16, 0);
    }
}

/// Shared bind group layout (group 1) for non-skinned debug MRT pipelines.
///
/// Contains one uniform binding for [`MrtGbufferOriginUniform`]. PBR MRT pipelines use a
/// different group-1 layout (scene uniforms + lights); only debug MRT variants use this.
pub fn create_mrt_gbuffer_origin_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("MRT g-buffer origin BGL"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<
                    MrtGbufferOriginUniform,
                >() as u64),
            },
            count: None,
        }],
    })
}

/// Position G-buffer format (`Rgba16Float`, camera-relative). Shared by all MRT pipelines.
pub(crate) const MRT_POSITION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Normal G-buffer format (`Rgba16Float`). Shared by all MRT pipelines.
pub(crate) const MRT_NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

// ─── Macro: non-skinned MRT RenderPipeline impl ───────────────────────────────

/// Generates the full `RenderPipeline` impl for a non-skinned MRT pipeline.
///
/// The two non-skinned MRT types (`NormalDebugMRTPipeline`, `UvDebugMRTPipeline`) share
/// every `RenderPipeline` method except `set_mesh_buffers`, which calls either
/// `buffers.normal_buffers()` or `buffers.uv_buffers()`. This macro eliminates the duplication.
macro_rules! impl_non_skinned_mrt_pipeline {
    ($ty:ty, $buf_method:ident) => {
        impl RenderPipeline for $ty {
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
                let (vb, ib) = buffers.$buf_method();
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
                mvp_models: &[(nalgebra::Matrix4<f32>, nalgebra::Matrix4<f32>)],
                frame_index: u64,
            ) {
                self.uniform_ring.upload(queue, mvp_models, frame_index);
            }
        }
    };
}

// ─── NormalDebugMRTPipeline ───────────────────────────────────────────────────

/// Normal debug MRT pipeline: outputs color, world position, world normal for RTAO.
///
/// Used when [`RenderConfig::rtao_enabled`](crate::config::RenderConfig) is true.
/// Writes to three color attachments in a single forward pass.
pub struct NormalDebugMRTPipeline {
    pipeline: wgpu::RenderPipeline,
    uniform_ring: UniformRingBuffer,
    bind_group: wgpu::BindGroup,
}

impl NormalDebugMRTPipeline {
    /// Creates an MRT normal debug pipeline. `mrt_gbuffer_origin_layout` must be the layout
    /// from [`create_mrt_gbuffer_origin_bind_group_layout`] owned by the same `PipelineManager`.
    pub fn new(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        mrt_gbuffer_origin_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("normal debug MRT shader"),
            source: wgpu::ShaderSource::Wgsl(NORMAL_DEBUG_MRT_SHADER_SRC.into()),
        });
        let bgl = builder::uniform_ring_bind_group_layout(device, "normal debug MRT BGL");
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("normal debug MRT pipeline layout"),
            bind_group_layouts: &[&bgl, mrt_gbuffer_origin_layout],
            immediate_size: 0,
        });
        let mrt_targets = builder::mrt_color_targets(config.format);
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("normal debug MRT pipeline"),
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
                targets: &mrt_targets,
                compilation_options: Default::default(),
            }),
            primitive: builder::standard_primitive_state(),
            depth_stencil: Some(builder::depth_stencil_opaque()),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let uniform_ring = UniformRingBuffer::new(device, "normal debug MRT uniform ring buffer");
        let bind_group = builder::uniform_ring_bind_group(
            device,
            "normal debug MRT BG",
            &bgl,
            &uniform_ring.buffer,
        );
        Self {
            pipeline,
            uniform_ring,
            bind_group,
        }
    }
}

impl_non_skinned_mrt_pipeline!(NormalDebugMRTPipeline, normal_buffers);

// ─── UvDebugMRTPipeline ───────────────────────────────────────────────────────

/// UV debug MRT pipeline: outputs color (HSV from UV), world position, world normal for RTAO.
///
/// UV meshes lack per-vertex normals; uses model +Y as fallback normal.
pub struct UvDebugMRTPipeline {
    pipeline: wgpu::RenderPipeline,
    uniform_ring: UniformRingBuffer,
    bind_group: wgpu::BindGroup,
}

impl UvDebugMRTPipeline {
    /// Creates an MRT UV debug pipeline. `mrt_gbuffer_origin_layout` must match
    /// [`create_mrt_gbuffer_origin_bind_group_layout`].
    pub fn new(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        mrt_gbuffer_origin_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("UV debug MRT shader"),
            source: wgpu::ShaderSource::Wgsl(UV_DEBUG_MRT_SHADER_SRC.into()),
        });
        let bgl = builder::uniform_ring_bind_group_layout(device, "UV debug MRT BGL");
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("UV debug MRT pipeline layout"),
            bind_group_layouts: &[&bgl, mrt_gbuffer_origin_layout],
            immediate_size: 0,
        });
        let mrt_targets = builder::mrt_color_targets(config.format);
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("UV debug MRT pipeline"),
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
                targets: &mrt_targets,
                compilation_options: Default::default(),
            }),
            primitive: builder::standard_primitive_state(),
            depth_stencil: Some(builder::depth_stencil_opaque()),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let uniform_ring = UniformRingBuffer::new(device, "UV debug MRT uniform ring buffer");
        let bind_group =
            builder::uniform_ring_bind_group(device, "UV debug MRT BG", &bgl, &uniform_ring.buffer);
        Self {
            pipeline,
            uniform_ring,
            bind_group,
        }
    }
}

impl_non_skinned_mrt_pipeline!(UvDebugMRTPipeline, uv_buffers);

// ─── SkinnedMRTPipeline ───────────────────────────────────────────────────────

/// Skinned MRT pipeline: bone-weighted vertices with three-target G-buffer output for RTAO.
///
/// Same bone skinning and blendshape logic as [`super::skinned::SkinnedPipeline`].
pub struct SkinnedMRTPipeline {
    pipeline: wgpu::RenderPipeline,
    uniform_ring: SkinnedUniformRingBuffer,
    bind_group_layout: wgpu::BindGroupLayout,
    dummy_blendshape_buffer: wgpu::Buffer,
}

impl SkinnedMRTPipeline {
    /// Creates an MRT skinned pipeline. `mrt_gbuffer_origin_layout` must match
    /// [`create_mrt_gbuffer_origin_bind_group_layout`].
    pub fn new(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        mrt_gbuffer_origin_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("skinned MRT shader"),
            source: wgpu::ShaderSource::Wgsl(SKINNED_MRT_SHADER_SRC.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("skinned MRT BGL"),
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
            label: Some("skinned MRT pipeline layout"),
            bind_group_layouts: &[&bind_group_layout, mrt_gbuffer_origin_layout],
            immediate_size: 0,
        });
        let uniform_ring = SkinnedUniformRingBuffer::new(device, "skinned MRT uniform ring buffer");
        let dummy_blendshape_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skinned MRT dummy blendshape buffer"),
            size: 1,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let mrt_targets = builder::mrt_color_targets(config.format);
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("skinned MRT pipeline"),
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
            bind_group_layout,
            dummy_blendshape_buffer,
        }
    }

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
            label: Some("skinned MRT draw BG"),
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

impl RenderPipeline for SkinnedMRTPipeline {
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
        let bind_group = draw_bind_group.expect("skinned MRT pipeline requires draw_bind_group");
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

// Suppress unused import warning: UNIFORM_ALIGNMENT is used transitively via builder constants.
const _: u64 = UNIFORM_ALIGNMENT;
