//! Pipeline abstraction: RenderPipeline trait, PipelineManager, and concrete implementations.
//!
//! Extension point for pipelines, materials, PBR.

use super::mesh::{GpuMeshBuffers, VertexPosNormal, VertexSkinned, VertexWithUv};
use nalgebra::Matrix4;

/// Per-draw uniform data; pipelines extract what they need.
#[derive(Clone, Copy)]
pub enum UniformData<'a> {
    /// MVP + model for non-skinned.
    Simple {
        mvp: Matrix4<f32>,
        model: Matrix4<f32>,
    },
    /// MVP + bone matrices for skinned.
    Skinned {
        mvp: Matrix4<f32>,
        bone_matrices: &'a [[[f32; 4]; 4]],
    },
}

/// Abstraction for a render pipeline (shader, bind groups, draw logic).
pub trait RenderPipeline {
    /// Binds this pipeline and its bind groups to the render pass.
    /// `frame_index` is used by batched pipelines for ring buffer offset; ignored by others.
    fn bind(
        &self,
        pass: &mut wgpu::RenderPass,
        batch_index: Option<u32>,
        frame_index: u64,
    );

    /// Draws a non-skinned mesh. No-op for pipelines that only support skinned.
    fn draw_mesh(
        &self,
        pass: &mut wgpu::RenderPass,
        buffers: &GpuMeshBuffers,
        _uniforms: &UniformData<'_>,
    ) {
        let _ = (pass, buffers, _uniforms);
    }

    /// Draws a skinned mesh. No-op for pipelines that only support non-skinned.
    fn draw_skinned(
        &self,
        pass: &mut wgpu::RenderPass,
        buffers: &GpuMeshBuffers,
        _uniforms: &UniformData<'_>,
    ) {
        let _ = (pass, buffers, _uniforms);
    }

    /// Uploads batched uniforms for non-skinned draws. No-op for pipelines that don't batch.
    /// `frame_index` advances each frame for ring buffer region selection.
    fn upload_batch(
        &self,
        _queue: &wgpu::Queue,
        _mvp_models: &[(Matrix4<f32>, Matrix4<f32>)],
        _frame_index: u64,
    ) {
    }

    /// Uploads skinned uniforms for a single draw. No-op for non-skinned pipelines.
    fn upload_skinned(
        &self,
        _queue: &wgpu::Queue,
        _mvp: Matrix4<f32>,
        _bone_matrices: &[[[f32; 4]; 4]],
    ) {
    }
}

/// Alignment for dynamic uniform buffer offsets (wgpu/Vulkan minimum).
const UNIFORM_ALIGNMENT: u64 = 256;
/// Number of frame regions in the ring buffer (avoids overwriting in-flight data).
const NUM_FRAMES_IN_FLIGHT: usize = 3;
/// Slots per frame region. Draws exceeding this are split into multiple chunks.
const SLOTS_PER_FRAME: usize = 16_384;

/// MVP + model matrix for non-skinned pipelines.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    mvp: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
}

/// MVP + 256 bone matrices for skinned pipeline.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SkinnedUniforms {
    mvp: [[f32; 4]; 4],
    bone_matrices: [[[f32; 4]; 4]; 256],
}

/// Ring buffer for batched uniform data. Supports arbitrary draw counts by chunking.
/// No panic or silent drop when draws exceed a single region.
struct UniformRingBuffer {
    buffer: wgpu::Buffer,
}

impl UniformRingBuffer {
    fn new(device: &wgpu::Device, label: &str) -> Self {
        let size = (NUM_FRAMES_IN_FLIGHT * SLOTS_PER_FRAME) as u64 * UNIFORM_ALIGNMENT;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { buffer }
    }

    /// Uploads uniforms to the ring buffer, chunking if needed. No limit panic.
    fn upload(
        &self,
        queue: &wgpu::Queue,
        mvp_models: &[(Matrix4<f32>, Matrix4<f32>)],
        frame_index: u64,
    ) {
        if mvp_models.is_empty() {
            return;
        }
        let uniform_size = std::mem::size_of::<Uniforms>();
        let region_base = (frame_index as usize % NUM_FRAMES_IN_FLIGHT) * SLOTS_PER_FRAME;
        for (chunk_idx, chunk) in mvp_models.chunks(SLOTS_PER_FRAME).enumerate() {
            let region = (region_base + chunk_idx) % NUM_FRAMES_IN_FLIGHT;
            let buffer_offset = (region * SLOTS_PER_FRAME) as u64 * UNIFORM_ALIGNMENT;
            let mut aligned = vec![0u8; (chunk.len() as u64 * UNIFORM_ALIGNMENT) as usize];
            for (i, (mvp, model)) in chunk.iter().enumerate() {
                let u = Uniforms {
                    mvp: (*mvp).into(),
                    model: (*model).into(),
                };
                let offset = (i as u64 * UNIFORM_ALIGNMENT) as usize;
                let bytes: &[u8] = bytemuck::bytes_of(&u);
                aligned[offset..offset + uniform_size].copy_from_slice(bytes);
            }
            queue.write_buffer(&self.buffer, buffer_offset, &aligned);
        }
    }

    /// Computes dynamic offset for draw index `i` given `frame_index`.
    fn dynamic_offset(&self, batch_index: u32, frame_index: u64) -> u32 {
        let i = batch_index as usize;
        let chunk_idx = i / SLOTS_PER_FRAME;
        let slot_in_chunk = i % SLOTS_PER_FRAME;
        let region_base = (frame_index as usize % NUM_FRAMES_IN_FLIGHT) * SLOTS_PER_FRAME;
        let region = (region_base + chunk_idx) % NUM_FRAMES_IN_FLIGHT;
        let slot = region * SLOTS_PER_FRAME + slot_in_chunk;
        (slot as u64 * UNIFORM_ALIGNMENT) as u32
    }
}

const NORMAL_SHADER_SRC: &str = r#"
struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
}
struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) world_normal: vec3f,
}
struct Uniforms {
    mvp: mat4x4f,
    model: mat4x4f,
}
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.mvp * vec4f(in.position, 1.0);
    out.world_normal = (uniforms.model * vec4f(in.normal, 0.0)).xyz;
    return out;
}
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let n = normalize(in.world_normal);
    return vec4f(n * 0.5 + 0.5, 1.0);
}
"#;

const UV_DEBUG_SHADER_SRC: &str = r#"
struct VertexInput {
    @location(0) position: vec3f,
    @location(1) uv: vec2f,
}
struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) uv: vec2f,
}
struct Uniforms {
    mvp: mat4x4f,
    model: mat4x4f,
}
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.mvp * vec4f(in.position, 1.0);
    out.uv = in.uv;
    return out;
}
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3f {
    let c = v * s;
    let h6 = h * 6.0;
    let h2 = h6 - 2.0 * floor(h6 / 2.0);
    let x = c * (1.0 - abs(h2 - 1.0));
    let m = v - c;
    var r = 0.0;
    var g = 0.0;
    var b = 0.0;
    if h6 < 1.0 {
        r = c; g = x; b = 0.0;
    } else if h6 < 2.0 {
        r = x; g = c; b = 0.0;
    } else if h6 < 3.0 {
        r = 0.0; g = c; b = x;
    } else if h6 < 4.0 {
        r = 0.0; g = x; b = c;
    } else if h6 < 5.0 {
        r = x; g = 0.0; b = c;
    } else {
        r = c; g = 0.0; b = x;
    }
    return vec3f(r + m, g + m, b + m);
}
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let u = clamp(in.uv.x, 0.0, 1.0);
    let v = clamp(in.uv.y, 0.0, 1.0);
    let hue = u * (300.0 / 360.0);
    let sat = 1.0 - v;
    let rgb = hsv_to_rgb(hue, sat, 1.0);
    return vec4f(rgb, 1.0);
}
"#;

const SKINNED_SHADER_SRC: &str = r#"
struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) bone_indices: vec4i,
    @location(3) bone_weights: vec4f,
}
struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) world_normal: vec3f,
}
struct SkinnedUniforms {
    mvp: mat4x4f,
    bone_matrices: array<mat4x4f, 256>,
}
@group(0) @binding(0) var<uniform> uniforms: SkinnedUniforms;
@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    var world_pos = vec4f(0.0, 0.0, 0.0, 0.0);
    var world_normal = vec4f(0.0, 0.0, 0.0, 0.0);
    for (var i = 0; i < 4; i++) {
        let idx = clamp(in.bone_indices[i], 0, 255);
        let w = in.bone_weights[i];
        if w > 0.0 {
            let bone = uniforms.bone_matrices[idx];
            world_pos += w * bone * vec4f(in.position, 1.0);
            world_normal += w * bone * vec4f(in.normal, 0.0);
        }
    }
    out.clip_position = uniforms.mvp * world_pos;
    let n = world_normal.xyz;
    let len = length(n);
    out.world_normal = select(vec3f(0.0, 1.0, 0.0), n / len, len > 1e-6);
    return out;
}
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let n = normalize(in.world_normal);
    return vec4f(n * 0.5 + 0.5, 1.0);
}
"#;

/// Normal debug pipeline: colors surfaces by smooth normal.
pub struct NormalDebugPipeline {
    pipeline: wgpu::RenderPipeline,
    uniform_ring: UniformRingBuffer,
    bind_group: wgpu::BindGroup,
}

impl NormalDebugPipeline {
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("normal debug shader"),
            source: wgpu::ShaderSource::Wgsl(NORMAL_SHADER_SRC.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("normal debug bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: std::num::NonZeroU64::new(
                        std::mem::size_of::<Uniforms>() as u64
                    ),
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("normal debug pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("normal debug pipeline"),
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
                format: wgpu::TextureFormat::Depth24Plus,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::GreaterEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let uniform_ring = UniformRingBuffer::new(device, "normal debug uniform ring buffer");
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("normal debug bind group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_ring.buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<Uniforms>() as u64),
                }),
            }],
        });
        Self {
            pipeline,
            uniform_ring,
            bind_group,
        }
    }
}

impl RenderPipeline for NormalDebugPipeline {
    fn bind(
        &self,
        pass: &mut wgpu::RenderPass,
        batch_index: Option<u32>,
        frame_index: u64,
    ) {
        pass.set_pipeline(&self.pipeline);
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
        pass.set_vertex_buffer(0, buffers.vertex_buffer.slice(..));
        pass.set_index_buffer(buffers.index_buffer.slice(..), buffers.index_format);
        for &(index_start, index_count) in &buffers.submeshes {
            pass.draw_indexed(index_start..index_start + index_count, 0, 0..1);
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

/// UV debug pipeline: colors surfaces by UV coordinates.
pub struct UvDebugPipeline {
    pipeline: wgpu::RenderPipeline,
    uniform_ring: UniformRingBuffer,
    bind_group: wgpu::BindGroup,
}

impl UvDebugPipeline {
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("UV debug shader"),
            source: wgpu::ShaderSource::Wgsl(UV_DEBUG_SHADER_SRC.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("UV debug bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: std::num::NonZeroU64::new(
                        std::mem::size_of::<Uniforms>() as u64
                    ),
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("UV debug pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
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
                format: wgpu::TextureFormat::Depth24Plus,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::GreaterEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let uniform_ring = UniformRingBuffer::new(device, "UV debug uniform ring buffer");
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("UV debug bind group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_ring.buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<Uniforms>() as u64),
                }),
            }],
        });
        Self {
            pipeline,
            uniform_ring,
            bind_group,
        }
    }
}

impl RenderPipeline for UvDebugPipeline {
    fn bind(
        &self,
        pass: &mut wgpu::RenderPass,
        batch_index: Option<u32>,
        frame_index: u64,
    ) {
        pass.set_pipeline(&self.pipeline);
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
        let vb = buffers
            .vertex_buffer_uv
            .as_ref()
            .map(|b| b.as_ref())
            .unwrap_or(buffers.vertex_buffer.as_ref());
        pass.set_vertex_buffer(0, vb.slice(..));
        pass.set_index_buffer(buffers.index_buffer.slice(..), buffers.index_format);
        for &(index_start, index_count) in &buffers.submeshes {
            pass.draw_indexed(index_start..index_start + index_count, 0, 0..1);
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

/// Skinned mesh pipeline: transforms vertices by weighted bone matrices.
pub struct SkinnedPipeline {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl SkinnedPipeline {
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("skinned mesh shader"),
            source: wgpu::ShaderSource::Wgsl(SKINNED_SHADER_SRC.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("skinned mesh bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(
                        std::mem::size_of::<SkinnedUniforms>() as u64,
                    ),
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("skinned mesh pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skinned mesh uniform buffer"),
            size: std::mem::size_of::<SkinnedUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skinned mesh bind group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<SkinnedUniforms>() as u64),
                }),
            }],
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
                        wgpu::VertexAttribute {
                            offset: 24,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Sint32x4,
                        },
                        wgpu::VertexAttribute {
                            offset: 40,
                            shader_location: 3,
                            format: wgpu::VertexFormat::Float32x4,
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
                format: wgpu::TextureFormat::Depth24Plus,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::GreaterEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        Self {
            pipeline,
            uniform_buffer,
            bind_group,
        }
    }
}

impl RenderPipeline for SkinnedPipeline {
    fn bind(
        &self,
        pass: &mut wgpu::RenderPass,
        _batch_index: Option<u32>,
        _frame_index: u64,
    ) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
    }

    fn draw_skinned(
        &self,
        pass: &mut wgpu::RenderPass,
        buffers: &GpuMeshBuffers,
        _uniforms: &UniformData<'_>,
    ) {
        let vb = buffers
            .vertex_buffer_skinned
            .as_ref()
            .expect("skinned pipeline requires vertex_buffer_skinned");
        pass.set_vertex_buffer(0, vb.slice(..));
        pass.set_index_buffer(buffers.index_buffer.slice(..), buffers.index_format);
        for &(index_start, index_count) in &buffers.submeshes {
            pass.draw_indexed(index_start..index_start + index_count, 0, 0..1);
        }
    }

    fn upload_skinned(
        &self,
        queue: &wgpu::Queue,
        mvp: Matrix4<f32>,
        bone_matrices: &[[[f32; 4]; 4]],
    ) {
        let mut u = SkinnedUniforms {
            mvp: mvp.into(),
            bone_matrices: [[[0.0; 4]; 4]; 256],
        };
        let n = bone_matrices.len().min(256);
        u.bone_matrices[..n].copy_from_slice(&bone_matrices[..n]);
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&u));
    }
}

/// Material pipeline stub. Reserved for future use.
pub struct MaterialPipeline;

impl MaterialPipeline {
    pub fn new(_device: &wgpu::Device, _config: &wgpu::SurfaceConfiguration) -> Self {
        Self
    }
}

impl RenderPipeline for MaterialPipeline {
    fn bind(
        &self,
        _pass: &mut wgpu::RenderPass,
        _batch_index: Option<u32>,
        _frame_index: u64,
    ) {
        // Stub: no-op
    }
}

/// PBR pipeline stub. Reserved for future use.
pub struct PbrPipeline;

impl PbrPipeline {
    pub fn new(_device: &wgpu::Device, _config: &wgpu::SurfaceConfiguration) -> Self {
        Self
    }
}

impl RenderPipeline for PbrPipeline {
    fn bind(
        &self,
        _pass: &mut wgpu::RenderPass,
        _batch_index: Option<u32>,
        _frame_index: u64,
    ) {
        // Stub: no-op
    }
}

