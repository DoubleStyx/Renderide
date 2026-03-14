//! wgpu mesh rendering with debug texture.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use nalgebra::Matrix4;
use crate::shared::{VertexAttributeFormat, VertexAttributeType};
use wgpu::util::DeviceExt;

use crate::assets::{self, MeshAsset};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct VertexWithUv {
    position: [f32; 3],
    uv: [f32; 2],
}

/// Position + smooth normal for normal debug shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct VertexPosNormal {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

/// Skinned vertex: position, normal, bone indices (4), bone weights (4).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct VertexSkinned {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub bone_indices: [i32; 4],
    pub bone_weights: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    mvp: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
}

/// Skinned uniforms: mvp + 256 bone matrices.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SkinnedUniforms {
    mvp: [[f32; 4]; 4],
    bone_matrices: [[[f32; 4]; 4]; 256],
}

/// Alignment for dynamic uniform buffer offsets (wgpu/Vulkan minimum).
const UNIFORM_ALIGNMENT: u64 = 256;
/// Maximum draws per frame for batched uniform buffer.
const MAX_BATCHED_DRAWS: usize = 4096;

/// Read a vec3 normal from vertex data at base+offset, converting from the given format to f32.
fn read_normal(data: &[u8], base: usize, offset: usize, format: VertexAttributeFormat) -> Option<[f32; 3]> {
    match format {
        VertexAttributeFormat::float32 => {
            if base + offset + 12 <= data.len() {
                Some([
                    f32::from_le_bytes(data[base + offset..base + offset + 4].try_into().ok()?),
                    f32::from_le_bytes(data[base + offset + 4..base + offset + 8].try_into().ok()?),
                    f32::from_le_bytes(data[base + offset + 8..base + offset + 12].try_into().ok()?),
                ])
            } else {
                None
            }
        }
        VertexAttributeFormat::half16 => {
            if base + offset + 6 <= data.len() {
                Some([
                    half_to_f32(u16::from_le_bytes(data[base + offset..base + offset + 2].try_into().ok()?)),
                    half_to_f32(u16::from_le_bytes(data[base + offset + 2..base + offset + 4].try_into().ok()?)),
                    half_to_f32(u16::from_le_bytes(data[base + offset + 4..base + offset + 6].try_into().ok()?)),
                ])
            } else {
                None
            }
        }
        VertexAttributeFormat::u_norm8 => {
            if base + offset + 3 <= data.len() {
                Some([
                    (data[base + offset] as f32 / 255.0) * 2.0 - 1.0,
                    (data[base + offset + 1] as f32 / 255.0) * 2.0 - 1.0,
                    (data[base + offset + 2] as f32 / 255.0) * 2.0 - 1.0,
                ])
            } else {
                None
            }
        }
        VertexAttributeFormat::u_norm16 => {
            if base + offset + 6 <= data.len() {
                Some([
                    (u16::from_le_bytes(data[base + offset..base + offset + 2].try_into().ok()?) as f32 / 65535.0) * 2.0 - 1.0,
                    (u16::from_le_bytes(data[base + offset + 2..base + offset + 4].try_into().ok()?) as f32 / 65535.0) * 2.0 - 1.0,
                    (u16::from_le_bytes(data[base + offset + 4..base + offset + 6].try_into().ok()?) as f32 / 65535.0) * 2.0 - 1.0,
                ])
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Read a vec2 UV from vertex data at base+offset, converting from the given format to f32.
fn read_uv(data: &[u8], base: usize, offset: usize, format: VertexAttributeFormat) -> Option<[f32; 2]> {
    match format {
        VertexAttributeFormat::float32 => {
            if base + offset + 8 <= data.len() {
                Some([
                    f32::from_le_bytes(data[base + offset..base + offset + 4].try_into().ok()?),
                    f32::from_le_bytes(data[base + offset + 4..base + offset + 8].try_into().ok()?),
                ])
            } else {
                None
            }
        }
        VertexAttributeFormat::half16 => {
            if base + offset + 4 <= data.len() {
                Some([
                    half_to_f32(u16::from_le_bytes(data[base + offset..base + offset + 2].try_into().ok()?)),
                    half_to_f32(u16::from_le_bytes(data[base + offset + 2..base + offset + 4].try_into().ok()?)),
                ])
            } else {
                None
            }
        }
        VertexAttributeFormat::u_norm8 => {
            if base + offset + 2 <= data.len() {
                Some([
                    data[base + offset] as f32 / 255.0,
                    data[base + offset + 1] as f32 / 255.0,
                ])
            } else {
                None
            }
        }
        VertexAttributeFormat::u_norm16 => {
            if base + offset + 4 <= data.len() {
                Some([
                    u16::from_le_bytes(data[base + offset..base + offset + 2].try_into().ok()?) as f32 / 65535.0,
                    u16::from_le_bytes(data[base + offset + 2..base + offset + 4].try_into().ok()?) as f32 / 65535.0,
                ])
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Convert IEEE 754 half-precision (f16) to f32.
fn half_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    if exp == 0 {
        let f = (sign << 31) | (mant << 13);
        f32::from_bits(f) * 5.960464477539063e-8
    } else if exp == 31 {
        let f = (sign << 31) | 0x7F800000 | (mant << 13);
        f32::from_bits(f)
    } else {
        let f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        f32::from_bits(f)
    }
}

/// Fallback cube with position+normal for normal debug pipeline (8 vertices, 12 triangles, 36 indices).
fn fallback_cube_pos_normal() -> (Vec<VertexPosNormal>, Vec<u16>) {
    let s = 0.5f32;
    let n = [0.0f32, 1.0, 0.0]; // placeholder normal (shader uses solid magenta)
    let vertices = vec![
        VertexPosNormal { position: [-s, -s, -s], normal: n },
        VertexPosNormal { position: [s, -s, -s], normal: n },
        VertexPosNormal { position: [s, s, -s], normal: n },
        VertexPosNormal { position: [-s, s, -s], normal: n },
        VertexPosNormal { position: [-s, -s, s], normal: n },
        VertexPosNormal { position: [s, -s, s], normal: n },
        VertexPosNormal { position: [s, s, s], normal: n },
        VertexPosNormal { position: [-s, s, s], normal: n },
    ];
    let indices: Vec<u16> = vec![
        0, 1, 2, 0, 2, 3,
        4, 6, 5, 4, 7, 6,
        0, 4, 5, 0, 5, 1,
        2, 6, 7, 2, 7, 3,
        0, 3, 7, 0, 7, 4,
        1, 5, 6, 1, 6, 2,
    ];
    (vertices, indices)
}

/// Fallback cube mesh (8 vertices, 12 triangles, 36 indices).
fn fallback_cube() -> (Vec<Vertex>, Vec<u16>) {
    let s = 0.5f32;
    let vertices = vec![
        Vertex { position: [-s, -s, -s] },
        Vertex { position: [s, -s, -s] },
        Vertex { position: [s, s, -s] },
        Vertex { position: [-s, s, -s] },
        Vertex { position: [-s, -s, s] },
        Vertex { position: [s, -s, s] },
        Vertex { position: [s, s, s] },
        Vertex { position: [-s, s, s] },
    ];
    let indices: Vec<u16> = vec![
        0, 1, 2, 0, 2, 3,
        4, 6, 5, 4, 7, 6,
        0, 4, 5, 0, 5, 1,
        2, 6, 7, 2, 7, 3,
        0, 3, 7, 0, 7, 4,
        1, 5, 6, 1, 6, 2,
    ];
    (vertices, indices)
}

/// Fallback cube with UVs for UV debug shader.
fn fallback_cube_with_uv() -> (Vec<VertexWithUv>, Vec<u16>) {
    let s = 0.5f32;
    let vertices = vec![
        VertexWithUv { position: [-s, -s, -s], uv: [0.0, 0.0] },
        VertexWithUv { position: [s, -s, -s], uv: [1.0, 0.0] },
        VertexWithUv { position: [s, s, -s], uv: [1.0, 1.0] },
        VertexWithUv { position: [-s, s, -s], uv: [0.0, 1.0] },
        VertexWithUv { position: [-s, -s, s], uv: [0.0, 0.0] },
        VertexWithUv { position: [s, -s, s], uv: [1.0, 0.0] },
        VertexWithUv { position: [s, s, s], uv: [1.0, 1.0] },
        VertexWithUv { position: [-s, s, s], uv: [0.0, 1.0] },
    ];
    let indices: Vec<u16> = vec![
        0, 1, 2, 0, 2, 3,
        4, 6, 5, 4, 7, 6,
        0, 4, 5, 0, 5, 1,
        2, 6, 7, 2, 7, 3,
        0, 3, 7, 0, 7, 4,
        1, 5, 6, 1, 6, 2,
    ];
    (vertices, indices)
}

/// Normal debug shader: colors surfaces by smooth normal (normal * 0.5 + 0.5 for 0-1 range).
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
    // World-space smooth normal: map [-1,1] to [0,1] for RGB visualization
    let n = normalize(in.world_normal);
    return vec4f(n * 0.5 + 0.5, 1.0);
}
"#;

/// Skinned mesh shader: transforms vertices by weighted bone matrices.
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

const UV_DEBUG_SHADER_SRC: &str = r#"
struct VertexInput {
    @location(0) position: vec3f,
    @location(1) uv: vec2f,
}
struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) uv: vec2f,
}
@group(0) @binding(0) var<uniform> mvp: mat4x4f;
@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = mvp * vec4f(in.position, 1.0);
    out.uv = in.uv;
    return out;
}
// HSV to RGB: H in [0,1] (0=red, ~0.833=violet), S in [0,1], V in [0,1]
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
    // U: red (0) -> violet (300/360) along horizontal
    // V: saturated (0) -> unsaturated (1) along vertical
    // Lightness: 1
    let u = clamp(in.uv.x, 0.0, 1.0);
    let v = clamp(in.uv.y, 0.0, 1.0);
    let hue = u * (300.0 / 360.0);
    let sat = 1.0 - v;
    let rgb = hsv_to_rgb(hue, sat, 1.0);
    return vec4f(rgb, 1.0);
}
"#;

pub struct MeshPipeline {
    pipeline: wgpu::RenderPipeline,
    skinned_pipeline: wgpu::RenderPipeline,
    material_pipeline: wgpu::RenderPipeline,
    debug_uv_pipeline: wgpu::RenderPipeline,
    fallback_vertex_buffer: wgpu::Buffer,
    fallback_vertex_buffer_uv: wgpu::Buffer,
    fallback_vertex_buffer_pos_normal: wgpu::Buffer,
    fallback_index_buffer: wgpu::Buffer,
    fallback_index_count: u32,
    uniform_buffer: wgpu::Buffer,
    uniform_buffer_batch: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
    /// Skinned pipeline bind group layout and uniform buffer (uploaded per skinned draw).
    skinned_bind_group_layout: wgpu::BindGroupLayout,
    skinned_uniform_buffer: wgpu::Buffer,
    skinned_bind_group: wgpu::BindGroup,
}

impl MeshPipeline {
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let normal_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("normal debug shader"),
            source: wgpu::ShaderSource::Wgsl(NORMAL_SHADER_SRC.into()),
        });
        let uv_debug_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("UV debug shader"),
            source: wgpu::ShaderSource::Wgsl(UV_DEBUG_SHADER_SRC.into()),
        });
        let skinned_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("skinned mesh shader"),
            source: wgpu::ShaderSource::Wgsl(SKINNED_SHADER_SRC.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mesh bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<Uniforms>() as u64),
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh pipeline (normal debug)"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &normal_shader,
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
                module: &normal_shader,
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
                depth_compare: wgpu::CompareFunction::GreaterEqual, // reverse-Z
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let skinned_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("skinned mesh bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<SkinnedUniforms>() as u64),
                },
                count: None,
            }],
        });
        let skinned_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("skinned mesh pipeline layout"),
            bind_group_layouts: &[&skinned_bind_group_layout],
            immediate_size: 0,
        });
        let skinned_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skinned mesh uniform buffer"),
            size: std::mem::size_of::<SkinnedUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let skinned_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skinned mesh bind group"),
            layout: &skinned_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &skinned_uniform_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<SkinnedUniforms>() as u64),
                }),
            }],
        });
        let skinned_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("skinned mesh pipeline"),
            layout: Some(&skinned_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &skinned_shader,
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
                module: &skinned_shader,
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

        let material_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("material pipeline (stub)"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &normal_shader,
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
                module: &normal_shader,
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

        let debug_uv_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("UV debug mesh pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &uv_debug_shader,
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
                module: &uv_debug_shader,
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
                depth_compare: wgpu::CompareFunction::GreaterEqual, // reverse-Z
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Fallback cube buffers: reserved for future use (e.g. "mesh not found" placeholder).
        // NOT used in the draw loop for invalid meshes—invalid meshes are skipped. Do not wire
        // these into the draw path for skipped_no_mesh or skipped_invalid_mesh.
        let (cube_verts, cube_indices) = fallback_cube();
        let (cube_verts_uv, _) = fallback_cube_with_uv();
        let (cube_verts_pn, _) = fallback_cube_pos_normal();
        let fallback_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fallback cube vertex buffer"),
            contents: bytemuck::cast_slice(&cube_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let fallback_vertex_buffer_uv = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fallback cube vertex buffer with UV"),
            contents: bytemuck::cast_slice(&cube_verts_uv),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let fallback_vertex_buffer_pos_normal = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fallback cube vertex buffer (pos+normal)"),
            contents: bytemuck::cast_slice(&cube_verts_pn),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let fallback_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fallback cube index buffer"),
            contents: bytemuck::cast_slice(&cube_indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let fallback_index_count = cube_indices.len() as u32;

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh uniform buffer (legacy single-draw)"),
            contents: bytemuck::cast_slice(&[Uniforms {
                mvp: Matrix4::identity().into(),
                model: Matrix4::identity().into(),
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_buffer_batch_size = (MAX_BATCHED_DRAWS as u64) * UNIFORM_ALIGNMENT;
        let uniform_buffer_batch = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh uniform buffer batch"),
            size: uniform_buffer_batch_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh bind group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buffer_batch,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<Uniforms>() as u64),
                }),
            }],
        });

        crate::log::log_write("[GPU STATE PROOF] cull=Back front=Cw depth=GreaterEqual reverseZ=true clear=0.0");

        Self {
            pipeline,
            skinned_pipeline,
            material_pipeline,
            debug_uv_pipeline,
            fallback_vertex_buffer,
            fallback_vertex_buffer_uv,
            fallback_vertex_buffer_pos_normal,
            fallback_index_buffer,
            fallback_index_count,
            uniform_buffer,
            uniform_buffer_batch,
            bind_group,
            bind_group_layout,
            skinned_bind_group_layout,
            skinned_uniform_buffer,
            skinned_bind_group,
        }
    }

    /// Uploads uniforms for multiple draws in one buffer write.
    /// Each (mvp, model) pair is written at `index * UNIFORM_ALIGNMENT` offset.
    pub fn upload_uniforms_batch(
        &self,
        queue: &wgpu::Queue,
        mvp_models: &[(Matrix4<f32>, Matrix4<f32>)],
    ) {
        if mvp_models.is_empty() || mvp_models.len() > MAX_BATCHED_DRAWS {
            return;
        }
        let mut aligned = vec![0u8; (mvp_models.len() as u64 * UNIFORM_ALIGNMENT) as usize];
        let uniform_size = std::mem::size_of::<Uniforms>();
        for (i, (mvp, model)) in mvp_models.iter().enumerate() {
            let u = Uniforms {
                mvp: (*mvp).into(),
                model: (*model).into(),
            };
            let offset = (i as u64 * UNIFORM_ALIGNMENT) as usize;
            let bytes: &[u8] = bytemuck::bytes_of(&u);
            aligned[offset..offset + uniform_size].copy_from_slice(bytes);
        }
        queue.write_buffer(&self.uniform_buffer_batch, 0, &aligned);
    }

    /// Uploads skinned uniforms (mvp + bone matrices) for a single skinned draw.
    pub fn upload_skinned_uniforms(
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
        queue.write_buffer(&self.skinned_uniform_buffer, 0, bytemuck::bytes_of(&u));
    }

    /// Draws a skinned mesh with pre-uploaded bone uniforms.
    pub fn draw_mesh_skinned(
        &self,
        pass: &mut wgpu::RenderPass,
        vertex_buffer: &wgpu::Buffer,
        index_buffer: &wgpu::Buffer,
        submeshes: &[(u32, u32)],
        index_format: wgpu::IndexFormat,
    ) {
        pass.set_pipeline(&self.skinned_pipeline);
        pass.set_bind_group(0, &self.skinned_bind_group, &[]);
        pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        pass.set_index_buffer(index_buffer.slice(..), index_format);
        for &(index_start, index_count) in submeshes {
            pass.draw_indexed(index_start..index_start + index_count, 0, 0..1);
        }
    }

    /// Draws a mesh using a pre-uploaded uniform at the given batch index.
    pub fn draw_mesh_with_offset(
        &self,
        pass: &mut wgpu::RenderPass,
        vertex_buffer: &wgpu::Buffer,
        index_buffer: &wgpu::Buffer,
        submeshes: &[(u32, u32)],
        index_format: wgpu::IndexFormat,
        batch_index: u32,
        use_debug_uv: bool,
        has_uvs: bool,
        is_skinned: bool,
        material_id: i32,
    ) {
        let pipeline = if use_debug_uv && has_uvs {
            &self.debug_uv_pipeline
        } else if is_skinned {
            &self.skinned_pipeline
        } else if material_id >= 0 {
            &self.material_pipeline
        } else {
            &self.pipeline
        };
        let dynamic_offset = (batch_index as u64 * UNIFORM_ALIGNMENT) as u32;
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.bind_group, &[dynamic_offset]);
        pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        pass.set_index_buffer(index_buffer.slice(..), index_format);
        for &(index_start, index_count) in submeshes {
            pass.draw_indexed(index_start..index_start + index_count, 0, 0..1);
        }
    }

    pub fn draw_mesh(
        &self,
        pass: &mut wgpu::RenderPass,
        queue: &wgpu::Queue,
        vertex_buffer: &wgpu::Buffer,
        index_buffer: &wgpu::Buffer,
        submeshes: &[(u32, u32)],
        index_format: wgpu::IndexFormat,
        mvp: Matrix4<f32>,
        model: Matrix4<f32>,
        use_debug_uv: bool,
        has_uvs: bool,
        is_skinned: bool,
        material_id: i32,
        _frame: u64,
    ) {
        let uniforms = Uniforms {
            mvp: mvp.into(),
            model: model.into(),
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
        let pipeline = if use_debug_uv && has_uvs {
            &self.debug_uv_pipeline
        } else if is_skinned {
            &self.skinned_pipeline
        } else if material_id >= 0 {
            &self.material_pipeline
        } else {
            &self.pipeline
        };
        pass.set_pipeline(pipeline);
        // Gated: was flooding (fires per draw, not per frame)
        // if frame % 200 == 0 {
        //     crate::log::log_write(&format!("[PIPELINE PROOF] frame={} → pipeline + bind_group + draw_indexed executed", frame));
        // }
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        pass.set_index_buffer(index_buffer.slice(..), index_format);
        for &(index_start, index_count) in submeshes {
            pass.draw_indexed(index_start..index_start + index_count, 0, 0..1);
        }
    }

    /// Fallback cube vertex buffer. Reserved for future placeholder use; not used in draw loop.
    pub fn fallback_vertex_buffer(&self) -> &wgpu::Buffer {
        &self.fallback_vertex_buffer
    }

    /// Fallback cube vertex buffer (pos+normal). Reserved for future placeholder use; not used in draw loop.
    pub fn fallback_vertex_buffer_pos_normal(&self) -> &wgpu::Buffer {
        &self.fallback_vertex_buffer_pos_normal
    }

    /// Fallback cube vertex buffer (pos+uv). Reserved for future placeholder use; not used in draw loop.
    pub fn fallback_vertex_buffer_uv(&self) -> &wgpu::Buffer {
        &self.fallback_vertex_buffer_uv
    }

    /// Fallback cube index buffer. Reserved for future placeholder use; not used in draw loop.
    pub fn fallback_index_buffer(&self) -> &wgpu::Buffer {
        &self.fallback_index_buffer
    }

    /// Fallback cube index count. Reserved for future placeholder use; not used in draw loop.
    pub fn fallback_index_count(&self) -> u32 {
        self.fallback_index_count
    }
}

/// Cached wgpu buffers for a mesh asset.
pub struct GpuMeshBuffers {
    pub vertex_buffer: Arc<wgpu::Buffer>,
    /// When present, used for UV debug shader (pos+uv layout). Main pipeline uses vertex_buffer (pos+normal).
    pub vertex_buffer_uv: Option<Arc<wgpu::Buffer>>,
    /// When present, used for skinned pipeline (pos+normal+bone_indices+bone_weights).
    pub vertex_buffer_skinned: Option<Arc<wgpu::Buffer>>,
    pub index_buffer: Arc<wgpu::Buffer>,
    /// Per-submesh (index_start, index_count). Empty means draw full range 0..index_count.
    pub submeshes: Vec<(u32, u32)>,
    pub index_format: wgpu::IndexFormat,
    /// True if vertex buffer includes UVs (for UV debug shader).
    pub has_uvs: bool,
}

/// Creates or gets GPU buffers for a mesh. Extracts position and smooth normal for normal debug shader.
/// Uses default normal (0,1,0) when mesh has no normal attribute.
pub fn create_mesh_buffers(
    device: &wgpu::Device,
    mesh: &MeshAsset,
    vertex_stride: usize,
) -> Option<GpuMeshBuffers> {
    if mesh.vertex_data.len() < 12 {
        return None;
    }
    if mesh.vertex_count <= 0 || mesh.index_count <= 0 {
        return None;
    }
    let vc = mesh.vertex_count as usize;
    if vertex_stride == 0 {
        return None;
    }
    let required_vb = vertex_stride * vc;
    if required_vb > mesh.vertex_data.len() {
        return None;
    }
    let pos_info = assets::attribute_offset_and_size(&mesh.vertex_attributes, VertexAttributeType::position)
        .unwrap_or((0, 12));
    let normal_info = assets::attribute_offset_size_format(&mesh.vertex_attributes, VertexAttributeType::normal);
    let uv_info = assets::attribute_offset_size_format(&mesh.vertex_attributes, VertexAttributeType::uv0);

    let (pos_off, _) = pos_info;
    let (normal_off, normal_size, normal_format) = normal_info
        .map(|(o, s, f)| (o, s, f))
        .unwrap_or((0, 0, VertexAttributeFormat::float32));
    let has_uvs = uv_info.map(|(_, s, _)| s >= 4).unwrap_or(false);

    let default_normal = [0.0f32, 1.0, 0.0];
    let default_uv = [0.0f32, 0.0];
    let (uv_off, uv_size, uv_format) = uv_info
        .map(|(o, s, f)| (o, s, f))
        .unwrap_or((0, 0, VertexAttributeFormat::float32));

    let mut vertices = Vec::with_capacity(mesh.vertex_count as usize);
    let mut vertices_uv: Option<Vec<VertexWithUv>> = if has_uvs {
        Some(Vec::with_capacity(mesh.vertex_count as usize))
    } else {
        None
    };

    for i in 0..mesh.vertex_count as usize {
        let base = i * vertex_stride;
        if base + pos_off + 12 > mesh.vertex_data.len() {
            continue;
        }
        let px = f32::from_le_bytes(mesh.vertex_data[base + pos_off..base + pos_off + 4].try_into().ok()?);
        let py = f32::from_le_bytes(mesh.vertex_data[base + pos_off + 4..base + pos_off + 8].try_into().ok()?);
        let pz = f32::from_le_bytes(mesh.vertex_data[base + pos_off + 8..base + pos_off + 12].try_into().ok()?);

        let mut normal = if normal_size > 0 {
            read_normal(&mesh.vertex_data, base, normal_off, normal_format)
                .unwrap_or(default_normal)
        } else {
            default_normal
        };
        // Normalize (half16/unorm may not be unit length)
        let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if len > 1e-6 {
            normal[0] /= len;
            normal[1] /= len;
            normal[2] /= len;
        }

        vertices.push(VertexPosNormal {
            position: [px, py, pz],
            normal,
        });

        if let Some(ref mut v_uv) = vertices_uv {
            let uv = if uv_size > 0 {
                read_uv(&mesh.vertex_data, base, uv_off, uv_format).unwrap_or(default_uv)
            } else {
                default_uv
            };
            v_uv.push(VertexWithUv {
                position: [px, py, pz],
                uv,
            });
        }
    }

    if vertices.len() < 3 {
        return None;
    }

    let vertex_buffer = Arc::new(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mesh vertex buffer (pos+normal)"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    }));

    let vertex_buffer_uv = vertices_uv.map(|v_uv| {
        Arc::new(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh vertex buffer (pos+uv)"),
            contents: bytemuck::cast_slice(&v_uv),
            usage: wgpu::BufferUsages::VERTEX,
        }))
    });

    let (index_data, index_format, index_count) = match mesh.index_format {
        crate::shared::IndexBufferFormat::u_int16 => {
            let count = mesh.index_data.len() / 2;
            if count == 0 {
                return None;
            }
            (mesh.index_data.clone(), wgpu::IndexFormat::Uint16, count as u32)
        }
        crate::shared::IndexBufferFormat::u_int32 => {
            let count = mesh.index_data.len() / 4;
            if count == 0 {
                return None;
            }
            (mesh.index_data.clone(), wgpu::IndexFormat::Uint32, count as u32)
        }
    };

    let index_buffer = Arc::new(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mesh index buffer"),
        contents: &index_data,
        usage: wgpu::BufferUsages::INDEX,
    }));

    let submeshes: Vec<(u32, u32)> = if mesh.submeshes.is_empty() {
        vec![(0, index_count)]
    } else {
        let s: Vec<(u32, u32)> = mesh
            .submeshes
            .iter()
            .map(|s| (s.index_start as u32, s.index_count as u32))
            .filter(|(start, count)| *count > 0 && start.saturating_add(*count) <= index_count)
            .collect();
        if s.is_empty() {
            vec![(0, index_count)]
        } else {
            s
        }
    };

    let vertex_buffer_skinned = if mesh.bone_count > 0 {
        build_skinned_vertices(device, mesh, vertex_stride, &vertices).map(Arc::new)
    } else {
        None
    };

    Some(GpuMeshBuffers {
        vertex_buffer,
        vertex_buffer_uv,
        vertex_buffer_skinned,
        index_buffer,
        submeshes,
        index_format,
        has_uvs,
    })
}

/// BoneWeight layout: weight (f32) + bone_index (i32) = 8 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BoneWeightPod {
    weight: f32,
    bone_index: i32,
}

/// Builds skinned vertex buffer from mesh bone data. Requires bone_counts and bone_weights.
fn build_skinned_vertices(
    device: &wgpu::Device,
    mesh: &MeshAsset,
    vertex_stride: usize,
    base_vertices: &[VertexPosNormal],
) -> Option<wgpu::Buffer> {
    let bone_counts = mesh.bone_counts.as_ref()?;
    let bone_weights = mesh.bone_weights.as_ref()?;
    if bone_counts.len() != base_vertices.len() {
        return None;
    }
    let vc = base_vertices.len();
    let mut skinned = Vec::with_capacity(vc);
    let mut weight_offset = 0;
    for (i, v) in base_vertices.iter().enumerate() {
        let n = bone_counts.get(i).copied().unwrap_or(0) as usize;
        let n = n.min(4);
        let mut indices = [0i32; 4];
        let mut weights = [0.0f32; 4];
        for j in 0..n {
            if weight_offset + 8 <= bone_weights.len() {
                let w: BoneWeightPod = bytemuck::pod_read_unaligned(
                    &bone_weights[weight_offset..weight_offset + 8],
                );
                indices[j] = w.bone_index.max(0).min(255);
                weights[j] = w.weight;
                weight_offset += 8;
            }
        }
        skinned.push(VertexSkinned {
            position: v.position,
            normal: v.normal,
            bone_indices: indices,
            bone_weights: weights,
        });
    }
    Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mesh vertex buffer (skinned)"),
        contents: bytemuck::cast_slice(&skinned),
        usage: wgpu::BufferUsages::VERTEX,
    }))
}

pub fn compute_vertex_stride_from_mesh(mesh: &MeshAsset) -> usize {
    mesh.vertex_data.len() / mesh.vertex_count.max(1) as usize
}
