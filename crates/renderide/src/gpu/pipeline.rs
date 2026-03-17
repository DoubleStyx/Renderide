//! Pipeline abstraction: RenderPipeline trait, PipelineManager, and concrete implementations.
//!
//! Extension point for pipelines, materials, PBR.

use super::mesh::{GpuMeshBuffers, VertexPosNormal, VertexSkinned, VertexWithUv};
use nalgebra::Matrix4;

/// Converts a nalgebra Matrix4 to WGSL column-major layout.
///
/// WGSL `mat4x4f` expects matrices in column-major order: column 0 (4 floats), then column 1,
/// then column 2, then column 3. Nalgebra is also column-major, so this explicitly ensures
/// the byte layout matches WGSL expectations when uploading to uniform buffers.
pub fn matrix4_to_wgsl_column_major(mat: &Matrix4<f32>) -> [[f32; 4]; 4] {
    let mut out = [[0.0f32; 4]; 4];
    for c in 0..4 {
        for r in 0..4 {
            out[c][r] = mat[(r, c)];
        }
    }
    out
}

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
    /// `draw_bind_group` is used by skinned pipeline for per-draw bind group (uniform + blendshape buffer).
    fn bind(
        &self,
        pass: &mut wgpu::RenderPass,
        batch_index: Option<u32>,
        frame_index: u64,
        draw_bind_group: Option<&wgpu::BindGroup>,
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

    /// Uploads batched uniforms for overlay stencil draws (includes clip_rect).
    /// Default calls `upload_batch` with mvp_models only.
    fn upload_batch_overlay(
        &self,
        queue: &wgpu::Queue,
        items: &[(Matrix4<f32>, Matrix4<f32>, Option<[f32; 4]>)],
        frame_index: u64,
    ) {
        let mvp_models: Vec<_> = items.iter().map(|(m, p, _)| (*m, *p)).collect();
        self.upload_batch(queue, &mvp_models, frame_index);
    }

    /// Uploads skinned uniforms for a single draw. No-op for non-skinned pipelines.
    fn upload_skinned(
        &self,
        _queue: &wgpu::Queue,
        _mvp: Matrix4<f32>,
        _bone_matrices: &[[[f32; 4]; 4]],
    ) {
    }

    /// Uploads batched skinned uniforms to the ring buffer. No-op for pipelines that don't batch.
    fn upload_skinned_batch(
        &self,
        _queue: &wgpu::Queue,
        _items: &[(Matrix4<f32>, &[[[f32; 4]; 4]], Option<&[f32]>, u32)],
        _frame_index: u64,
    ) {
    }

    /// Creates a per-draw bind group for skinned pipeline (uniform + blendshape buffer).
    /// Returns None for non-skinned pipelines.
    fn create_skinned_draw_bind_group(
        &self,
        _device: &wgpu::Device,
        _buffers: &GpuMeshBuffers,
    ) -> Option<wgpu::BindGroup> {
        None
    }
}

/// Alignment for dynamic uniform buffer offsets (wgpu/Vulkan minimum).
const UNIFORM_ALIGNMENT: u64 = 256;
/// Number of frame regions in the ring buffer (avoids overwriting in-flight data).
const NUM_FRAMES_IN_FLIGHT: usize = 3;
/// Slots per frame region. Draws exceeding this are split into multiple chunks.
const SLOTS_PER_FRAME: usize = 16_384;

/// Slot stride for skinned uniforms (mvp + 256 bones + num_blendshapes + 128 weights).
/// Aligned to 256 for dynamic offset. Struct size 16964, round up to 17152.
const SKINNED_SLOT_STRIDE: u64 = 17_152;
/// Slots per frame for skinned draws. Smaller than `SLOTS_PER_FRAME` to limit memory (~25 MB).
const SKINNED_SLOTS_PER_FRAME: usize = 512;

/// MVP + model matrix for non-skinned pipelines.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    mvp: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
}

/// Overlay stencil uniforms: MVP, model, and clip rect (x, y, width, height).
/// Pad to 256 bytes for dynamic offset alignment.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct OverlayStencilUniforms {
    mvp: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    clip_rect: [f32; 4],
    _pad: [f32; 16],
}

/// Maximum blendshape weights per draw. Meshes with more blendshapes are truncated; weights
/// beyond this index are ignored. The host (SkinnedMeshRendererManager) has no limit; Gloobie
/// uses a storage buffer for unbounded weights.
pub const MAX_BLENDSHAPE_WEIGHTS: usize = 128;

/// MVP + 256 bone matrices + blendshape weights for skinned pipeline.
///
/// Blendshape weights are applied in the vertex shader before bone skinning, matching
/// Gloobie's `applySkinning` order: base vertex → blendshapes → bones.
/// Weights stored as 32× vec4 ([`MAX_BLENDSHAPE_WEIGHTS`] floats) for WGSL uniform 16-byte alignment.
/// Meshes with more than [`MAX_BLENDSHAPE_WEIGHTS`] blendshapes are truncated; consider a storage
/// buffer for unbounded weight counts if needed.
/// Padding before blendshape_weights matches WGSL layout (vec4 requires 16-byte alignment).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SkinnedUniforms {
    mvp: [[f32; 4]; 4],
    bone_matrices: [[[f32; 4]; 4]; 256],
    num_blendshapes: u32,
    num_vertices: u32,
    /// Padding so blendshape_weights is 16-byte aligned (WGSL vec4 alignment).
    _pad: [u32; 2],
    /// Blendshape weights packed as 32 vec4s ([`MAX_BLENDSHAPE_WEIGHTS`] floats). Weights beyond
    /// index 127 are truncated.
    blendshape_weights: [[f32; 4]; 32],
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
                    mvp: matrix4_to_wgsl_column_major(mvp),
                    model: matrix4_to_wgsl_column_major(model),
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

/// Ring buffer for batched skinned uniform data (mvp + bone_matrices[256] per slot).
struct SkinnedUniformRingBuffer {
    /// GPU buffer backing the ring.
    buffer: wgpu::Buffer,
}

impl SkinnedUniformRingBuffer {
    /// Creates a new ring buffer sized for `NUM_FRAMES_IN_FLIGHT * SKINNED_SLOTS_PER_FRAME` slots.
    fn new(device: &wgpu::Device, label: &str) -> Self {
        let size =
            (NUM_FRAMES_IN_FLIGHT * SKINNED_SLOTS_PER_FRAME) as u64 * SKINNED_SLOT_STRIDE;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { buffer }
    }

    /// Uploads skinned uniforms to the ring buffer, chunking if needed. No limit panic.
    fn upload(
        &self,
        queue: &wgpu::Queue,
        items: &[(Matrix4<f32>, &[[[f32; 4]; 4]], Option<&[f32]>, u32)],
        frame_index: u64,
    ) {
        if items.is_empty() {
            return;
        }
        let uniform_size = std::mem::size_of::<SkinnedUniforms>();
        let region_base =
            (frame_index as usize % NUM_FRAMES_IN_FLIGHT) * SKINNED_SLOTS_PER_FRAME;
        for (chunk_idx, chunk) in items.chunks(SKINNED_SLOTS_PER_FRAME).enumerate() {
            let region = (region_base + chunk_idx) % NUM_FRAMES_IN_FLIGHT;
            let buffer_offset =
                (region * SKINNED_SLOTS_PER_FRAME) as u64 * SKINNED_SLOT_STRIDE;
            let mut aligned =
                vec![0u8; (chunk.len() as u64 * SKINNED_SLOT_STRIDE) as usize];
            for (i, (mvp, bone_matrices, blendshape_weights, num_vertices)) in chunk.iter().enumerate()
            {
                let mut u = SkinnedUniforms {
                    mvp: matrix4_to_wgsl_column_major(mvp),
                    bone_matrices: [[[0.0; 4]; 4]; 256],
                    num_blendshapes: 0,
                    num_vertices: *num_vertices,
                    _pad: [0, 0],
                    blendshape_weights: [[0.0; 4]; 32],
                };
                let n = bone_matrices.len().min(256);
                u.bone_matrices[..n].copy_from_slice(&bone_matrices[..n]);
                if let Some(weights) = blendshape_weights {
                    let count = weights.len().min(MAX_BLENDSHAPE_WEIGHTS);
                    u.num_blendshapes = count as u32;
                    for (i, &w) in weights.iter().take(MAX_BLENDSHAPE_WEIGHTS).enumerate() {
                        u.blendshape_weights[i / 4][i % 4] = w;
                    }
                }
                let offset = (i as u64 * SKINNED_SLOT_STRIDE) as usize;
                let bytes: &[u8] = bytemuck::bytes_of(&u);
                aligned[offset..offset + uniform_size].copy_from_slice(bytes);
            }
            queue.write_buffer(&self.buffer, buffer_offset, &aligned);
        }
    }

    /// Computes dynamic offset for draw index `i` given `frame_index`.
    fn dynamic_offset(&self, batch_index: u32, frame_index: u64) -> u32 {
        let i = batch_index as usize;
        let chunk_idx = i / SKINNED_SLOTS_PER_FRAME;
        let slot_in_chunk = i % SKINNED_SLOTS_PER_FRAME;
        let region_base =
            (frame_index as usize % NUM_FRAMES_IN_FLIGHT) * SKINNED_SLOTS_PER_FRAME;
        let region = (region_base + chunk_idx) % NUM_FRAMES_IN_FLIGHT;
        let slot = region * SKINNED_SLOTS_PER_FRAME + slot_in_chunk;
        (slot as u64 * SKINNED_SLOT_STRIDE) as u32
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

/// Overlay stencil shader with optional rect clip (IUIX_Material.RectClip).
const OVERLAY_STENCIL_SHADER_SRC: &str = r#"
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
    clip_rect: vec4f,
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
    let rect = uniforms.clip_rect;
    if rect.z > 0.0 {
        let ndc = in.clip_position.xy / in.clip_position.w;
        let nx = (ndc.x + 1.0) * 0.5;
        let ny = 1.0 - (ndc.y + 1.0) * 0.5;
        if nx < rect.x || nx > rect.x + rect.z || ny < rect.y || ny > rect.y + rect.w {
            discard;
        }
    }
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
    @location(2) tangent: vec3f,
    @location(3) bone_indices: vec4i,
    @location(4) bone_weights: vec4f,
}
struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) world_normal: vec3f,
}
struct SkinnedUniforms {
    mvp: mat4x4f,
    bone_matrices: array<mat4x4f, 256>,
    num_blendshapes: u32,
    num_vertices: u32,
    blendshape_weights: array<vec4f, 32>,
}
struct BlendshapeOffset {
    position_offset: vec3f,
    normal_offset: vec3f,
    tangent_offset: vec3f,
}
@group(0) @binding(0) var<uniform> uniforms: SkinnedUniforms;
@group(0) @binding(1) var<storage, read> blendshape_offsets: array<BlendshapeOffset>;
@vertex
fn vs_main(
    in: VertexInput,
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    var pos = in.position;
    var norm = in.normal;
    var tang = in.tangent;
    for (var i = 0u; i < uniforms.num_blendshapes; i++) {
        let q = i / 4u;
        let r = i % 4u;
        let v = uniforms.blendshape_weights[q];
        let weight = select(select(select(v.x, v.y, r == 1u), select(v.z, v.w, r == 3u), r >= 2u), v.x, r == 0u);
        if weight > 0.0 {
            let offset_idx = i * uniforms.num_vertices + vertex_index;
            let offset = blendshape_offsets[offset_idx];
            pos += offset.position_offset * weight;
            norm += offset.normal_offset * weight;
            tang += offset.tangent_offset * weight;
        }
    }
    var world_pos = vec4f(0.0, 0.0, 0.0, 0.0);
    var world_normal = vec4f(0.0, 0.0, 0.0, 0.0);
    var world_tangent = vec4f(0.0, 0.0, 0.0, 0.0);
    let total_weight = in.bone_weights[0] + in.bone_weights[1] + in.bone_weights[2] + in.bone_weights[3];
    let inv_total = select(1.0, 1.0 / total_weight, total_weight > 1e-6);
    for (var i = 0; i < 4; i++) {
        let idx = clamp(in.bone_indices[i], 0, 255);
        let w = in.bone_weights[i] * inv_total;
        if w > 0.0 {
            let bone = uniforms.bone_matrices[idx];
            world_pos += w * bone * vec4f(pos, 1.0);
            world_normal += w * bone * vec4f(norm, 0.0);
            world_tangent += w * bone * vec4f(tang, 0.0);
        }
    }
    _ = world_tangent;
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
    /// Creates a normal debug pipeline. When `disable_depth_test` is true, uses
    /// `CompareFunction::Always` and disables depth write for screen-space overlay.
    pub fn new(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        disable_depth_test: bool,
    ) -> Self {
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
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: !disable_depth_test,
                depth_compare: if disable_depth_test {
                    wgpu::CompareFunction::Always
                } else {
                    wgpu::CompareFunction::GreaterEqual
                },
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
        _draw_bind_group: Option<&wgpu::BindGroup>,
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
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: !disable_depth_test,
                depth_compare: if disable_depth_test {
                    wgpu::CompareFunction::Always
                } else {
                    wgpu::CompareFunction::GreaterEqual
                },
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
        _draw_bind_group: Option<&wgpu::BindGroup>,
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
    fn bind(
        &self,
        pass: &mut wgpu::RenderPass,
        batch_index: Option<u32>,
        frame_index: u64,
        _draw_bind_group: Option<&wgpu::BindGroup>,
    ) {
        pass.set_pipeline(&self.pipeline);
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
        let vb = buffers
            .vertex_buffer_uv
            .as_ref()
            .map(|b| b.as_ref())
            .unwrap_or(&buffers.vertex_buffer);
        pass.set_vertex_buffer(0, vb.slice(..));
        pass.set_index_buffer(buffers.index_buffer.slice(..), buffers.index_format);
        for &(index_start, index_count) in &buffers.submeshes {
            pass.draw_indexed(index_start..index_start + index_count, 0, 0..1);
        }
    }
}

impl RenderPipeline for OverlayStencilMaskClearPipeline {
    fn bind(
        &self,
        pass: &mut wgpu::RenderPass,
        batch_index: Option<u32>,
        frame_index: u64,
        _draw_bind_group: Option<&wgpu::BindGroup>,
    ) {
        pass.set_pipeline(&self.pipeline);
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
        let vb = buffers
            .vertex_buffer_uv
            .as_ref()
            .map(|b| b.as_ref())
            .unwrap_or(&buffers.vertex_buffer);
        pass.set_vertex_buffer(0, vb.slice(..));
        pass.set_index_buffer(buffers.index_buffer.slice(..), buffers.index_format);
        for &(index_start, index_count) in &buffers.submeshes {
            pass.draw_indexed(index_start..index_start + index_count, 0, 0..1);
        }
    }
}

/// Stencil phase for GraphicsChunk RenderType flow.
pub(crate) enum OverlayStencilPhase {
    Content,   // Equal, Keep, write_mask=0
    MaskWrite, // Always, Replace, write_mask=0xFF
    MaskClear, // Always, Zero, write_mask=0xFF
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
                min_binding_size: std::num::NonZeroU64::new(
                    std::mem::size_of::<OverlayStencilUniforms>() as u64,
                ),
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
                size: wgpu::BufferSize::new(
                    std::mem::size_of::<OverlayStencilUniforms>() as u64,
                ),
            }),
        }],
    });
    (pipeline, uniform_ring, bind_group)
}

/// Ring buffer for overlay stencil uniforms (mvp + model + clip_rect per slot).
struct OverlayStencilUniformRingBuffer {
    buffer: wgpu::Buffer,
}

impl OverlayStencilUniformRingBuffer {
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

    fn upload(
        &self,
        queue: &wgpu::Queue,
        items: &[(Matrix4<f32>, Matrix4<f32>, Option<[f32; 4]>)],
        frame_index: u64,
    ) {
        if items.is_empty() {
            return;
        }
        let uniform_size = std::mem::size_of::<OverlayStencilUniforms>();
        let region_base = (frame_index as usize % NUM_FRAMES_IN_FLIGHT) * SLOTS_PER_FRAME;
        for (chunk_idx, chunk) in items.chunks(SLOTS_PER_FRAME).enumerate() {
            let region = (region_base + chunk_idx) % NUM_FRAMES_IN_FLIGHT;
            let buffer_offset = (region * SLOTS_PER_FRAME) as u64 * UNIFORM_ALIGNMENT;
            let mut aligned = vec![0u8; (chunk.len() as u64 * UNIFORM_ALIGNMENT) as usize];
            for (i, (mvp, model, clip_rect)) in chunk.iter().enumerate() {
                let clip = clip_rect.unwrap_or([0.0, 0.0, 0.0, 0.0]);
                let u = OverlayStencilUniforms {
                    mvp: matrix4_to_wgsl_column_major(mvp),
                    model: matrix4_to_wgsl_column_major(model),
                    clip_rect: clip,
                    _pad: [0.0; 16],
                };
                let offset = (i as u64 * UNIFORM_ALIGNMENT) as usize;
                let bytes: &[u8] = bytemuck::bytes_of(&u);
                aligned[offset..offset + uniform_size].copy_from_slice(bytes);
            }
            queue.write_buffer(&self.buffer, buffer_offset, &aligned);
        }
    }

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

impl RenderPipeline for OverlayStencilPipeline {
    fn bind(
        &self,
        pass: &mut wgpu::RenderPass,
        batch_index: Option<u32>,
        frame_index: u64,
        _draw_bind_group: Option<&wgpu::BindGroup>,
    ) {
        pass.set_pipeline(&self.pipeline);
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
        _queue: &wgpu::Queue,
        _mvp_models: &[(Matrix4<f32>, Matrix4<f32>)],
        _frame_index: u64,
    ) {
    }
}

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
                        min_binding_size: std::num::NonZeroU64::new(
                            std::mem::size_of::<SkinnedUniforms>() as u64,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: true,
                        },
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
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset: 36,
                            shader_location: 3,
                            format: wgpu::VertexFormat::Sint32x4,
                        },
                        wgpu::VertexAttribute {
                            offset: 52,
                            shader_location: 4,
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
    fn bind(
        &self,
        pass: &mut wgpu::RenderPass,
        batch_index: Option<u32>,
        frame_index: u64,
        draw_bind_group: Option<&wgpu::BindGroup>,
    ) {
        pass.set_pipeline(&self.pipeline);
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
        let Some(vb) = buffers.vertex_buffer_skinned.as_ref() else {
            return;
        };
        pass.set_vertex_buffer(0, vb.slice(..));
        pass.set_index_buffer(buffers.index_buffer.slice(..), buffers.index_format);
        for &(index_start, index_count) in &buffers.submeshes {
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

/// Skinned overlay with stencil for GraphicsChunk masking.
///
/// Same stencil semantics as [`OverlayStencilPipeline`] (Content phase). Call
/// `set_stencil_reference` before each draw.
pub struct OverlayStencilSkinnedPipeline {
    inner: SkinnedPipeline,
}

impl OverlayStencilSkinnedPipeline {
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let inner = SkinnedPipeline::new(device, config, Some(OverlayStencilPhase::Content), false);
        Self { inner }
    }
}

/// Skinned overlay stencil MaskWrite pipeline.
pub struct OverlayStencilMaskWriteSkinnedPipeline {
    inner: SkinnedPipeline,
}

impl OverlayStencilMaskWriteSkinnedPipeline {
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let inner =
            SkinnedPipeline::new(device, config, Some(OverlayStencilPhase::MaskWrite), false);
        Self { inner }
    }
}

/// Skinned overlay stencil MaskClear pipeline.
pub struct OverlayStencilMaskClearSkinnedPipeline {
    inner: SkinnedPipeline,
}

impl OverlayStencilMaskClearSkinnedPipeline {
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let inner =
            SkinnedPipeline::new(device, config, Some(OverlayStencilPhase::MaskClear), false);
        Self { inner }
    }
}

impl RenderPipeline for OverlayStencilMaskWriteSkinnedPipeline {
    fn bind(
        &self,
        pass: &mut wgpu::RenderPass,
        batch_index: Option<u32>,
        frame_index: u64,
        draw_bind_group: Option<&wgpu::BindGroup>,
    ) {
        self.inner.bind(pass, batch_index, frame_index, draw_bind_group);
    }

    fn create_skinned_draw_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuMeshBuffers,
    ) -> Option<wgpu::BindGroup> {
        self.inner.create_skinned_draw_bind_group(device, buffers)
    }

    fn draw_skinned(
        &self,
        pass: &mut wgpu::RenderPass,
        buffers: &GpuMeshBuffers,
        uniforms: &UniformData<'_>,
    ) {
        self.inner.draw_skinned(pass, buffers, uniforms);
    }

    fn upload_skinned_batch(
        &self,
        queue: &wgpu::Queue,
        items: &[(Matrix4<f32>, &[[[f32; 4]; 4]], Option<&[f32]>, u32)],
        frame_index: u64,
    ) {
        self.inner.upload_skinned_batch(queue, items, frame_index);
    }
}

impl RenderPipeline for OverlayStencilMaskClearSkinnedPipeline {
    fn bind(
        &self,
        pass: &mut wgpu::RenderPass,
        batch_index: Option<u32>,
        frame_index: u64,
        draw_bind_group: Option<&wgpu::BindGroup>,
    ) {
        self.inner.bind(pass, batch_index, frame_index, draw_bind_group);
    }

    fn create_skinned_draw_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuMeshBuffers,
    ) -> Option<wgpu::BindGroup> {
        self.inner.create_skinned_draw_bind_group(device, buffers)
    }

    fn draw_skinned(
        &self,
        pass: &mut wgpu::RenderPass,
        buffers: &GpuMeshBuffers,
        uniforms: &UniformData<'_>,
    ) {
        self.inner.draw_skinned(pass, buffers, uniforms);
    }

    fn upload_skinned_batch(
        &self,
        queue: &wgpu::Queue,
        items: &[(Matrix4<f32>, &[[[f32; 4]; 4]], Option<&[f32]>, u32)],
        frame_index: u64,
    ) {
        self.inner.upload_skinned_batch(queue, items, frame_index);
    }
}

impl RenderPipeline for OverlayStencilSkinnedPipeline {
    fn bind(
        &self,
        pass: &mut wgpu::RenderPass,
        batch_index: Option<u32>,
        frame_index: u64,
        draw_bind_group: Option<&wgpu::BindGroup>,
    ) {
        self.inner.bind(pass, batch_index, frame_index, draw_bind_group);
    }

    fn create_skinned_draw_bind_group(
        &self,
        device: &wgpu::Device,
        buffers: &GpuMeshBuffers,
    ) -> Option<wgpu::BindGroup> {
        self.inner.create_skinned_draw_bind_group(device, buffers)
    }

    fn draw_skinned(
        &self,
        pass: &mut wgpu::RenderPass,
        buffers: &GpuMeshBuffers,
        uniforms: &UniformData<'_>,
    ) {
        self.inner.draw_skinned(pass, buffers, uniforms);
    }

    fn upload_skinned_batch(
        &self,
        queue: &wgpu::Queue,
        items: &[(Matrix4<f32>, &[[[f32; 4]; 4]], Option<&[f32]>, u32)],
        frame_index: u64,
    ) {
        self.inner.upload_skinned_batch(queue, items, frame_index);
    }
}
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
        _draw_bind_group: Option<&wgpu::BindGroup>,
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
        _draw_bind_group: Option<&wgpu::BindGroup>,
    ) {
        // Stub: no-op
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that matrix4_to_wgsl_column_major produces byte layout matching WGSL mat4x4f:
    /// column 0 at offset 0, column 1 at 16, column 2 at 32, column 3 at 48.
    #[test]
    fn test_matrix4_to_wgsl_column_major_layout() {
        let mat = Matrix4::from_fn(|r, c| (r * 4 + c) as f32);
        let arr = matrix4_to_wgsl_column_major(&mat);
        let bytes: &[u8] = bytemuck::bytes_of(&arr);
        assert_eq!(bytes.len(), 64);
        for c in 0..4 {
            let offset = c * 16;
            for r in 0..4 {
                let val = f32::from_le_bytes(bytes[offset + r * 4..offset + r * 4 + 4].try_into().unwrap());
                assert_eq!(val, (r * 4 + c) as f32, "col {} row {}", c, r);
            }
        }
    }

    /// Verifies translation in column 3 (typical model matrix layout).
    #[test]
    fn test_matrix4_wgsl_translation_column3() {
        let mat = Matrix4::new_translation(&nalgebra::Vector3::new(1.0, 2.0, 3.0));
        let arr = matrix4_to_wgsl_column_major(&mat);
        let bytes: &[u8] = bytemuck::bytes_of(&arr);
        let col3_offset = 48;
        let tx = f32::from_le_bytes(bytes[col3_offset..col3_offset + 4].try_into().unwrap());
        let ty = f32::from_le_bytes(bytes[col3_offset + 4..col3_offset + 8].try_into().unwrap());
        let tz = f32::from_le_bytes(bytes[col3_offset + 8..col3_offset + 12].try_into().unwrap());
        assert!((tx - 1.0).abs() < 1e-6);
        assert!((ty - 2.0).abs() < 1e-6);
        assert!((tz - 3.0).abs() < 1e-6);
    }
}

