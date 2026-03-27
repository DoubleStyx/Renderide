//! Interleaved vertex layouts uploaded to GPU vertex buffers for mesh pipelines.

use bytemuck::{Pod, Zeroable};

/// Vertex with position and UV for UV debug pipeline.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct VertexWithUv {
    pub position: [f32; 3],
    pub uv: [f32; 2],
}

/// Position + smooth normal for normal debug shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct VertexPosNormal {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

/// Position, normal, and UV0 for host-albedo forward PBR (`PipelineVariant::PbrHostAlbedo`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct VertexPosNormalUv {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

/// Interleaved vertex for Resonite Canvas / `UI_Unlit` and `UI_TextUnlit` (position, UV, color, aux).
///
/// `aux` stores `TANGENT` (lerp color) for image UI when tangents are present, otherwise `NORMAL`
/// (SDF per-vertex dilate/outline bias in xyz, matching the NORMAL slot used as extra data for UI
/// text shaders). UI text glyph meshes from the host typically include normals for this data and
/// omit tangents, so `aux` is taken from the normal stream for text.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct VertexUiCanvas {
    /// Object-space position.
    pub position: [f32; 3],
    /// UV0.
    pub uv: [f32; 2],
    /// Vertex color (tint multiplier), linear or sRGB per host.
    pub color: [f32; 4],
    /// `UI_Unlit`: lerp color from tangent; `UI_TextUnlit`: packed extra data from normal.xyz.
    pub aux: [f32; 4],
}

/// Skinned vertex: position, normal, tangent, bone indices (4), bone weights (4).
///
/// Tangent is used for blendshape tangent_offset application and normal mapping. Defaults to
/// [1, 0, 0] when the mesh has no tangent attribute.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct VertexSkinned {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tangent: [f32; 3],
    pub bone_indices: [i32; 4],
    pub bone_weights: [f32; 4],
}

/// Per-vertex blendshape offset for storage buffer binding (48 bytes).
///
/// WGSL vec3 has 16-byte alignment, so layout is: position (0-12), pad, normal (16-28), pad,
/// tangent (32-44), pad. Indexed in the shader as `blendshape_index * num_vertices + vertex_index`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlendshapeOffset {
    pub position_offset: [f32; 3],
    _pad0: f32,
    pub normal_offset: [f32; 3],
    _pad1: f32,
    pub tangent_offset: [f32; 3],
    _pad2: f32,
}
