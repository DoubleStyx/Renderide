//! Shared per-draw instance data layout.

#define_import_path renderide::draw::types

struct PerDrawUniforms {
    view_proj_left: mat4x4<f32>,
    view_proj_right: mat4x4<f32>,
    model: mat4x4<f32>,
    /// Inverse transpose of the upper 3x3 of `model` (correct normals under non-uniform scale).
    normal_matrix: mat3x3<f32>,
    /// Metadata. `x` marks world-space position streams; `yzw` pack reflection-probe selection.
    _pad: vec4<f32>,
    /// Particle draw metadata: kind, alignment, min screen size, max screen size.
    particle_header: vec4<f32>,
    /// Particle draw color/tint.
    particle_tint: vec4<f32>,
    /// Particle draw metadata: motion mode, frame index, trail texture mode, trail lighting flag.
    particle_extra: vec4<f32>,
    /// Padding so each dynamic storage row remains aligned to 512 bytes.
    particle_padding: array<vec4<f32>, 13>,
}

/// `_pad.x` marker for world-space position streams.
const POSITION_STREAM_WORLD_SPACE_FLAG: f32 = 1.0;
const PARTICLE_KIND_NONE: u32 = 0u;
const PARTICLE_KIND_BILLBOARD: u32 = 1u;
const PARTICLE_KIND_MESH: u32 = 2u;
const PARTICLE_KIND_TRAIL: u32 = 3u;
const BILLBOARD_ALIGNMENT_VIEW: u32 = 0u;
const BILLBOARD_ALIGNMENT_FACING: u32 = 1u;
const BILLBOARD_ALIGNMENT_LOCAL: u32 = 2u;
const BILLBOARD_ALIGNMENT_GLOBAL: u32 = 3u;
const BILLBOARD_ALIGNMENT_DIRECTION: u32 = 4u;
const MESH_ALIGNMENT_VIEW: u32 = 0u;
const MESH_ALIGNMENT_FACING: u32 = 1u;
const MESH_ALIGNMENT_LOCAL: u32 = 2u;
const MESH_ALIGNMENT_GLOBAL: u32 = 3u;

/// Selects the view-projection matrix for a mono or stereo draw.
fn select_view_proj(draw: PerDrawUniforms, view_idx: u32) -> mat4x4<f32> {
    if (view_idx == 0u) {
        return draw.view_proj_left;
    }
    return draw.view_proj_right;
}

/// `true` when the bound position stream has already been transformed into world space.
fn position_stream_is_world_space(draw: PerDrawUniforms) -> bool {
    return draw._pad.x > 0.5 * POSITION_STREAM_WORLD_SPACE_FLAG;
}

/// Bit mask indicating, for each local probe,
/// if it is of lower importance than its predecessor
fn reflection_probe_importance_mask(draw: PerDrawUniforms) -> u32 {
    let packed = bitcast<u32>(draw._pad.y);
    return packed & 0xFFFFu;
}

/// Fallback reflection probe atlas index packed into the per-draw metadata.
fn fallback_reflection_probe_index(draw: PerDrawUniforms) -> u32 {
    let packed = bitcast<u32>(draw._pad.y);
    return packed >> 16u;
}

/// Reflection probe atlas indices packed into the per-draw metadata.
fn local_reflection_probe_indices(draw: PerDrawUniforms) -> vec4<u32> {
    let packed_z = bitcast<u32>(draw._pad.z);
    let packed_w = bitcast<u32>(draw._pad.w);
    return vec4<u32>(packed_z & 0xFFFFu, packed_z >> 16u, packed_w & 0xFFFFu, packed_w >> 16u);
}

/// Returns whether any fallback or local reflection probe is selected.
fn has_reflection_probe_selection(draw: PerDrawUniforms) -> bool {
    let locals = local_reflection_probe_indices(draw);
    return fallback_reflection_probe_index(draw) != 0u || any(locals != vec4<u32>(0u));
}

fn particle_kind(draw: PerDrawUniforms) -> u32 {
    return u32(max(draw.particle_header.x, 0.0) + 0.5);
}

fn particle_alignment(draw: PerDrawUniforms) -> u32 {
    return u32(max(draw.particle_header.y, 0.0) + 0.5);
}

fn particle_min_screen_size(draw: PerDrawUniforms) -> f32 {
    return max(draw.particle_header.z, 0.0);
}

fn particle_max_screen_size(draw: PerDrawUniforms) -> f32 {
    return max(draw.particle_header.w, 0.0);
}

fn particle_color(draw: PerDrawUniforms) -> vec4<f32> {
    return draw.particle_tint;
}

fn particle_frame_index(draw: PerDrawUniforms) -> u32 {
    if (draw.particle_extra.y >= 4294967040.0) {
        return 0xffffffffu;
    }
    return u32(max(draw.particle_extra.y, 0.0) + 0.5);
}

fn particle_trail_generates_lighting_data(draw: PerDrawUniforms) -> bool {
    return draw.particle_extra.w > 0.5;
}
