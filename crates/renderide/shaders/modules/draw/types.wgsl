//! Shared per-draw instance data layout.

#define_import_path renderide::draw::types

struct PerDrawUniforms {
    view_proj_left: mat4x4<f32>,
    view_proj_right: mat4x4<f32>,
    model: mat4x4<f32>,
    /// Cofactor matrix of the upper 3x3 of `model` for singular-safe normal transforms.
    normal_matrix: mat3x3<f32>,
    /// Metadata. `x` marks world-space position streams; `yzw` pack reflection-probe selection.
    _pad: vec4<f32>,
}

/// `_pad.x` marker for world-space position streams.
const POSITION_STREAM_WORLD_SPACE_FLAG: f32 = 1.0;

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

/// Reflection probe atlas indices packed into the per-draw metadata.
fn reflection_probe_indices(draw: PerDrawUniforms) -> vec2<u32> {
    let packed = bitcast<u32>(draw._pad.y);
    return vec2<u32>(packed & 0xFFFFu, packed >> 16u);
}

/// Blend weight for the second reflection probe hit.
fn reflection_probe_second_weight(draw: PerDrawUniforms) -> f32 {
    return clamp(draw._pad.z, 0.0, 1.0);
}

/// Number of local reflection probe hits represented in the per-draw metadata.
/// A single local hit may still carry a render-space fallback in the second atlas index.
fn reflection_probe_hit_count(draw: PerDrawUniforms) -> u32 {
    return min(u32(draw._pad.w + 0.5), 2u);
}
