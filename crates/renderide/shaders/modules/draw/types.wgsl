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
fn reflection_probe_indices(draw: PerDrawUniforms) -> vec3<u32> {
    let packed_y = bitcast<u32>(draw._pad.y);
    let packed_z = bitcast<u32>(draw._pad.z);
    return vec3<u32>(packed_y & 0xFFFFu, packed_y >> 16u, packed_z & 0xFFFFu);
}

/// Number of local reflection probe hits represented in the per-draw metadata.
/// A single local hit may still carry a render-space fallback in the second atlas index.
fn reflection_probe_hit_count(draw: PerDrawUniforms) -> u32 {
    let packed = bitcast<u32>(draw._pad.z);
    return min(packed >> 16u, 2u);
}
