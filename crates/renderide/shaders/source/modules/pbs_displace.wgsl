//! Shared vertex-stage displacement helpers for the Unity PBSDisplace material family.
//!
//! Material files keep their Unity property structs and texture binding names local; this module
//! only centralizes the vertex math so metallic, specular, transparent, and shadow variants remain
//! bit-for-bit aligned when displacement semantics change.

#import renderide::uv_utils as uvu

#define_import_path renderide::pbs::displace

/// Object-space position and UV after applying enabled displacement keywords.
struct DisplacementResult {
    /// Object-space position passed to the draw model matrix.
    position: vec3<f32>,
    /// Vertex UV after optional `_UVOffsetMap` warp.
    uv: vec2<f32>,
}

/// Applies the PBSDisplace vertex-stage offset keywords.
fn apply_vertex_offsets(
    position: vec3<f32>,
    normal: vec3<f32>,
    uv0: vec2<f32>,
    vertex_offset_enabled: bool,
    uv_offset_enabled: bool,
    object_position_offset_enabled: bool,
    vertex_position_offset_enabled: bool,
    vertex_offset_st: vec4<f32>,
    position_offset_magnitude: vec3<f32>,
    vertex_offset_magnitude: f32,
    vertex_offset_bias: f32,
    uv_offset_magnitude: f32,
    uv_offset_bias: f32,
    vertex_offset_map: texture_2d<f32>,
    vertex_offset_sampler: sampler,
    uv_offset_map: texture_2d<f32>,
    uv_offset_sampler: sampler,
    position_offset_map: texture_2d<f32>,
    position_offset_sampler: sampler,
) -> DisplacementResult {
    var displaced = position;
    var uv = uv0;

    if (vertex_offset_enabled) {
        let uv_off = uvu::apply_st(uv0, vertex_offset_st);
        let h = textureSampleLevel(vertex_offset_map, vertex_offset_sampler, uv_off, 0.0).r;
        displaced = displaced + normal * (h * vertex_offset_magnitude + vertex_offset_bias);
    }
    if (uv_offset_enabled) {
        let s = textureSampleLevel(uv_offset_map, uv_offset_sampler, uv0, 0.0).rg;
        uv = uv + (s * uv_offset_magnitude + vec2<f32>(uv_offset_bias));
    }
    if (object_position_offset_enabled || vertex_position_offset_enabled) {
        let off = textureSampleLevel(position_offset_map, position_offset_sampler, uv0, 0.0).rgb;
        displaced = displaced + off * position_offset_magnitude;
    }

    return DisplacementResult(displaced, uv);
}
