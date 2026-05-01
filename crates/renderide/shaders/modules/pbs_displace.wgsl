//! Shared vertex-stage displacement helpers for the Unity PBSDisplace material family.
//!
//! Material files keep their Unity property structs and texture binding names local; this module
//! only centralizes the vertex math so metallic, specular, and transparent variants remain aligned
//! when displacement semantics change.

#import renderide::uv_utils as uvu

#define_import_path renderide::pbs::displace

/// Object-space position after applying enabled vertex displacement keywords.
struct DisplacementResult {
    /// Object-space position passed to the draw model matrix.
    position: vec3<f32>,
    /// Raw mesh UV forwarded to the fragment stage. Unity applies `_UVOffsetMap` in `surf`.
    uv: vec2<f32>,
}

/// Applies the PBSDisplace vertex-stage offset keywords.
fn apply_vertex_offsets(
    position: vec3<f32>,
    normal: vec3<f32>,
    uv0: vec2<f32>,
    model: mat4x4<f32>,
    vertex_offset_enabled: bool,
    object_position_offset_enabled: bool,
    vertex_position_offset_enabled: bool,
    vertex_offset_st: vec4<f32>,
    vertex_offset_storage_v_inverted: f32,
    position_offset_st: vec4<f32>,
    position_offset_storage_v_inverted: f32,
    position_offset_magnitude: vec2<f32>,
    vertex_offset_magnitude: f32,
    vertex_offset_bias: f32,
    vertex_offset_map: texture_2d<f32>,
    vertex_offset_sampler: sampler,
    position_offset_map: texture_2d<f32>,
    position_offset_sampler: sampler,
) -> DisplacementResult {
    var displaced = position;

    if (vertex_offset_enabled) {
        var vertex_uv = uv0;
        if (object_position_offset_enabled || vertex_position_offset_enabled) {
            let object_xz = model[3].xz;
            let vertex_world_xz = (model * vec4<f32>(position, 1.0)).xz;
            let position_xz = select(object_xz, vertex_world_xz, vertex_position_offset_enabled);
            let position_uv = uvu::apply_st_for_storage(
                position_xz,
                position_offset_st,
                position_offset_storage_v_inverted,
            );
            let uv_offset = textureSampleLevel(
                position_offset_map,
                position_offset_sampler,
                position_uv,
                0.0,
            ).xy * position_offset_magnitude;
            vertex_uv = vertex_uv + uv_offset;
        }

        let uv_off = uvu::apply_st_for_storage(
            vertex_uv,
            vertex_offset_st,
            vertex_offset_storage_v_inverted,
        );
        let h = textureSampleLevel(vertex_offset_map, vertex_offset_sampler, uv_off, 0.0).r;
        displaced = displaced + normal * (h * vertex_offset_magnitude + vertex_offset_bias);
    }

    return DisplacementResult(displaced, uv0);
}
