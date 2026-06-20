//! Shared vertex-stage displacement helpers for the Unity PBSDisplace material family.
//!
//! Material files keep their Unity property structs and texture binding names local; this module
//! only centralizes the displacement math so metallic, specular, and transparent variants remain
//! aligned when displacement semantics change.

#import renderide::core::uv as uvu
#import renderide::core::texture_sampling as ts
#import renderide::draw::types as dt
#import renderide::mesh::vertex as mv

#define_import_path renderide::pbs::displace

/// Object-space position and UV after applying enabled displacement keywords.
struct DisplacementResult {
    /// Object-space position passed to the draw model matrix.
    position: vec3<f32>,
    /// Raw mesh UV forwarded to the fragment stage. Unity applies `_UVOffsetMap` in `surf`.
    uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) world_t: vec4<f32>,
    @location(3) base_uv: vec2<f32>,
    @location(4) @interpolate(flat) view_layer: u32,
}

fn vertex_output(
    draw: dt::PerDrawUniforms,
    instance_index: u32,
    view_idx: u32,
    n: vec4<f32>,
    t: vec4<f32>,
    displaced: vec3<f32>,
    uv: vec2<f32>,
) -> VertexOutput {
    let world_p = draw.model * vec4<f32>(displaced, 1.0);
    let vp = mv::select_view_proj(draw, view_idx);

    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.world_n = mv::world_normal(draw, n);
    out.world_t = mv::world_tangent(draw, t);
    out.base_uv = uv;
    out.view_layer = mv::packed_view_layer(instance_index, view_idx);
    return out;
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
    position_offset_st: vec4<f32>,
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
            let position_xz = select(vertex_world_xz, object_xz, object_position_offset_enabled);
            let position_uv = uvu::apply_st(position_xz, position_offset_st);
            let uv_offset = textureSampleLevel(
                position_offset_map,
                position_offset_sampler,
                position_uv,
                0.0,
            ).xy * position_offset_magnitude;
            vertex_uv = vertex_uv + uv_offset;
        }

        let uv_off = uvu::apply_st(vertex_uv, vertex_offset_st);
        let h = textureSampleLevel(vertex_offset_map, vertex_offset_sampler, uv_off, 0.0).r;
        displaced = displaced + normal * (h * vertex_offset_magnitude + vertex_offset_bias);
    }

    return DisplacementResult(displaced, uv0);
}

/// Applies the PBSDisplace fragment-stage `_UVOffsetMap` warp to the already transformed main UV.
fn apply_fragment_uv_offset(
    base_main_uv: vec2<f32>,
    raw_uv0: vec2<f32>,
    uv_offset_enabled: bool,
    uv_offset_st: vec4<f32>,
    uv_offset_magnitude: f32,
    uv_offset_bias: f32,
    uv_offset_lod_bias: f32,
    uv_offset_map: texture_2d<f32>,
    uv_offset_sampler: sampler,
) -> vec2<f32> {
    if (!uv_offset_enabled) {
        return base_main_uv;
    }

    let offset_uv = uvu::apply_st(raw_uv0, uv_offset_st);
    let offset_sample = ts::sample_tex_2d(uv_offset_map, uv_offset_sampler, offset_uv, uv_offset_lod_bias).rg;
    return base_main_uv + (offset_sample * uv_offset_magnitude + vec2<f32>(uv_offset_bias));
}
