//! Shared vertex payload for screen-space filter materials.

#define_import_path renderide::filter_vertex

#import renderide::per_draw as pd

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) primary_uv: vec2<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) world_n: vec3<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
}

fn select_view_proj(d: pd::PerDrawUniforms, view_idx: u32) -> mat4x4<f32> {
    if (view_idx == 0u) {
        return d.view_proj_left;
    }
    return d.view_proj_right;
}

fn vertex_main(
    instance_index: u32,
    view_idx: u32,
    pos: vec4<f32>,
    n: vec4<f32>,
    primary_uv: vec2<f32>,
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
    let world_p = d.model * vec4<f32>(pos.xyz, 1.0);
    let vp = select_view_proj(d, view_idx);
    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.primary_uv = primary_uv;
    out.world_pos = world_p.xyz;
    out.world_n = normalize(d.normal_matrix * n.xyz);
    out.view_layer = view_idx;
    return out;
}
