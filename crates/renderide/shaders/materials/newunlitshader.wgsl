//! Unity unlit `Shader "Unlit/NewUnlitShader"`: Shader Forge diffuse output driven by
//! `_node_2829` and scene lighting.


//#mat_default _node_2829 vec4 0.5 0.5 0.5 1.0

#import renderide::lighting::diffuse as dl
#import renderide::core::math as rmath
#import renderide::draw::per_draw as pd
#import renderide::mesh::vertex as mv

struct NewUnlitMaterial {
    _node_2829: vec4<f32>,
}

@group(1) @binding(0) var<uniform> mat: NewUnlitMaterial;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) @interpolate(flat) view_layer: u32,
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
#ifdef MULTIVIEW
    let view_layer = view_idx;
#else
    let view_layer = 0u;
#endif
    let world_p = mv::world_position(d, pos);
    let vp = mv::select_view_proj(d, view_layer);
    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.world_n = rmath::safe_normalize(d.normal_matrix * n.xyz, vec3<f32>(0.0, 1.0, 0.0));
    out.view_layer = view_layer;
    return out;
}

//#pass type=forward offset=0,1
@fragment
fn fs_main(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let n = rmath::safe_normalize(world_n, vec3<f32>(0.0, 1.0, 0.0));
    let lit = dl::shade_clustered_diffuse(frag_pos.xy, world_pos, n, mat._node_2829.rgb, view_layer);
    return vec4<f32>(lit, 1.0);
}
