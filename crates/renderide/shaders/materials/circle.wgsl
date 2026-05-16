//! Unity unlit `Shader "Unlit/Circle"`: SDF circle (Manhattan distance) with smoothstep edge fade.


//#mat_default _Color vec4 1.0 1.0 1.0 1.0

#import renderide::frame::globals as rg
#import renderide::mesh::vertex as mv

struct CircleMaterial {
    _Color: vec4<f32>,
}

@group(1) @binding(0) var<uniform> mat: CircleMaterial;

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) _n: vec4<f32>,
    @location(2) uv: vec2<f32>,
) -> mv::UvVertexOutput {
#ifdef MULTIVIEW
    return mv::uv_vertex_main(instance_index, view_idx, pos, uv);
#else
    return mv::uv_vertex_main(instance_index, 0u, pos, uv);
#endif
}

//#pass transparent_rgb
@fragment
fn fs_main(
    @location(0) uv: vec2<f32>,
) -> @location(0) vec4<f32> {
    let coord = uv;
    let center = vec2<f32>(0.5, 0.5);
    let dst = dot(abs(coord - center), vec2<f32>(1.0, 1.0));
    let aaf = fwidth(dst);
    let mask = 1.0 - smoothstep(0.2 - aaf, 0.2, dst);
    return rg::retain_globals_additive(vec4<f32>(1.0, 1.0, 1.0, mask));
}
