//! Unity unlit `Shader "Unlit/Circle"`: SDF circle (Manhattan distance) with smoothstep edge fade.

// unity-shader-name: Unlit/Circle
// unity-shader-name: Circle

#import renderide::globals as rg
#import renderide::per_draw as pd

struct CircleMaterial {
    _Color: vec4<f32>,
}

@group(1) @binding(0) var<uniform> mat: CircleMaterial;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) _n: vec4<f32>,
    @location(2) uv: vec2<f32>,
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
    let world_p = d.model * vec4<f32>(pos.xyz, 1.0);
#ifdef MULTIVIEW
    var vp: mat4x4<f32>;
    if (view_idx == 0u) {
        vp = d.view_proj_left;
    } else {
        vp = d.view_proj_right;
    }
#else
    let vp = d.view_proj_left;
#endif
    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.uv = uv;
    return out;
}

//#pass forward
@fragment
fn fs_main(
    @location(0) uv: vec2<f32>,
) -> @location(0) vec4<f32> {
    let coord = uv;
    let center = vec2<f32>(0.5, 0.5);
    let dst = dot(abs(coord - center), vec2<f32>(1.0, 1.0));
    let aaf = fwidth(dst);
    let mask = 1.0 - smoothstep(0.2 - aaf, 0.2, dst);
    return rg::retain_globals_additive(vec4<f32>(mat._Color.rgb, mask));
}
