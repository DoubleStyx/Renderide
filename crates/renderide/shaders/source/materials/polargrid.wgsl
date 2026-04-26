//! Unity unlit `Shader "Unlit/PolarGrid"`: procedural polar grid visualizing radius bands and density.

// unity-shader-name: Unlit/PolarGrid
// unity-shader-name: PolarGrid

#import renderide::globals as rg
#import renderide::per_draw as pd

struct PolarGridMaterial {
    _pad: vec4<f32>,
}

@group(1) @binding(0) var<uniform> mat: PolarGridMaterial;

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
    let centered = uv * 2.0 - 1.0;
    let radius = length(centered);
    let ref_radius = round(radius * 100.0) / 100.0;
    var d = abs(radius - ref_radius) * 100.0;
    let aaf = fwidth(d);
    let raaf = fwidth(radius * 100.0);
    d = 1.0 - smoothstep(0.05 - aaf, 0.05, d);
    let debug = smoothstep(0.15, 0.25, raaf);
    let band = d - debug;
    let col = vec3<f32>(band * debug, band * (1.0 - debug), 0.0);
    let touch = mat._pad.x * 0.0;
    return rg::retain_globals_additive(vec4<f32>(col + vec3<f32>(touch), 1.0));
}
