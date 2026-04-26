//! Unity `Shader "Invisible"`: vertex collapses to origin and the fragment unconditionally
//! discards. Used as a hit-volume material that contributes nothing to color or depth.

// unity-shader-name: Invisible

#import renderide::globals as rg
#import renderide::per_draw as pd

struct InvisibleMaterial {
    _pad: vec4<f32>,
}

@group(1) @binding(0) var<uniform> mat: InvisibleMaterial;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) _pos: vec4<f32>,
    @location(1) _n: vec4<f32>,
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
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
    out.clip_pos = vp * vec4<f32>(0.0, 0.0, 0.0, 1.0);
    return out;
}

//#pass forward
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    discard;
    return rg::retain_globals_additive(vec4<f32>(mat._pad.x * 0.0));
}
