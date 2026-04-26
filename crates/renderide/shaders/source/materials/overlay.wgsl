//! Unity unlit `Shader "Unlit/Overlay"`: texture × tint with `_ZTest=Always` (host-driven) for
//! HUD-style overlays.

// unity-shader-name: Unlit/Overlay
// unity-shader-name: Overlay

#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::uv_utils as uvu

struct OverlayMaterial {
    _Blend: vec4<f32>,
    _MainTexture_ST: vec4<f32>,
}

@group(1) @binding(0) var<uniform> mat: OverlayMaterial;
@group(1) @binding(1) var _MainTexture: texture_2d<f32>;
@group(1) @binding(2) var _MainTexture_sampler: sampler;

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
    out.uv = uvu::apply_st(uv, mat._MainTexture_ST);
    return out;
}

//#pass forward
@fragment
fn fs_main(
    @location(0) uv: vec2<f32>,
) -> @location(0) vec4<f32> {
    let s = textureSample(_MainTexture, _MainTexture_sampler, uv);
    return rg::retain_globals_additive(vec4<f32>(s.rgb * mat._Blend.rgb, s.a * mat._Blend.a));
}
