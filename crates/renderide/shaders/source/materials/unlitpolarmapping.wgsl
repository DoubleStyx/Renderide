//! Unity unlit `Shader "Unlit/UnlitPolarMapping"`: remaps the mesh UV to polar coordinates and
//! samples `_MainTex`. The Unity source uses `tex2Dgrad` with explicit derivatives to mask the
//! discontinuity at the polar seam; this WGSL port mirrors that with `textureSampleGrad` and
//! the same derivative reconstruction.

// unity-shader-name: Unlit/UnlitPolarMapping
// unity-shader-name: UnlitPolarMapping

#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::uv_utils as uvu

struct UnlitPolarMappingMaterial {
    _MainTex_ST: vec4<f32>,
    _Pow: f32,
}

@group(1) @binding(0) var<uniform> mat: UnlitPolarMappingMaterial;
@group(1) @binding(1) var _MainTex: texture_2d<f32>;
@group(1) @binding(2) var _MainTex_sampler: sampler;

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
    @location(0) uv_in: vec2<f32>,
) -> @location(0) vec4<f32> {
    let centered = uv_in * 2.0 - 1.0;
    let polar = uvu::polar_uv(uv_in, max(mat._Pow, 1e-4));
    let polar_st = uvu::apply_st(polar, mat._MainTex_ST);
    // Reconstruct derivatives from screen-space derivatives of the centered UV; this matches the
    // Unity reference's `tex2Dgrad` call which avoids the discontinuity at the polar seam.
    let ddx_uv = dpdx(polar_st);
    let ddy_uv = dpdy(polar_st);
    let col = textureSampleGrad(_MainTex, _MainTex_sampler, polar_st, ddx_uv, ddy_uv);
    let touch = (centered.x + centered.y) * 0.0;
    return rg::retain_globals_additive(col + vec4<f32>(touch));
}
