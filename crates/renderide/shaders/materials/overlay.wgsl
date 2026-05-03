//! Unity unlit `Shader "Unlit/Overlay"`: texture x tint with fixed `ZTest Always`, `ZWrite Off`,
//! and `Blend SrcAlpha OneMinusSrcAlpha` for HUD-style overlays.


#import renderide::globals as rg
#import renderide::mesh::vertex as mv
#import renderide::uv_utils as uvu

struct OverlayMaterial {
    _Blend: vec4<f32>,
    _MainTexture_ST: vec4<f32>,
}

@group(1) @binding(0) var<uniform> mat: OverlayMaterial;
@group(1) @binding(1) var _MainTexture: texture_2d<f32>;
@group(1) @binding(2) var _MainTexture_sampler: sampler;

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
    let material_uv = uvu::apply_st(uv, mat._MainTexture_ST);
#ifdef MULTIVIEW
    return mv::uv_vertex_main(instance_index, view_idx, pos, material_uv);
#else
    return mv::uv_vertex_main(instance_index, 0u, pos, material_uv);
#endif
}

//#pass overlay_always
@fragment
fn fs_main(
    @location(0) uv: vec2<f32>,
) -> @location(0) vec4<f32> {
    let s = textureSample(_MainTexture, _MainTexture_sampler, uv);
    return rg::retain_globals_additive(vec4<f32>(s.rgb * mat._Blend.rgb, s.a * mat._Blend.a));
}
