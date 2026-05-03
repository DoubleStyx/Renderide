//! Overlay Unlit (`Shader "OverlayUnlit"`): front/behind unlit layers with Unity's two overlay
//! depth tests (`Greater` behind, `LEqual` front).
//!
//! Keyword-style float fields mirror Unity `#pragma multi_compile` values:
//! `_POLARUV`, `_MUL_RGB_BY_ALPHA`, `_MUL_ALPHA_INTENSITY`.


#import renderide::globals as rg
#import renderide::alpha_clip_sample as acs
#import renderide::material::sample as ms
#import renderide::mesh::vertex as mv
#import renderide::uv_utils as uvu

struct OverlayUnlitMaterial {
    _BehindColor: vec4<f32>,
    _FrontColor: vec4<f32>,
    _BehindTex_ST: vec4<f32>,
    _FrontTex_ST: vec4<f32>,
    _Cutoff: f32,
    _PolarPow: f32,
    _POLARUV: f32,
    _MUL_RGB_BY_ALPHA: f32,
    _MUL_ALPHA_INTENSITY: f32,
    _ALPHATEST: f32,
}

@group(1) @binding(0) var<uniform> mat: OverlayUnlitMaterial;
@group(1) @binding(1) var _BehindTex: texture_2d<f32>;
@group(1) @binding(2) var _BehindTex_sampler: sampler;
@group(1) @binding(3) var _FrontTex: texture_2d<f32>;
@group(1) @binding(4) var _FrontTex_sampler: sampler;

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) _n: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
) -> mv::UvColorVertexOutput {
#ifdef MULTIVIEW
    return mv::uv_color_vertex_main(instance_index, view_idx, pos, uv, color);
#else
    return mv::uv_color_vertex_main(instance_index, 0u, pos, uv, color);
#endif
}

fn sample_layer(
    tex: texture_2d<f32>,
    samp: sampler,
    tint: vec4<f32>,
    uv: vec2<f32>,
    st: vec4<f32>,
) -> vec4<f32> {
    let sample_uv = ms::sample_uv(
        uv,
        st,
        mat._PolarPow,
        mat._POLARUV > 0.99,
    );
    return textureSample(tex, samp, sample_uv) * tint;
}

/// Same UV as [`sample_layer`], base mip -- for `_Cutoff` vs composited alpha only.
fn sample_layer_lod0(
    tex: texture_2d<f32>,
    samp: sampler,
    tint: vec4<f32>,
    uv: vec2<f32>,
    st: vec4<f32>,
) -> vec4<f32> {
    let sample_uv = ms::sample_uv(
        uv,
        st,
        mat._PolarPow,
        mat._POLARUV > 0.99,
    );
    return acs::texture_rgba_base_mip(tex, samp, sample_uv) * tint;
}

fn finalize_layer_color(color_in: vec4<f32>, clip_color: vec4<f32>, vertex_color: vec4<f32>) -> vec4<f32> {
    if (uvu::kw_enabled(mat._ALPHATEST) && clip_color.a <= mat._Cutoff) {
        discard;
    }

    var color = color_in * vertex_color;
    if (mat._MUL_RGB_BY_ALPHA > 0.99) {
        color = vec4<f32>(color.rgb * color.a, color.a);
    }

    if (mat._MUL_ALPHA_INTENSITY > 0.99) {
        let factor = (color.r + color.g + color.b) * 0.3333333;
        color.a = color.a * factor;
    }

    return rg::retain_globals_additive(color);
}

//#pass overlay_behind
@fragment
fn fs_behind(in: mv::UvColorVertexOutput) -> @location(0) vec4<f32> {
    let color = sample_layer(
        _BehindTex,
        _BehindTex_sampler,
        mat._BehindColor,
        in.uv,
        mat._BehindTex_ST,
    );
    let clip_color = sample_layer_lod0(
        _BehindTex,
        _BehindTex_sampler,
        mat._BehindColor,
        in.uv,
        mat._BehindTex_ST,
    );
    return finalize_layer_color(color, clip_color, in.color);
}

//#pass overlay_front
@fragment
fn fs_front(in: mv::UvColorVertexOutput) -> @location(0) vec4<f32> {
    let color = sample_layer(
        _FrontTex,
        _FrontTex_sampler,
        mat._FrontColor,
        in.uv,
        mat._FrontTex_ST,
    );
    let clip_color = sample_layer_lod0(
        _FrontTex,
        _FrontTex_sampler,
        mat._FrontColor,
        in.uv,
        mat._FrontTex_ST,
    );
    return finalize_layer_color(color, clip_color, in.color);
}
