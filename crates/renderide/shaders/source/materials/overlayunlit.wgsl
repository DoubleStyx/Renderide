//! Overlay Unlit (`Shader "OverlayUnlit"`): front/behind unlit layers composed in a single pass.
//!
//! Unity implements this as two passes with different depth tests (`Greater` and `LEqual`).
//! The current renderer has a single fixed forward pass, so this WGSL path approximates the effect
//! by sampling both layers and compositing `front over behind` in one fragment shader.
//!
//! Keyword-style float fields mirror Unity `#pragma multi_compile` values:
//! `_POLARUV`, `_MUL_RGB_BY_ALPHA`, `_MUL_ALPHA_INTENSITY`.


#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::alpha_clip_sample as acs
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
    _ALPHATEST_ON: f32,
}

@group(1) @binding(0) var<uniform> mat: OverlayUnlitMaterial;
@group(1) @binding(1) var _BehindTex: texture_2d<f32>;
@group(1) @binding(2) var _BehindTex_sampler: sampler;
@group(1) @binding(3) var _FrontTex: texture_2d<f32>;
@group(1) @binding(4) var _FrontTex_sampler: sampler;

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

fn sample_layer(
    tex: texture_2d<f32>,
    samp: sampler,
    tint: vec4<f32>,
    uv: vec2<f32>,
    st: vec4<f32>,
) -> vec4<f32> {
    let use_polar = mat._POLARUV > 0.99;
    let sample_uv = select(uvu::apply_st(uv, st), uvu::apply_st(uvu::polar_uv(uv, mat._PolarPow), st), use_polar);
    return textureSample(tex, samp, sample_uv) * tint;
}

/// Same UV as [`sample_layer`], base mip — for `_Cutoff` vs composited alpha only.
fn sample_layer_lod0(
    tex: texture_2d<f32>,
    samp: sampler,
    tint: vec4<f32>,
    uv: vec2<f32>,
    st: vec4<f32>,
) -> vec4<f32> {
    let use_polar = mat._POLARUV > 0.99;
    let sample_uv = select(uvu::apply_st(uv, st), uvu::apply_st(uvu::polar_uv(uv, mat._PolarPow), st), use_polar);
    return acs::texture_rgba_base_mip(tex, samp, sample_uv) * tint;
}

fn alpha_over(front: vec4<f32>, behind: vec4<f32>) -> vec4<f32> {
    let out_a = front.a + behind.a * (1.0 - front.a);
    if (out_a <= 1e-6) {
        return vec4<f32>(0.0);
    }
    let out_rgb =
        (front.rgb * front.a + behind.rgb * behind.a * (1.0 - front.a)) / out_a;
    return vec4<f32>(out_rgb, out_a);
}

//#material forward
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let behind = sample_layer(
        _BehindTex,
        _BehindTex_sampler,
        mat._BehindColor,
        in.uv,
        mat._BehindTex_ST,
    );
    let front = sample_layer(
        _FrontTex,
        _FrontTex_sampler,
        mat._FrontColor,
        in.uv,
        mat._FrontTex_ST,
    );

    var color = alpha_over(front, behind);

    let behind_clip = sample_layer_lod0(
        _BehindTex,
        _BehindTex_sampler,
        mat._BehindColor,
        in.uv,
        mat._BehindTex_ST,
    );
    let front_clip = sample_layer_lod0(
        _FrontTex,
        _FrontTex_sampler,
        mat._FrontColor,
        in.uv,
        mat._FrontTex_ST,
    );
    let color_clip = alpha_over(front_clip, behind_clip);

    if (uvu::kw_enabled(mat._ALPHATEST_ON) && color_clip.a <= mat._Cutoff) {
        discard;
    }

    if (mat._MUL_RGB_BY_ALPHA > 0.99) {
        color = vec4<f32>(color.rgb * color.a, color.a);
    }

    if (mat._MUL_ALPHA_INTENSITY > 0.99) {
        let lum = (color.r + color.g + color.b) * 0.33333334;
        color.a = color.a * lum;
    }

    return rg::retain_globals_additive(color);
}
