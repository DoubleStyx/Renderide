//! Overlay Unlit (`Shader "OverlayUnlit"`): front/behind unlit layers with Unity's two overlay
//! depth tests (`Greater` behind, `LEqual` front).
//!
//! Froox variant bits populate `_RenderideVariantBits`; this shader decodes its sorted
//! `UniqueKeywords` locally. The keyword groups gate texture sampling, alpha test, vertex-color
//! multiply, vertex-color color-space conversion, polar UV, RGB premultiply, and alpha intensity.


//#texture_default _BehindTex white
//#texture_default _FrontTex white
//#mat_default _BehindColor vec4 0.5 0.5 0.5 0.5
//#mat_default _FrontColor vec4 1.0 1.0 1.0 1.0
//#mat_default _PolarPow float 1.0

#import renderide::frame::globals as rg
#import renderide::material::variant_bits as vb
#import renderide::material::vertex_color as vc
#import renderide::mesh::vertex as mv
#import renderide::core::uv as uvu

struct OverlayUnlitMaterial {
    _BehindColor: vec4<f32>,
    _FrontColor: vec4<f32>,
    _BehindTex_ST: vec4<f32>,
    _FrontTex_ST: vec4<f32>,
    _Cutoff: f32,
    _PolarPow: f32,
    _RenderideVariantBits: u32,
}

const OVERLAYUNLIT_KW_ALPHATEST: u32 = 1u << 0u;
const OVERLAYUNLIT_KW_MUL_ALPHA_INTENSITY: u32 = 1u << 1u;
const OVERLAYUNLIT_KW_MUL_RGB_BY_ALPHA: u32 = 1u << 2u;
const OVERLAYUNLIT_KW_POLARUV: u32 = 1u << 3u;
const OVERLAYUNLIT_KW_TEXTURE: u32 = 1u << 4u;
const OVERLAYUNLIT_KW_VERTEX_HDRSRGB_COLOR: u32 = 1u << 5u;
const OVERLAYUNLIT_KW_VERTEX_LINEAR_COLOR: u32 = 1u << 6u;
const OVERLAYUNLIT_KW_VERTEX_SRGB_COLOR: u32 = 1u << 7u;
const OVERLAYUNLIT_KW_VERTEXCOLORS: u32 = 1u << 8u;

@group(1) @binding(0) var<uniform> mat: OverlayUnlitMaterial;
@group(1) @binding(1) var _BehindTex: texture_2d<f32>;
@group(1) @binding(2) var _BehindTex_sampler: sampler;
@group(1) @binding(3) var _FrontTex: texture_2d<f32>;
@group(1) @binding(4) var _FrontTex_sampler: sampler;

fn overlayunlit_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn kw_ALPHATEST() -> bool {
    return overlayunlit_kw(OVERLAYUNLIT_KW_ALPHATEST);
}

fn kw_MUL_ALPHA_INTENSITY() -> bool {
    return overlayunlit_kw(OVERLAYUNLIT_KW_MUL_ALPHA_INTENSITY);
}

fn kw_MUL_RGB_BY_ALPHA() -> bool {
    return overlayunlit_kw(OVERLAYUNLIT_KW_MUL_RGB_BY_ALPHA);
}

fn kw_POLARUV() -> bool {
    return overlayunlit_kw(OVERLAYUNLIT_KW_POLARUV);
}

fn kw_TEXTURE() -> bool {
    return overlayunlit_kw(OVERLAYUNLIT_KW_TEXTURE);
}

fn kw_VERTEX_HDRSRGB_COLOR() -> bool {
    return overlayunlit_kw(OVERLAYUNLIT_KW_VERTEX_HDRSRGB_COLOR);
}

fn kw_VERTEX_SRGB_COLOR() -> bool {
    return overlayunlit_kw(OVERLAYUNLIT_KW_VERTEX_SRGB_COLOR);
}

fn kw_VERTEXCOLORS() -> bool {
    return overlayunlit_kw(OVERLAYUNLIT_KW_VERTEXCOLORS);
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
    if (!kw_TEXTURE()) {
        return tint;
    }
    if (kw_POLARUV()) {
        let mapped = uvu::polar_mapping(uv, st, mat._PolarPow);
        return textureSampleGrad(tex, samp, mapped.uv, mapped.ddx_uv, mapped.ddy_uv) * tint;
    }
    return textureSample(tex, samp, uvu::apply_st(uv, st)) * tint;
}

fn apply_vertex_color(color_in: vec4<f32>, vertex_color: vec4<f32>) -> vec4<f32> {
    if (!kw_VERTEXCOLORS()) {
        return color_in;
    }
    var vc_linear = vertex_color;
    if (kw_VERTEX_HDRSRGB_COLOR()) {
        vc_linear = vc::srgb_to_linear_hdr(vc_linear);
    } else if (kw_VERTEX_SRGB_COLOR()) {
        vc_linear = vc::srgb_to_linear_ldr(vc_linear);
    }
    return color_in * vc_linear;
}

fn finalize_layer_color(color_in: vec4<f32>, vertex_color: vec4<f32>) -> vec4<f32> {
    if (kw_ALPHATEST() && color_in.a <= mat._Cutoff) {
        discard;
    }

    var color = apply_vertex_color(color_in, vertex_color);
    if (kw_MUL_RGB_BY_ALPHA()) {
        color = vec4<f32>(color.rgb * color.a, color.a);
    }

    if (kw_MUL_ALPHA_INTENSITY()) {
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
    return finalize_layer_color(color, in.color);
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
    return finalize_layer_color(color, in.color);
}
