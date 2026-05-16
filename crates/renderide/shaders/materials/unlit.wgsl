//! World Unlit (`Shader "Unlit"`): texture x tint, optional alpha test,
//! optional UV-shift from a packed offset texture, vertex color, stereo texture transform,
//! polar UVs, normal-map display mode, and alpha mask.
//!
//! Build emits `unlit_default` / `unlit_multiview` targets via [`MULTIVIEW`](https://docs.rs/naga_oil).
//! `@group(1)` identifiers match Unity material property names (`_Color`, `_Tex`, `_MaskTex`, `_OffsetTex`, ...)
//! so host binding picks them up by reflection.
//!
//! Per-frame bindings (`@group(0)`) are imported from `globals.wgsl` so composed targets match the frame bind group layout used by the renderer.
//! Per-draw uniforms (`@group(2)`) use [`renderide::draw::per_draw`].
//!
//! Froox variant bits populate `_RenderideVariantBits`; this shader decodes Unlit's
//! shader-specific keyword bits locally.

//#texture_default _Tex white
//#texture_default _OffsetTex black
//#texture_default _MaskTex white
//#mat_default _Color vec4 1.0 1.0 1.0 1.0
//#mat_default _OffsetMagnitude vec4 0.1 0.1 0.0 0.0
//#mat_default _PolarPow float 1.0

#import renderide::core::texture_sampling as ts
#import renderide::frame::globals as rg
#import renderide::material::alpha as ma
#import renderide::material::variant_bits as vb
#import renderide::material::vertex_color as vc
#import renderide::mesh::vertex as mv
#import renderide::core::normal_decode as nd
#import renderide::draw::per_draw as pd
#import renderide::core::uv as uvu

struct UnlitMaterial {
    _Color: vec4<f32>,
    _Tex_ST: vec4<f32>,
    _RightEye_ST: vec4<f32>,
    _MaskTex_ST: vec4<f32>,
    _OffsetTex_ST: vec4<f32>,
    _OffsetMagnitude: vec4<f32>,
    _Cutoff: f32,
    _PolarPow: f32,
    _RenderideVariantBits: u32,
    _Tex_LodBias: f32,
    _OffsetTex_LodBias: f32,
    _MaskTex_LodBias: f32,
}

const UNLIT_KW_ALPHATEST: u32 = 1u << 0u;
const UNLIT_KW_COLOR: u32 = 1u << 1u;
const UNLIT_KW_MASK_TEXTURE_CLIP: u32 = 1u << 2u;
const UNLIT_KW_MASK_TEXTURE_MUL: u32 = 1u << 3u;
const UNLIT_KW_MUL_ALPHA_INTENSITY: u32 = 1u << 4u;
const UNLIT_KW_MUL_RGB_BY_ALPHA: u32 = 1u << 5u;
const UNLIT_KW_OFFSET_TEXTURE: u32 = 1u << 6u;
const UNLIT_KW_POLARUV: u32 = 1u << 7u;
const UNLIT_KW_RIGHT_EYE_ST: u32 = 1u << 8u;
const UNLIT_KW_TEXTURE: u32 = 1u << 9u;
const UNLIT_KW_TEXTURE_NORMALMAP: u32 = 1u << 10u;
const UNLIT_KW_VERTEX_LINEAR_COLOR: u32 = 1u << 11u;
const UNLIT_KW_VERTEX_SRGB_COLOR: u32 = 1u << 12u;
const UNLIT_KW_VERTEXCOLORS: u32 = 1u << 13u;

@group(1) @binding(0) var<uniform> mat: UnlitMaterial;
@group(1) @binding(1) var _Tex: texture_2d<f32>;
@group(1) @binding(2) var _Tex_sampler: sampler;
@group(1) @binding(3) var _OffsetTex: texture_2d<f32>;
@group(1) @binding(4) var _OffsetTex_sampler: sampler;
@group(1) @binding(5) var _MaskTex: texture_2d<f32>;
@group(1) @binding(6) var _MaskTex_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) @interpolate(flat) view_layer: u32,
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
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
    let world_p = mv::world_position(d, pos);
#ifdef MULTIVIEW
    let view_layer = view_idx;
#else
    let view_layer = 0u;
#endif
    let vp = mv::select_view_proj(d, view_layer);

    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.uv = uv;
    out.color = color;
    out.view_layer = view_layer;
    return out;
}

fn main_texture_st(view_layer: u32) -> vec4<f32> {
    if (kw_RIGHT_EYE_ST() && view_layer != 0u) {
        return mat._RightEye_ST;
    }
    return mat._Tex_ST;
}

fn unlit_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn kw_ALPHATEST() -> bool {
    return unlit_kw(UNLIT_KW_ALPHATEST);
}

fn kw_COLOR() -> bool {
    return unlit_kw(UNLIT_KW_COLOR);
}

fn kw_MASK_TEXTURE_CLIP() -> bool {
    return unlit_kw(UNLIT_KW_MASK_TEXTURE_CLIP);
}

fn kw_MASK_TEXTURE_MUL() -> bool {
    return unlit_kw(UNLIT_KW_MASK_TEXTURE_MUL);
}

fn kw_MUL_ALPHA_INTENSITY() -> bool {
    return unlit_kw(UNLIT_KW_MUL_ALPHA_INTENSITY);
}

fn kw_MUL_RGB_BY_ALPHA() -> bool {
    return unlit_kw(UNLIT_KW_MUL_RGB_BY_ALPHA);
}

fn kw_OFFSET_TEXTURE() -> bool {
    return unlit_kw(UNLIT_KW_OFFSET_TEXTURE);
}

fn kw_POLARUV() -> bool {
    return unlit_kw(UNLIT_KW_POLARUV);
}

fn kw_RIGHT_EYE_ST() -> bool {
    return unlit_kw(UNLIT_KW_RIGHT_EYE_ST);
}

fn kw_TEXTURE() -> bool {
    return unlit_kw(UNLIT_KW_TEXTURE);
}

fn kw_TEXTURE_NORMALMAP() -> bool {
    return unlit_kw(UNLIT_KW_TEXTURE_NORMALMAP);
}

fn kw_VERTEX_SRGB_COLOR() -> bool {
    return unlit_kw(UNLIT_KW_VERTEX_SRGB_COLOR);
}

fn kw_VERTEXCOLORS() -> bool {
    return unlit_kw(UNLIT_KW_VERTEXCOLORS);
}

fn vertex_color_to_linear(color: vec4<f32>) -> vec4<f32> {
    if (kw_VERTEX_SRGB_COLOR()) {
        return vc::srgb_to_linear_ldr(color);
    }
    return color;
}

//#pass forward
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let use_texture = kw_TEXTURE() || kw_TEXTURE_NORMALMAP();
    let use_color = kw_COLOR();
    let use_polar_uv = use_texture && kw_POLARUV();
    let main_st = main_texture_st(in.view_layer);

    var uv_main: vec2<f32>;
    var ddx_uv: vec2<f32>;
    var ddy_uv: vec2<f32>;
    if (use_polar_uv) {
        let mapped = uvu::polar_mapping(in.uv, main_st, max(mat._PolarPow, 1e-4));
        uv_main = mapped.uv;
        ddx_uv = mapped.ddx_uv;
        ddy_uv = mapped.ddy_uv;
    } else {
        uv_main = uvu::apply_st(in.uv, main_st);
        ddx_uv = dpdx(uv_main);
        ddy_uv = dpdy(uv_main);
    }

    if (use_texture && kw_OFFSET_TEXTURE()) {
        let uv_off = uvu::apply_st(in.uv, mat._OffsetTex_ST);
        let offset_s = ts::sample_tex_2d(_OffsetTex, _OffsetTex_sampler, uv_off, mat._OffsetTex_LodBias);
        uv_main = uv_main + offset_s.xy * mat._OffsetMagnitude.xy;
    }

    var color: vec4<f32>;
    var main_texture_alpha = 1.0;
    if (use_texture) {
        var tex_color: vec4<f32>;
        if (use_polar_uv) {
            tex_color = textureSampleGrad(_Tex, _Tex_sampler, uv_main, ddx_uv, ddy_uv);
        } else {
            tex_color = ts::sample_tex_2d(_Tex, _Tex_sampler, uv_main, mat._Tex_LodBias);
        }
        main_texture_alpha = tex_color.a;
        if (kw_TEXTURE_NORMALMAP()) {
            tex_color = vec4<f32>(nd::decode_ts_normal_with_placeholder_sample(tex_color, 1.0) * 0.5 + vec3<f32>(0.5), 1.0);
        }
        if (use_color) {
            tex_color = tex_color * mat._Color;
        }
        color = tex_color;
    } else if (use_color) {
        color = mat._Color;
    } else {
        color = vec4<f32>(1.0);
    }

    let alpha_test = kw_ALPHATEST();
    let mask_clip = kw_MASK_TEXTURE_CLIP();
    let mask_mul = kw_MASK_TEXTURE_MUL();
    let mul_rgb_by_alpha = kw_MUL_RGB_BY_ALPHA();

    let uv_mask = uvu::apply_st(in.uv, mat._MaskTex_ST);
    var mask_lum_for_clip = 1.0;

    if (mask_mul || mask_clip) {
        let mask_sample = ts::sample_tex_2d(_MaskTex, _MaskTex_sampler, uv_mask, mat._MaskTex_LodBias);
        let mask_lum = ma::mask_luminance(mask_sample);
        mask_lum_for_clip = mask_lum;

        if (mask_mul) {
            color.a = color.a * mask_lum;
        }
        if (mask_clip && mask_lum <= mat._Cutoff) {
            discard;
        }
    }

    if (alpha_test && !mask_clip) {
        var clip_alpha = color.a;
        if (use_texture && kw_TEXTURE_NORMALMAP()) {
            clip_alpha = main_texture_alpha;
            if (use_color) {
                clip_alpha = clip_alpha * mat._Color.a;
            }
            if (mask_mul) {
                clip_alpha = clip_alpha * mask_lum_for_clip;
            }
        }
        if (clip_alpha <= mat._Cutoff) {
            discard;
        }
    }

    if (kw_VERTEXCOLORS()) {
        color = color * vertex_color_to_linear(in.color);
    }

    if (mul_rgb_by_alpha) {
        color = vec4<f32>(ma::apply_premultiply(color.rgb, color.a, true), color.a);
    }

    if (kw_MUL_ALPHA_INTENSITY()) {
        color = vec4<f32>(color.rgb, ma::alpha_intensity(color.a, color.rgb));
    }

    return rg::retain_globals_additive(color);
}
