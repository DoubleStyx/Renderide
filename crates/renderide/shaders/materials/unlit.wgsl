//! World Unlit (`Shader "Unlit"`): texture x tint, optional alpha test,
//! optional UV-shift from a packed offset texture, vertex color, stereo texture transform,
//! polar UVs, normal-map display mode, and alpha mask.
//!
//! Build emits `unlit_default` / `unlit_multiview` targets via [`MULTIVIEW`](https://docs.rs/naga_oil).
//! `@group(1)` identifiers match Unity material property names (`_Color`, `_Tex`, `_MaskTex`, `_OffsetTex`, ...)
//! so host binding picks them up by reflection.
//!
//! Per-frame bindings (`@group(0)`) are imported from `globals.wgsl` so composed targets match the frame bind group layout used by the renderer.
//! Per-draw uniforms (`@group(2)`) use [`renderide::per_draw`].
//!
//! Some Unity keywords are reconstructed from host-visible material state in the uniform
//! packer. Unobservable keyword fields still remain explicit so property blocks can drive them
//! if the host starts sending matching values.

#import renderide::texture_sampling as ts
#import renderide::globals as rg
#import renderide::alpha_clip_sample as acs
#import renderide::material::alpha as ma
#import renderide::mesh::vertex as mv
#import renderide::normal_decode as nd
#import renderide::per_draw as pd
#import renderide::uv_utils as uvu

struct UnlitMaterial {
    _Color: vec4<f32>,
    _Tex_ST: vec4<f32>,
    _RightEye_ST: vec4<f32>,
    _MaskTex_ST: vec4<f32>,
    _OffsetTex_ST: vec4<f32>,
    _OffsetMagnitude: vec4<f32>,
    _Cutoff: f32,
    _PolarPow: f32,
    _MUL_RGB_BY_ALPHA: f32,
    _MUL_ALPHA_INTENSITY: f32,
    _TEXTURE_NORMALMAP: f32,
    _MASK_TEXTURE_MUL: f32,
    _MASK_TEXTURE_CLIP: f32,
    _POLARUV: f32,
    _RIGHT_EYE_ST: f32,
    _ALPHATEST: f32,
    _ALPHATEST_ON: f32,
    _ALPHABLEND_ON: f32,
    _Tex_LodBias: f32,
    _OffsetTex_LodBias: f32,
    _MaskTex_LodBias: f32,
}

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
    if (uvu::kw_enabled(mat._RIGHT_EYE_ST) && view_layer != 0u) {
        return mat._RightEye_ST;
    }
    return mat._Tex_ST;
}

//#pass forward
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let use_polar_uv = uvu::kw_enabled(mat._POLARUV);
    let main_st = main_texture_st(in.view_layer);

    let uv_off = uvu::apply_st(in.uv, mat._OffsetTex_ST);
    let offset_s = ts::sample_tex_2d(_OffsetTex, _OffsetTex_sampler, uv_off, mat._OffsetTex_LodBias);

    var uv_main: vec2<f32>;
    if (use_polar_uv) {
        let polar = uvu::polar_uv(in.uv, max(mat._PolarPow, 1e-4));
        uv_main = uvu::apply_st(polar, main_st);
    } else {
        uv_main = uvu::apply_st(in.uv, main_st);
    }
    let ddx_uv = dpdx(uv_main);
    let ddy_uv = dpdy(uv_main);
    uv_main = uv_main + offset_s.xy * mat._OffsetMagnitude.xy;

    var t: vec4<f32>;
    if (use_polar_uv) {
        t = textureSampleGrad(_Tex, _Tex_sampler, uv_main, ddx_uv, ddy_uv);
    } else {
        t = ts::sample_tex_2d(_Tex, _Tex_sampler, uv_main, mat._Tex_LodBias);
    }
    if (uvu::kw_enabled(mat._TEXTURE_NORMALMAP)) {
        t = vec4<f32>(nd::decode_ts_normal_with_placeholder_sample(t, 1.0) * 0.5 + vec3<f32>(0.5), 1.0);
    }
    var color = mat._Color * t;

    let alpha_test = uvu::kw_enabled(mat._ALPHATEST_ON) || uvu::kw_enabled(mat._ALPHATEST);
    let alpha_blend = uvu::kw_enabled(mat._ALPHABLEND_ON);
    let mask_clip = uvu::kw_enabled(mat._MASK_TEXTURE_CLIP);
    let mask_mul = uvu::kw_enabled(mat._MASK_TEXTURE_MUL) || (!mask_clip && (alpha_test || alpha_blend));
    let mul_rgb_by_alpha = uvu::kw_enabled(mat._MUL_RGB_BY_ALPHA);

    let uv_mask = uvu::apply_st(in.uv, mat._MaskTex_ST);

    if (mask_mul || mask_clip) {
        let mask_sample = ts::sample_tex_2d(_MaskTex, _MaskTex_sampler, uv_mask, mat._MaskTex_LodBias);
        let mask_lum = ma::mask_luminance(mask_sample);
        let mask_clip_alpha = acs::mask_luminance_mul_base_mip(_MaskTex, _MaskTex_sampler, uv_mask);

        if (mask_mul) {
            color.a = color.a * mask_lum;
        }
        if (mask_clip && mask_clip_alpha <= mat._Cutoff) {
            discard;
        }
    }

    if (alpha_test && !mask_clip) {
        var tex_clip_alpha = mat._Color.a * acs::texture_alpha_base_mip(_Tex, _Tex_sampler, uv_main);
        if (mask_mul) {
            tex_clip_alpha = tex_clip_alpha * acs::mask_luminance_mul_base_mip(_MaskTex, _MaskTex_sampler, uv_mask);
        }
        if (tex_clip_alpha <= mat._Cutoff) {
            discard;
        }
    }

    color = color * in.color;

    if (mul_rgb_by_alpha) {
        color = vec4<f32>(ma::apply_premultiply(color.rgb, color.a, true), color.a);
    }

    if (uvu::kw_enabled(mat._MUL_ALPHA_INTENSITY)) {
        color = vec4<f32>(color.rgb, ma::alpha_intensity(color.a, color.rgb));
    }

    return rg::retain_globals_additive(color);
}
