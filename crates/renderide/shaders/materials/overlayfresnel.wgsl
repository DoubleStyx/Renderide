//! OverlayFresnel (`Shader "OverlayFresnel"`): two-pass fresnel overlay.
//!
//! The `behind` pass uses reverse-Z `depth=Less` (Unity `ZTest Greater`) so the fresnel glow paints
//! only where the geometry lies behind existing depth; the `front` pass uses the standard
//! `depth=GreaterEqual` (Unity `ZTest LEqual`) for the visible silhouette.
//!
//! Froox variant bits populate `_RenderideVariantBits`; this shader decodes its sorted
//! `UniqueKeywords` (`_MUL_ALPHA_INTENSITY`, `_NORMALMAP`, `_POLARUV`, `_TEXTURE`).


//#texture_default _BehindFarTex white
//#texture_default _BehindNearTex white
//#texture_default _FrontFarTex white
//#texture_default _FrontNearTex white
//#texture_default _NormalMap bump
//#mat_default _BehindFarColor vec4 0.0 0.0 0.0 1.0
//#mat_default _BehindNearColor vec4 1.0 1.0 1.0 1.0
//#mat_default _Exp float 1.0
//#mat_default _FrontFarColor vec4 0.0 0.0 0.0 1.0
//#mat_default _FrontNearColor vec4 1.0 1.0 1.0 1.0
//#mat_default _GammaCurve float 2.2
//#mat_default _PolarPow float 1.0

#import renderide::frame::globals as rg
#import renderide::material::fresnel as mf
#import renderide::material::variant_bits as vb
#import renderide::mesh::vertex as mv
#import renderide::pbs::normal as pnorm
#import renderide::core::uv as uvu
#import renderide::core::normal_decode as nd

struct OverlayFresnelMaterial {
    _BehindFarColor: vec4<f32>,
    _BehindNearColor: vec4<f32>,
    _FrontFarColor: vec4<f32>,
    _FrontNearColor: vec4<f32>,
    _BehindFarTex_ST: vec4<f32>,
    _BehindNearTex_ST: vec4<f32>,
    _FrontFarTex_ST: vec4<f32>,
    _FrontNearTex_ST: vec4<f32>,
    _NormalMap_ST: vec4<f32>,
    _Exp: f32,
    _GammaCurve: f32,
    _PolarPow: f32,
    _RenderideVariantBits: u32,
}

const OVERLAYFRESNEL_KW_MUL_ALPHA_INTENSITY: u32 = 1u << 0u;
const OVERLAYFRESNEL_KW_NORMALMAP: u32 = 1u << 1u;
const OVERLAYFRESNEL_KW_POLARUV: u32 = 1u << 2u;
const OVERLAYFRESNEL_KW_TEXTURE: u32 = 1u << 3u;

@group(1) @binding(0) var<uniform> mat: OverlayFresnelMaterial;
@group(1) @binding(1) var _BehindFarTex: texture_2d<f32>;
@group(1) @binding(2) var _BehindFarTex_sampler: sampler;
@group(1) @binding(3) var _BehindNearTex: texture_2d<f32>;
@group(1) @binding(4) var _BehindNearTex_sampler: sampler;
@group(1) @binding(5) var _FrontFarTex: texture_2d<f32>;
@group(1) @binding(6) var _FrontFarTex_sampler: sampler;
@group(1) @binding(7) var _FrontNearTex: texture_2d<f32>;
@group(1) @binding(8) var _FrontNearTex_sampler: sampler;
@group(1) @binding(9) var _NormalMap: texture_2d<f32>;
@group(1) @binding(10) var _NormalMap_sampler: sampler;

fn overlayfresnel_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn kw_MUL_ALPHA_INTENSITY() -> bool {
    return overlayfresnel_kw(OVERLAYFRESNEL_KW_MUL_ALPHA_INTENSITY);
}

fn kw_NORMALMAP() -> bool {
    return overlayfresnel_kw(OVERLAYFRESNEL_KW_NORMALMAP);
}

fn kw_POLARUV() -> bool {
    return overlayfresnel_kw(OVERLAYFRESNEL_KW_POLARUV);
}

fn kw_TEXTURE() -> bool {
    return overlayfresnel_kw(OVERLAYFRESNEL_KW_TEXTURE);
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(4) t: vec4<f32>,
) -> mv::WorldVertexOutput {
#ifdef MULTIVIEW
    return mv::world_vertex_main(instance_index, view_idx, pos, n, t, uv);
#else
    return mv::world_vertex_main(instance_index, 0u, pos, n, t, uv);
#endif
}

fn sample_overlay_tex(
    tex: texture_2d<f32>,
    samp: sampler,
    uv: vec2<f32>,
    st: vec4<f32>,
) -> vec4<f32> {
    if (kw_POLARUV()) {
        let mapped = uvu::polar_mapping(uv, st, mat._PolarPow);
        return textureSampleGrad(tex, samp, mapped.uv, mapped.ddx_uv, mapped.ddy_uv);
    }
    return textureSample(tex, samp, uvu::apply_st(uv, st));
}

fn overlay_normal(in: mv::WorldVertexOutput) -> vec3<f32> {
    var n = normalize(in.world_n);
    let t = normalize(in.world_t);
    if (kw_NORMALMAP()) {
        let uv_n = uvu::apply_st(
            in.primary_uv,
            mat._NormalMap_ST,
        );
        let tbn = pnorm::orthonormal_tbn(n, t);
        let ts_n = nd::decode_ts_normal_with_placeholder_sample(
            textureSample(_NormalMap, _NormalMap_sampler, uv_n),
            1.0,
        );
        n = normalize(tbn * ts_n);
    }
    return n;
}

fn fresnel_value(in: mv::WorldVertexOutput, apply_gamma: bool) -> f32 {
    let n = overlay_normal(in);
    let view_dir = rg::view_dir_for_world_pos(in.world_pos, in.view_layer);
    return mf::view_angle_fresnel(n, view_dir, mat._Exp, select(1.0, mat._GammaCurve, apply_gamma));
}

fn apply_alpha_intensity(color_in: vec4<f32>) -> vec4<f32> {
    var color = color_in;
    if (kw_MUL_ALPHA_INTENSITY()) {
        let factor = (color.r + color.g + color.b) * 0.3333333;
        color.a = color.a * factor * factor;
    }
    return color;
}

fn layer_color(
    tint: vec4<f32>,
    tex: texture_2d<f32>,
    samp: sampler,
    uv: vec2<f32>,
    st: vec4<f32>,
) -> vec4<f32> {
    if (kw_TEXTURE()) {
        return tint * sample_overlay_tex(tex, samp, uv, st);
    }
    return tint;
}

//#pass overlay_behind
@fragment
fn fs_main_behind(in: mv::WorldVertexOutput) -> @location(0) vec4<f32> {
    let fresnel = fresnel_value(in, false);
    let far_color = layer_color(
        mat._BehindFarColor,
        _BehindFarTex,
        _BehindFarTex_sampler,
        in.primary_uv,
        mat._BehindFarTex_ST,
    );
    let near_color = layer_color(
        mat._BehindNearColor,
        _BehindNearTex,
        _BehindNearTex_sampler,
        in.primary_uv,
        mat._BehindNearTex_ST,
    );
    let color = apply_alpha_intensity(mf::near_far_color(near_color, far_color, fresnel));
    return rg::retain_globals_additive(color);
}

//#pass overlay_front
@fragment
fn fs_main_front(in: mv::WorldVertexOutput) -> @location(0) vec4<f32> {
    let fresnel = fresnel_value(in, true);
    let far_color = layer_color(
        mat._FrontFarColor,
        _FrontFarTex,
        _FrontFarTex_sampler,
        in.primary_uv,
        mat._FrontFarTex_ST,
    );
    let near_color = layer_color(
        mat._FrontNearColor,
        _FrontNearTex,
        _FrontNearTex_sampler,
        in.primary_uv,
        mat._FrontNearTex_ST,
    );
    let color = apply_alpha_intensity(mf::near_far_color(near_color, far_color, fresnel));
    return rg::retain_globals_additive(color);
}
