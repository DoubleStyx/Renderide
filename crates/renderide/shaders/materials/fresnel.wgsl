//! Fresnel (`Shader "Fresnel"`): blends near/far colors from view-angle Fresnel and optional normal/mask textures.
//!
//! Froox variant bits populate `_RenderideVariantBits`; this shader decodes Fresnel's
//! shader-specific keyword bits locally.

//#texture_default _FarTex white
//#texture_default _NearTex white
//#texture_default _NormalMap bump
//#texture_default _MaskTex white
//#mat_default _Exp float 1.0
//#mat_default _FarColor vec4 0.0 0.0 0.0 1.0
//#mat_default _GammaCurve float 1.0
//#mat_default _NearColor vec4 1.0 1.0 1.0 1.0
//#mat_default _NormalScale float 1.0
//#mat_default _PolarPow float 1.0

#import renderide::frame::globals as rg
#import renderide::pbs::normal as pnorm
#import renderide::material::alpha as ma
#import renderide::material::fresnel as mf
#import renderide::material::sample as ms
#import renderide::material::variant_bits as vb
#import renderide::material::vertex_color as vc
#import renderide::mesh::vertex as mv
#import renderide::core::uv as uvu
#import renderide::core::normal_decode as nd

struct FresnelMaterial {
    _FarColor: vec4<f32>,
    _NearColor: vec4<f32>,
    _FarTex_ST: vec4<f32>,
    _NearTex_ST: vec4<f32>,
    _MaskTex_ST: vec4<f32>,
    _NormalMap_ST: vec4<f32>,
    _Exp: f32,
    _GammaCurve: f32,
    _NormalScale: f32,
    _Cutoff: f32,
    _PolarPow: f32,
    _RenderideVariantBits: u32,
    _pad0: vec2<u32>,
}

const FRESNEL_KW_ALPHATEST: u32 = 1u << 0u;
const FRESNEL_KW_MASK_TEXTURE_CLIP: u32 = 1u << 1u;
const FRESNEL_KW_MASK_TEXTURE_MUL: u32 = 1u << 2u;
const FRESNEL_KW_MUL_ALPHA_INTENSITY: u32 = 1u << 3u;
const FRESNEL_KW_NORMALMAP: u32 = 1u << 4u;
const FRESNEL_KW_POLARUV: u32 = 1u << 5u;
const FRESNEL_KW_TEXTURE: u32 = 1u << 6u;
const FRESNEL_KW_VERTEX_HDRSRGB_COLOR: u32 = 1u << 7u;
const FRESNEL_KW_VERTEX_LINEAR_COLOR: u32 = 1u << 8u;
const FRESNEL_KW_VERTEX_SRGB_COLOR: u32 = 1u << 9u;
const FRESNEL_KW_VERTEXCOLORS: u32 = 1u << 10u;

@group(1) @binding(0) var<uniform> mat: FresnelMaterial;
@group(1) @binding(1) var _FarTex: texture_2d<f32>;
@group(1) @binding(2) var _FarTex_sampler: sampler;
@group(1) @binding(3) var _NearTex: texture_2d<f32>;
@group(1) @binding(4) var _NearTex_sampler: sampler;
@group(1) @binding(5) var _NormalMap: texture_2d<f32>;
@group(1) @binding(6) var _NormalMap_sampler: sampler;
@group(1) @binding(7) var _MaskTex: texture_2d<f32>;
@group(1) @binding(8) var _MaskTex_sampler: sampler;

fn fresnel_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn kw_ALPHATEST() -> bool {
    return fresnel_kw(FRESNEL_KW_ALPHATEST);
}

fn kw_MASK_TEXTURE_CLIP() -> bool {
    return fresnel_kw(FRESNEL_KW_MASK_TEXTURE_CLIP);
}

fn kw_MASK_TEXTURE_MUL() -> bool {
    return fresnel_kw(FRESNEL_KW_MASK_TEXTURE_MUL);
}

fn kw_MUL_ALPHA_INTENSITY() -> bool {
    return fresnel_kw(FRESNEL_KW_MUL_ALPHA_INTENSITY);
}

fn kw_NORMALMAP() -> bool {
    return fresnel_kw(FRESNEL_KW_NORMALMAP);
}

fn kw_POLARUV() -> bool {
    return fresnel_kw(FRESNEL_KW_POLARUV);
}

fn kw_TEXTURE() -> bool {
    return fresnel_kw(FRESNEL_KW_TEXTURE);
}

fn kw_VERTEX_HDRSRGB_COLOR() -> bool {
    return fresnel_kw(FRESNEL_KW_VERTEX_HDRSRGB_COLOR);
}

fn kw_VERTEX_SRGB_COLOR() -> bool {
    return fresnel_kw(FRESNEL_KW_VERTEX_SRGB_COLOR);
}

fn kw_VERTEXCOLORS() -> bool {
    return fresnel_kw(FRESNEL_KW_VERTEXCOLORS);
}

fn vertex_color_to_linear(color: vec4<f32>) -> vec4<f32> {
    if (kw_VERTEX_HDRSRGB_COLOR()) {
        return vc::srgb_to_linear_hdr(color);
    }
    if (kw_VERTEX_SRGB_COLOR()) {
        return vc::srgb_to_linear_ldr(color);
    }
    return color;
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
    @location(3) color: vec4<f32>,
    @location(4) t: vec4<f32>,
) -> mv::WorldColorVertexOutput {
#ifdef MULTIVIEW
    return mv::world_color_vertex_main(instance_index, view_idx, pos, n, t, uv, color);
#else
    return mv::world_color_vertex_main(instance_index, 0u, pos, n, t, uv, color);
#endif
}

//#pass forward
@fragment
fn fs_main(in: mv::WorldColorVertexOutput) -> @location(0) vec4<f32> {
    var n = normalize(in.world_n);
    if (kw_NORMALMAP()) {
        let uv_n = uvu::apply_st(in.primary_uv, mat._NormalMap_ST);
        let tbn = pnorm::orthonormal_tbn(n, in.world_t);
        let ts_n = nd::decode_ts_normal_with_placeholder_sample(
            textureSample(_NormalMap, _NormalMap_sampler, uv_n),
            mat._NormalScale,
        );
        n = normalize(tbn * ts_n);
    }

    let view_dir = rg::view_dir_for_world_pos(in.world_pos, in.view_layer);
    let fres = mf::view_angle_fresnel(n, view_dir, mat._Exp, mat._GammaCurve);

    let use_polar = kw_POLARUV();
    var far_color = mat._FarColor;
    var near_color = mat._NearColor;
    if (kw_TEXTURE()) {
        far_color = far_color * ms::sample_rgba(_FarTex, _FarTex_sampler, in.primary_uv, mat._FarTex_ST, 0.0, mat._PolarPow, use_polar);
        near_color =
            near_color * ms::sample_rgba(_NearTex, _NearTex_sampler, in.primary_uv, mat._NearTex_ST, 0.0, mat._PolarPow, use_polar);
    }

    var color = mf::near_far_color(near_color, far_color, fres);
    var clip_a = color.a;

    if (kw_MASK_TEXTURE_MUL() || kw_MASK_TEXTURE_CLIP()) {
        let uv_mask = uvu::apply_st(in.primary_uv, mat._MaskTex_ST);
        let mask = textureSample(_MaskTex, _MaskTex_sampler, uv_mask);
        let mul = ma::mask_luminance(mask);

        if (kw_MASK_TEXTURE_MUL()) {
            color.a = color.a * mul;
            clip_a = clip_a * mul;
        }
        if (kw_MASK_TEXTURE_CLIP() && mul <= mat._Cutoff) {
            discard;
        }
    }

    if (!kw_MASK_TEXTURE_CLIP() && kw_ALPHATEST() && clip_a <= mat._Cutoff) {
        discard;
    }

    if (kw_VERTEXCOLORS()) {
        color = color * vertex_color_to_linear(in.color);
    }

    if (kw_MUL_ALPHA_INTENSITY()) {
        color.a = ma::alpha_intensity_squared(color.a, color.rgb);
    }
    color.a = pow(color.a, mat._GammaCurve);

    return rg::retain_globals_additive(color);
}
