//! Fresnel (`Shader "Fresnel"`): blends near/far colors from view-angle Fresnel and optional normal/mask textures.
//!
//! Keyword-style float fields mirror Unity `#pragma multi_compile` values:
//! `_TEXTURE`, `_POLARUV`, `_NORMALMAP`, `_MASK_TEXTURE_MUL`, `_MASK_TEXTURE_CLIP`,
//! `_VERTEXCOLORS`, `_MUL_ALPHA_INTENSITY`.

#import renderide::globals as rg
#import renderide::pbs::normal as pnorm
#import renderide::alpha_clip_sample as acs
#import renderide::material::alpha as ma
#import renderide::material::fresnel as mf
#import renderide::material::sample as ms
#import renderide::mesh::vertex as mv
#import renderide::uv_utils as uvu
#import renderide::normal_decode as nd

struct FresnelMaterial {
    _FarColor: vec4<f32>,
    _NearColor: vec4<f32>,
    _FarTex_ST: vec4<f32>,
    _NearTex_ST: vec4<f32>,
    _MaskTex_ST: vec4<f32>,
    _Exp: f32,
    _GammaCurve: f32,
    _NormalScale: f32,
    _Cutoff: f32,
    _PolarPow: f32,
    _TEXTURE: f32,
    _POLARUV: f32,
    _NORMALMAP: f32,
    _MASK_TEXTURE_MUL: f32,
    _MASK_TEXTURE_CLIP: f32,
    _VERTEXCOLORS: f32,
    _MUL_ALPHA_INTENSITY: f32,
    _ALPHATEST_ON: f32,
}

@group(1) @binding(0) var<uniform> mat: FresnelMaterial;
@group(1) @binding(1) var _FarTex: texture_2d<f32>;
@group(1) @binding(2) var _FarTex_sampler: sampler;
@group(1) @binding(3) var _NearTex: texture_2d<f32>;
@group(1) @binding(4) var _NearTex_sampler: sampler;
@group(1) @binding(5) var _NormalMap: texture_2d<f32>;
@group(1) @binding(6) var _NormalMap_sampler: sampler;
@group(1) @binding(7) var _MaskTex: texture_2d<f32>;
@group(1) @binding(8) var _MaskTex_sampler: sampler;

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
) -> mv::WorldColorVertexOutput {
#ifdef MULTIVIEW
    return mv::world_color_vertex_main(instance_index, view_idx, pos, n, uv, color);
#else
    return mv::world_color_vertex_main(instance_index, 0u, pos, n, uv, color);
#endif
}

//#pass forward
@fragment
fn fs_main(in: mv::WorldColorVertexOutput) -> @location(0) vec4<f32> {
    var n = normalize(in.world_n);
    if (mat._NORMALMAP > 0.99) {
        let uv_n = vec2<f32>(in.primary_uv.x, 1.0 - in.primary_uv.y);
        let tbn = pnorm::orthonormal_tbn(n);
        let ts_n = nd::decode_ts_normal_with_placeholder_sample(
            textureSample(_NormalMap, _NormalMap_sampler, uv_n),
            mat._NormalScale,
        );
        n = normalize(tbn * ts_n);
    }

    let view_dir = rg::view_dir_for_world_pos(in.world_pos, in.view_layer);
    let fres = mf::view_angle_fresnel(n, view_dir, mat._Exp, mat._GammaCurve);

    let use_polar = mat._POLARUV > 0.99;
    var far_color = mat._FarColor;
    var near_color = mat._NearColor;
    if (uvu::kw_enabled(mat._TEXTURE)) {
        far_color = far_color * ms::sample_rgba(_FarTex, _FarTex_sampler, in.primary_uv, mat._FarTex_ST, 0.0, mat._PolarPow, use_polar);
        near_color =
            near_color * ms::sample_rgba(_NearTex, _NearTex_sampler, in.primary_uv, mat._NearTex_ST, 0.0, mat._PolarPow, use_polar);
    }

    var color = mf::near_far_color(near_color, far_color, fres);
    var clip_a = color.a;
    if (uvu::kw_enabled(mat._TEXTURE)) {
        let far_clip = mat._FarColor * ms::sample_rgba_lod0(_FarTex, _FarTex_sampler, in.primary_uv, mat._FarTex_ST, mat._PolarPow, use_polar);
        let near_clip = mat._NearColor * ms::sample_rgba_lod0(_NearTex, _NearTex_sampler, in.primary_uv, mat._NearTex_ST, mat._PolarPow, use_polar);
        clip_a = mix(near_clip.a, far_clip.a, clamp(fres, 0.0, 1.0));
    }

    if (mat._MASK_TEXTURE_MUL > 0.99 || mat._MASK_TEXTURE_CLIP > 0.99) {
        let uv_mask = uvu::apply_st(in.primary_uv, mat._MaskTex_ST);
        let mask = textureSample(_MaskTex, _MaskTex_sampler, uv_mask);
        let mul = ma::mask_luminance(mask);
        let mul_clip = acs::mask_luminance_mul_base_mip(_MaskTex, _MaskTex_sampler, uv_mask);

        if (mat._MASK_TEXTURE_MUL > 0.99) {
            color.a = color.a * mul;
            clip_a = clip_a * mul_clip;
        }
        if (mat._MASK_TEXTURE_CLIP > 0.99 && mul_clip <= mat._Cutoff) {
            discard;
        }
    }

    if (!(mat._MASK_TEXTURE_CLIP > 0.99) && uvu::kw_enabled(mat._ALPHATEST_ON) && clip_a <= mat._Cutoff) {
        discard;
    }

    if (uvu::kw_enabled(mat._VERTEXCOLORS)) {
        color = color * in.color;
    }

    if (mat._MUL_ALPHA_INTENSITY > 0.99) {
        color.a = ma::alpha_intensity_squared(color.a, color.rgb);
    }

    return rg::retain_globals_additive(color);
}
