//! FresnelLerp (`Shader "FresnelLerp"`): blends two fresnel material sets by `_Lerp` or `_LerpTex`.
//!
//! Froox variant bits populate `_RenderideVariantBits`; this shader decodes FresnelLerp's
//! shader-specific keyword bits locally.

//#texture_default _FarTex0 white
//#texture_default _NearTex0 white
//#texture_default _FarTex1 white
//#texture_default _NearTex1 white
//#texture_default _LerpTex black
//#texture_default _NormalMap0 bump
//#texture_default _NormalMap1 bump

#import renderide::frame::globals as rg
#import renderide::pbs::normal as pnorm
#import renderide::material::fresnel as mf
#import renderide::material::variant_bits as vb
#import renderide::mesh::vertex as mv
#import renderide::core::uv as uvu
#import renderide::core::normal_decode as nd

struct FresnelLerpMaterial {
    _FarColor0: vec4<f32>,
    _NearColor0: vec4<f32>,
    _FarColor1: vec4<f32>,
    _NearColor1: vec4<f32>,
    _FarTex0_ST: vec4<f32>,
    _NearTex0_ST: vec4<f32>,
    _FarTex1_ST: vec4<f32>,
    _NearTex1_ST: vec4<f32>,
    _LerpTex_ST: vec4<f32>,
    _NormalMap0_ST: vec4<f32>,
    _NormalMap1_ST: vec4<f32>,
    _Lerp: f32,
    _Exp0: f32,
    _Exp1: f32,
    _GammaCurve: f32,
    _LerpPolarPow: f32,
    _RenderideVariantBits: u32,
    _pad0: vec2<u32>,
}

const FRESNELLERP_KW_LERPTEX: u32 = 1u << 0u;
const FRESNELLERP_KW_LERPTEX_POLARUV: u32 = 1u << 1u;
const FRESNELLERP_KW_MULTI_VALUES: u32 = 1u << 2u;
const FRESNELLERP_KW_NORMALMAP: u32 = 1u << 3u;
const FRESNELLERP_KW_TEXTURE: u32 = 1u << 4u;

@group(1) @binding(0)  var<uniform> mat: FresnelLerpMaterial;
@group(1) @binding(1)  var _FarTex0: texture_2d<f32>;
@group(1) @binding(2)  var _FarTex0_sampler: sampler;
@group(1) @binding(3)  var _NearTex0: texture_2d<f32>;
@group(1) @binding(4)  var _NearTex0_sampler: sampler;
@group(1) @binding(5)  var _FarTex1: texture_2d<f32>;
@group(1) @binding(6)  var _FarTex1_sampler: sampler;
@group(1) @binding(7)  var _NearTex1: texture_2d<f32>;
@group(1) @binding(8)  var _NearTex1_sampler: sampler;
@group(1) @binding(9)  var _LerpTex: texture_2d<f32>;
@group(1) @binding(10) var _LerpTex_sampler: sampler;
@group(1) @binding(11) var _NormalMap0: texture_2d<f32>;
@group(1) @binding(12) var _NormalMap0_sampler: sampler;
@group(1) @binding(13) var _NormalMap1: texture_2d<f32>;
@group(1) @binding(14) var _NormalMap1_sampler: sampler;

fn fresnellerp_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn kw_LERPTEX() -> bool {
    return fresnellerp_kw(FRESNELLERP_KW_LERPTEX);
}

fn kw_LERPTEX_POLARUV() -> bool {
    return fresnellerp_kw(FRESNELLERP_KW_LERPTEX_POLARUV);
}

fn kw_MULTI_VALUES() -> bool {
    return fresnellerp_kw(FRESNELLERP_KW_MULTI_VALUES);
}

fn kw_NORMALMAP() -> bool {
    return fresnellerp_kw(FRESNELLERP_KW_NORMALMAP);
}

fn kw_TEXTURE() -> bool {
    return fresnellerp_kw(FRESNELLERP_KW_TEXTURE);
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

fn compute_lerp(uv: vec2<f32>) -> f32 {
    var l = mat._Lerp;
    if (kw_LERPTEX()) {
        l = textureSample(_LerpTex, _LerpTex_sampler, uvu::apply_st(uv, mat._LerpTex_ST)).r;
        if (kw_MULTI_VALUES()) {
            l = l * mat._Lerp;
        }
    } else if (kw_LERPTEX_POLARUV()) {
        let polar_uv = uvu::apply_st(uvu::polar_uv(uv, mat._LerpPolarPow), mat._LerpTex_ST);
        l = textureSample(_LerpTex, _LerpTex_sampler, polar_uv).r;
        if (kw_MULTI_VALUES()) {
            l = l * mat._Lerp;
        }
    }
    return l;
}

fn sample_normal(uv: vec2<f32>, world_n: vec3<f32>, world_t: vec4<f32>, l: f32) -> vec3<f32> {
    var n = normalize(world_n);
    let t = normalize(world_t);
    if (kw_NORMALMAP()) {
        let n0 = textureSample(_NormalMap0, _NormalMap0_sampler, uvu::apply_st(uv, mat._NormalMap0_ST));
        let n1 = textureSample(_NormalMap1, _NormalMap1_sampler, uvu::apply_st(uv, mat._NormalMap1_ST));
        let ts_n = nd::decode_ts_normal_with_placeholder_sample(mix(n0, n1, vec4<f32>(l)), 1.0);
        let tbn = pnorm::orthonormal_tbn(n, t);
        n = normalize(tbn * ts_n);
    }
    return n;
}

//#pass forward
@fragment
fn fs_main(in: mv::WorldVertexOutput) -> @location(0) vec4<f32> {
    let l = compute_lerp(in.primary_uv);
    let n = sample_normal(in.primary_uv, in.world_n, in.world_t, l);
    let view_dir = rg::view_dir_for_world_pos(in.world_pos, in.view_layer);

    let exp = mix(mat._Exp0, mat._Exp1, l);
    let fresnel = mf::view_angle_fresnel(n, view_dir, exp, mat._GammaCurve);

    var far_color = mix(mat._FarColor0, mat._FarColor1, l);
    var near_color = mix(mat._NearColor0, mat._NearColor1, l);
    if (kw_TEXTURE()) {
        let far_tex0 = textureSample(_FarTex0, _FarTex0_sampler, uvu::apply_st(in.primary_uv, mat._FarTex0_ST));
        let far_tex1 = textureSample(_FarTex1, _FarTex1_sampler, uvu::apply_st(in.primary_uv, mat._FarTex1_ST));
        let near_tex0 = textureSample(_NearTex0, _NearTex0_sampler, uvu::apply_st(in.primary_uv, mat._NearTex0_ST));
        let near_tex1 = textureSample(_NearTex1, _NearTex1_sampler, uvu::apply_st(in.primary_uv, mat._NearTex1_ST));
        far_color = far_color * mix(far_tex0, far_tex1, l);
        near_color = near_color * mix(near_tex0, near_tex1, l);
    }

    return rg::retain_globals_additive(mix(near_color, far_color, fresnel));
}
