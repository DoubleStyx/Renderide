//! FresnelLerp (`Shader "FresnelLerp"`): blends two fresnel material sets by `_Lerp` or `_LerpTex`.
//!
//! Mirrors the Unity keyword/property surface for `_TEXTURE`, `_NORMALMAP`, `_LERPTEX`,
//! `_LERPTEX_POLARUV`, and `_MULTI_VALUES`.


#import renderide::globals as rg
#import renderide::pbs::normal as pnorm
#import renderide::material::fresnel as mf
#import renderide::mesh::vertex as mv
#import renderide::uv_utils as uvu
#import renderide::normal_decode as nd

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
    _TEXTURE: f32,
    _NORMALMAP: f32,
    _LERPTEX: f32,
    _LERPTEX_POLARUV: f32,
    _MULTI_VALUES: f32,
}

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
    return mv::world_model_normal_vertex_main(instance_index, view_idx, pos, n, t, uv);
#else
    return mv::world_model_normal_vertex_main(instance_index, 0u, pos, n, t, uv);
#endif
}

fn compute_lerp(uv: vec2<f32>) -> f32 {
    var l = mat._Lerp;
    if (uvu::kw_enabled(mat._LERPTEX)) {
        l = textureSample(_LerpTex, _LerpTex_sampler, uvu::apply_st(uv, mat._LerpTex_ST)).r;
        if (uvu::kw_enabled(mat._MULTI_VALUES)) {
            l = l * mat._Lerp;
        }
    } else if (uvu::kw_enabled(mat._LERPTEX_POLARUV)) {
        let polar_uv = uvu::apply_st(uvu::polar_uv(uv, mat._LerpPolarPow), mat._LerpTex_ST);
        l = textureSample(_LerpTex, _LerpTex_sampler, polar_uv).r;
        if (uvu::kw_enabled(mat._MULTI_VALUES)) {
            l = l * mat._Lerp;
        }
    }
    return clamp(l, 0.0, 1.0);
}

fn sample_normal(uv: vec2<f32>, world_n: vec3<f32>, world_t: vec4<f32>, l: f32) -> vec3<f32> {
    var n = normalize(world_n);
    let t = normalize(world_t);
    if (uvu::kw_enabled(mat._NORMALMAP)) {
        let n0 = textureSample(_NormalMap0, _NormalMap0_sampler, uvu::apply_st(uv, mat._NormalMap0_ST)).xyz;
        let n1 = textureSample(_NormalMap1, _NormalMap1_sampler, uvu::apply_st(uv, mat._NormalMap1_ST)).xyz;
        let ts_n = nd::decode_ts_normal_with_placeholder(mix(n0, n1, vec3<f32>(l)), 1.0);
        let tbn = pnorm::orthonormal_tbn(n, t);
        n = normalize(tbn * ts_n);
    }
    return n;
}

fn sample_set_color(
    far_tex: texture_2d<f32>,
    far_sampler: sampler,
    near_tex: texture_2d<f32>,
    near_sampler: sampler,
    uv: vec2<f32>,
    far_st: vec4<f32>,
    near_st: vec4<f32>,
    far_color: vec4<f32>,
    near_color: vec4<f32>,
    fresnel: f32,
) -> vec4<f32> {
    var far = far_color;
    var near = near_color;
    if (uvu::kw_enabled(mat._TEXTURE)) {
        far = far * textureSample(far_tex, far_sampler, uvu::apply_st(uv, far_st));
        near = near * textureSample(near_tex, near_sampler, uvu::apply_st(uv, near_st));
    }
    return mix(near, far, fresnel);
}

//#pass forward
@fragment
fn fs_main(in: mv::WorldVertexOutput) -> @location(0) vec4<f32> {
    let l = compute_lerp(in.primary_uv);
    let n = sample_normal(in.primary_uv, in.world_n, in.world_t, l);
    let view_dir = rg::view_dir_for_world_pos(in.world_pos, in.view_layer);

    let exp = mix(mat._Exp0, mat._Exp1, l);
    let fresnel = mf::view_angle_fresnel(n, view_dir, exp, mat._GammaCurve);

    let col0 = sample_set_color(
        _FarTex0,
        _FarTex0_sampler,
        _NearTex0,
        _NearTex0_sampler,
        in.primary_uv,
        mat._FarTex0_ST,
        mat._NearTex0_ST,
        mat._FarColor0,
        mat._NearColor0,
        fresnel,
    );
    let col1 = sample_set_color(
        _FarTex1,
        _FarTex1_sampler,
        _NearTex1,
        _NearTex1_sampler,
        in.primary_uv,
        mat._FarTex1_ST,
        mat._NearTex1_ST,
        mat._FarColor1,
        mat._NearColor1,
        fresnel,
    );

    return rg::retain_globals_additive(mix(col0, col1, l));
}
