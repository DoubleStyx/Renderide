//! FresnelLerp (`Shader "FresnelLerp"`): blends two fresnel material sets by `_Lerp` or `_LerpTex`.
//!
//! Mirrors the Unity keyword/property surface for `_TEXTURE`, `_NORMALMAP`, `_LERPTEX`,
//! `_LERPTEX_POLARUV`, and `_MULTI_VALUES`.

// unity-shader-name: FresnelLerp

#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::pbs::brdf as brdf
#import renderide::pbs::normal as pnorm
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

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv: vec2<f32>,
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
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
    let world_p = d.model * vec4<f32>(pos.xyz, 1.0);
    let wn = normalize((d.model * vec4<f32>(n.xyz, 0.0)).xyz);
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
    out.world_pos = world_p.xyz;
    out.world_n = wn;
    out.uv = uv;
    return out;
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

fn sample_normal(uv: vec2<f32>, world_n: vec3<f32>, l: f32) -> vec3<f32> {
    var n = normalize(world_n);
    if (uvu::kw_enabled(mat._NORMALMAP)) {
        let n0 = textureSample(_NormalMap0, _NormalMap0_sampler, uvu::apply_st(uv, mat._NormalMap0_ST)).xyz;
        let n1 = textureSample(_NormalMap1, _NormalMap1_sampler, uvu::apply_st(uv, mat._NormalMap1_ST)).xyz;
        let ts_n = nd::decode_ts_normal_with_placeholder(mix(n0, n1, vec3<f32>(l)), 1.0);
        let tbn = pnorm::orthonormal_tbn(n);
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
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let l = compute_lerp(in.uv);
    let n = sample_normal(in.uv, in.world_n, l);
    let view_dir = normalize(rg::frame.camera_world_pos.xyz - in.world_pos);

    let exp = mix(mat._Exp0, mat._Exp1, l);
    let base_fresnel = pow(max(1.0 - abs(dot(n, view_dir)), 0.0), max(exp, 1e-4));
    let fresnel = pow(clamp(base_fresnel, 0.0, 1.0), max(mat._GammaCurve, 1e-4));

    let col0 = sample_set_color(
        _FarTex0,
        _FarTex0_sampler,
        _NearTex0,
        _NearTex0_sampler,
        in.uv,
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
        in.uv,
        mat._FarTex1_ST,
        mat._NearTex1_ST,
        mat._FarColor1,
        mat._NearColor1,
        fresnel,
    );

    return rg::retain_globals_additive(mix(col0, col1, l));
}
