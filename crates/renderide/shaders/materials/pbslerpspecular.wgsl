//! Unity PBS lerp specular (`Shader "PBSLerpSpecular"`): specular workflow blending between two
//! material sets with `_Lerp` or `_LerpTex`.
//!
//! This mirrors the same keyword/property surface as the Unity shader:
//! `_LERPTEX`, `_ALBEDOTEX`, `_EMISSIONTEX`, `_NORMALMAP`, `_SPECULARMAP`,
//! `_OCCLUSION`, `_MULTI_VALUES`, `_DUALSIDED`, `_ALPHACLIP`.

#import renderide::mesh::vertex as mv
#import renderide::pbs::lighting as plight
#import renderide::pbs::normal as pnorm
#import renderide::pbs::sampling as psamp
#import renderide::pbs::surface as psurf
#import renderide::alpha_clip_sample as acs
#import renderide::uv_utils as uvu
#import renderide::normal_decode as nd

struct PbsLerpSpecularMaterial {
    _Color: vec4<f32>,
    _Color1: vec4<f32>,
    _SpecularColor: vec4<f32>,
    _SpecularColor1: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _EmissionColor1: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _MainTex_StorageVInverted: f32,
    _MainTex1_ST: vec4<f32>,
    _MainTex1_StorageVInverted: f32,
    _LerpTex_ST: vec4<f32>,
    _Lerp: f32,
    _NormalScale: f32,
    _NormalScale1: f32,
    _AlphaClip: f32,
    _LERPTEX: f32,
    _ALBEDOTEX: f32,
    _EMISSIONTEX: f32,
    _NORMALMAP: f32,
    _SPECULARMAP: f32,
    _OCCLUSION: f32,
    _MULTI_VALUES: f32,
    _DUALSIDED: f32,
    _ALPHACLIP: f32,
}

@group(1) @binding(0)  var<uniform> mat: PbsLerpSpecularMaterial;
@group(1) @binding(1)  var _MainTex: texture_2d<f32>;
@group(1) @binding(2)  var _MainTex_sampler: sampler;
@group(1) @binding(3)  var _MainTex1: texture_2d<f32>;
@group(1) @binding(4)  var _MainTex1_sampler: sampler;
@group(1) @binding(5)  var _LerpTex: texture_2d<f32>;
@group(1) @binding(6)  var _LerpTex_sampler: sampler;
@group(1) @binding(7)  var _NormalMap: texture_2d<f32>;
@group(1) @binding(8)  var _NormalMap_sampler: sampler;
@group(1) @binding(9)  var _NormalMap1: texture_2d<f32>;
@group(1) @binding(10) var _NormalMap1_sampler: sampler;
@group(1) @binding(11) var _EmissionMap: texture_2d<f32>;
@group(1) @binding(12) var _EmissionMap_sampler: sampler;
@group(1) @binding(13) var _EmissionMap1: texture_2d<f32>;
@group(1) @binding(14) var _EmissionMap1_sampler: sampler;
@group(1) @binding(15) var _Occlusion: texture_2d<f32>;
@group(1) @binding(16) var _Occlusion_sampler: sampler;
@group(1) @binding(17) var _Occlusion1: texture_2d<f32>;
@group(1) @binding(18) var _Occlusion1_sampler: sampler;
@group(1) @binding(19) var _SpecularMap: texture_2d<f32>;
@group(1) @binding(20) var _SpecularMap_sampler: sampler;
@group(1) @binding(21) var _SpecularMap1: texture_2d<f32>;
@group(1) @binding(22) var _SpecularMap1_sampler: sampler;

fn sample_normal_world(
    uv0: vec2<f32>,
    uv1: vec2<f32>,
    world_n: vec3<f32>,
    world_t: vec4<f32>,
    front_facing: bool,
    lerp_factor: f32,
) -> vec3<f32> {
    if (!uvu::kw_enabled(mat._NORMALMAP)) {
        var n = normalize(world_n);
        if (uvu::kw_enabled(mat._DUALSIDED) && !front_facing) {
            n = -n;
        }
        return n;
    }

    let tbn = pnorm::orthonormal_tbn(world_n, world_t);
    let ts0 = nd::decode_ts_normal_with_placeholder_sample(
        textureSample(_NormalMap, _NormalMap_sampler, uv0),
        mat._NormalScale,
    );
    let ts1 = nd::decode_ts_normal_with_placeholder_sample(
        textureSample(_NormalMap1, _NormalMap1_sampler, uv1),
        mat._NormalScale1,
    );
    var ts = normalize(mix(ts0, ts1, vec3<f32>(lerp_factor)));
    if (uvu::kw_enabled(mat._DUALSIDED) && !front_facing) {
        ts.z = -ts.z;
    }
    return normalize(tbn * ts);
}

fn compute_lerp_factor(uv_lerp: vec2<f32>) -> f32 {
    var l = mat._Lerp;
    if (uvu::kw_enabled(mat._LERPTEX)) {
        l = textureSample(_LerpTex, _LerpTex_sampler, uv_lerp).r;
        if (uvu::kw_enabled(mat._MULTI_VALUES)) {
            l = l * mat._Lerp;
        }
    }
    return clamp(l, 0.0, 1.0);
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
    @location(4) t: vec4<f32>,
) -> mv::WorldVertexOutput {
#ifdef MULTIVIEW
    return mv::world_vertex_main(instance_index, view_idx, pos, n, t, uv0);
#else
    return mv::world_vertex_main(instance_index, 0u, pos, n, t, uv0);
#endif
}

//#pass forward
@fragment
fn fs_main(
    @builtin(position) frag_pos: vec4<f32>,
    @builtin(front_facing) front_facing: bool,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) world_t: vec4<f32>,
    @location(3) uv0_raw: vec2<f32>,
    @location(4) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let uv_main0 = uvu::apply_st_for_storage(uv0_raw, mat._MainTex_ST, mat._MainTex_StorageVInverted);
    let uv_main1 = uvu::apply_st_for_storage(uv0_raw, mat._MainTex1_ST, mat._MainTex1_StorageVInverted);
    let uv_lerp = uvu::apply_st(uv0_raw, mat._LerpTex_ST);
    let l = compute_lerp_factor(uv_lerp);

    var c0 = mat._Color;
    var c1 = mat._Color1;
    var clip_a = mix(mat._Color.a, mat._Color1.a, l);
    if (uvu::kw_enabled(mat._ALBEDOTEX)) {
        c0 = c0 * textureSample(_MainTex, _MainTex_sampler, uv_main0);
        c1 = c1 * textureSample(_MainTex1, _MainTex1_sampler, uv_main1);
        clip_a = mix(
            mat._Color.a * acs::texture_alpha_base_mip(_MainTex, _MainTex_sampler, uv_main0),
            mat._Color1.a * acs::texture_alpha_base_mip(_MainTex1, _MainTex1_sampler, uv_main1),
            l,
        );
    }

    let c = mix(c0, c1, l);
    if (uvu::kw_enabled(mat._ALPHACLIP) && clip_a <= mat._AlphaClip) {
        discard;
    }

    let base_color = c.rgb;
    let alpha = c.a;

    var occlusion0 = 1.0;
    var occlusion1 = 1.0;
    if (uvu::kw_enabled(mat._OCCLUSION)) {
        occlusion0 = textureSample(_Occlusion, _Occlusion_sampler, uv_main0).r;
        occlusion1 = textureSample(_Occlusion1, _Occlusion1_sampler, uv_main1).r;
    }
    let occlusion = mix(occlusion0, occlusion1, l);

    var emission0 = mat._EmissionColor.xyz;
    var emission1 = mat._EmissionColor1.xyz;
    if (uvu::kw_enabled(mat._EMISSIONTEX)) {
        emission0 =
            emission0 * textureSample(_EmissionMap, _EmissionMap_sampler, uv_main0).xyz;
        emission1 =
            emission1 * textureSample(_EmissionMap1, _EmissionMap1_sampler, uv_main1).xyz;
    }
    let emission = mix(emission0, emission1, l);

    var spec0 = mat._SpecularColor;
    var spec1 = mat._SpecularColor1;
    if (uvu::kw_enabled(mat._SPECULARMAP)) {
        spec0 = textureSample(_SpecularMap, _SpecularMap_sampler, uv_main0);
        spec1 = textureSample(_SpecularMap1, _SpecularMap1_sampler, uv_main1);
        if (uvu::kw_enabled(mat._MULTI_VALUES)) {
            spec0 = spec0 * mat._SpecularColor;
            spec1 = spec1 * mat._SpecularColor1;
        }
    }
    let spec = mix(spec0, spec1, l);
    let f0 = spec.rgb;
    let smoothness = clamp(spec.a, 0.0, 1.0);
    let roughness = psamp::roughness_from_smoothness(smoothness);

    let n = sample_normal_world(uv_main0, uv_main1, world_n, world_t, front_facing, l);

    let surface = psurf::specular(base_color, alpha, f0, roughness, occlusion, n, emission);
    let color = plight::shade_specular_clustered(
        frag_pos.xy,
        world_pos,
        view_layer,
        surface,
        plight::default_lighting_options(),
    );
    return vec4<f32>(color, alpha);
}
