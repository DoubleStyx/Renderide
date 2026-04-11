//! Unity PBS lerp specular (`Shader "PBSLerpSpecular"`): specular workflow blending between two
//! material sets with `_Lerp` or `_LerpTex`.
//!
//! This mirrors the same keyword/property surface as the Unity shader:
//! `_LERPTEX`, `_ALBEDOTEX`, `_EMISSIONTEX`, `_NORMALMAP`, `_SPECULARMAP`,
//! `_OCCLUSION`, `_MULTI_VALUES`, `_DUALSIDED`, `_ALPHACLIP`.

// unity-shader-name: PBSLerpSpecular

#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::pbs::brdf as brdf
#import renderide::pbs::cluster as pcls
#import renderide::alpha_clip_sample as acs
#import renderide::unity_st as ust
#import renderide::view_proj as vp

struct PbsLerpSpecularMaterial {
    _Color: vec4<f32>,
    _Color1: vec4<f32>,
    _SpecularColor: vec4<f32>,
    _SpecularColor1: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _EmissionColor1: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _MainTex1_ST: vec4<f32>,
    _LerpTex_ST: vec4<f32>,
    _Lerp: f32,
    _NormalScale: f32,
    _NormalScale1: f32,
    _AlphaClip: f32,
    _Cull: f32,
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

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
}

fn kw_enabled(v: f32) -> bool {
    return v > 0.5;
}

fn sample_normal_world(
    uv0: vec2<f32>,
    uv1: vec2<f32>,
    world_n: vec3<f32>,
    front_facing: bool,
    lerp_factor: f32,
) -> vec3<f32> {
    if (!kw_enabled(mat._NORMALMAP)) {
        var n = normalize(world_n);
        if (kw_enabled(mat._DUALSIDED) && !front_facing) {
            n = -n;
        }
        return n;
    }

    let tbn = brdf::orthonormal_tbn(normalize(world_n));
    let ts0 =
        brdf::decode_ts_normal_placeholder_flat(textureSample(_NormalMap, _NormalMap_sampler, uv0).xyz, mat._NormalScale);
    let ts1 = brdf::decode_ts_normal_placeholder_flat(
        textureSample(_NormalMap1, _NormalMap1_sampler, uv1).xyz,
        mat._NormalScale1,
    );
    var ts = normalize(mix(ts0, ts1, vec3<f32>(lerp_factor)));
    if (kw_enabled(mat._DUALSIDED) && !front_facing) {
        ts.z = -ts.z;
    }
    return normalize(tbn * ts);
}

fn compute_lerp_factor(uv_lerp: vec2<f32>) -> f32 {
    var l = mat._Lerp;
    if (kw_enabled(mat._LERPTEX)) {
        l = textureSample(_LerpTex, _LerpTex_sampler, uv_lerp).r;
        if (kw_enabled(mat._MULTI_VALUES)) {
            l = l * mat._Lerp;
        }
    }
    return clamp(l, 0.0, 1.0);
}

@vertex
fn vs_main(
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> VertexOutput {
    let world_p = pd::draw.model * vec4<f32>(pos.xyz, 1.0);
    let wn = normalize((pd::draw.model * vec4<f32>(n.xyz, 0.0)).xyz);
#ifdef MULTIVIEW
    let vpm = vp::view_projection_for_eye(view_idx);
#else
    let vpm = vp::view_projection_for_eye(0u);
#endif
    var out: VertexOutput;
    out.clip_pos = vpm * world_p;
    out.world_pos = world_p.xyz;
    out.world_n = wn;
    out.uv0 = uv0;
    return out;
}

@fragment
fn fs_main(
    @builtin(position) frag_pos: vec4<f32>,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @builtin(front_facing) front_facing: bool,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0_raw: vec2<f32>,
) -> @location(0) vec4<f32> {
    let uv_main0 = ust::apply_st(uv0_raw, mat._MainTex_ST);
    let uv_main1 = ust::apply_st(uv0_raw, mat._MainTex1_ST);
    let uv_lerp = ust::apply_st(uv0_raw, mat._LerpTex_ST);
    let l = compute_lerp_factor(uv_lerp);

    var c0 = mat._Color;
    var c1 = mat._Color1;
    var clip_a = mix(mat._Color.a, mat._Color1.a, l);
    if (kw_enabled(mat._ALBEDOTEX)) {
        c0 = c0 * textureSample(_MainTex, _MainTex_sampler, uv_main0);
        c1 = c1 * textureSample(_MainTex1, _MainTex1_sampler, uv_main1);
        clip_a = mix(
            mat._Color.a * acs::texture_alpha_base_mip(_MainTex, _MainTex_sampler, uv_main0),
            mat._Color1.a * acs::texture_alpha_base_mip(_MainTex1, _MainTex1_sampler, uv_main1),
            l,
        );
    }

    let c = mix(c0, c1, l);
    if (kw_enabled(mat._ALPHACLIP) && clip_a <= mat._AlphaClip) {
        discard;
    }

    let base_color = c.rgb;
    let alpha = c.a;

    var occlusion0 = 1.0;
    var occlusion1 = 1.0;
    if (kw_enabled(mat._OCCLUSION)) {
        occlusion0 = textureSample(_Occlusion, _Occlusion_sampler, uv_main0).r;
        occlusion1 = textureSample(_Occlusion1, _Occlusion1_sampler, uv_main1).r;
    }
    let occlusion = mix(occlusion0, occlusion1, l);

    var emission0 = mat._EmissionColor.xyz;
    var emission1 = mat._EmissionColor1.xyz;
    if (kw_enabled(mat._EMISSIONTEX)) {
        emission0 =
            emission0 * textureSample(_EmissionMap, _EmissionMap_sampler, uv_main0).xyz;
        emission1 =
            emission1 * textureSample(_EmissionMap1, _EmissionMap1_sampler, uv_main1).xyz;
    }
    let emission = mix(emission0, emission1, l);

    var spec0 = mat._SpecularColor;
    var spec1 = mat._SpecularColor1;
    if (kw_enabled(mat._SPECULARMAP)) {
        spec0 = textureSample(_SpecularMap, _SpecularMap_sampler, uv_main0);
        spec1 = textureSample(_SpecularMap1, _SpecularMap1_sampler, uv_main1);
        if (kw_enabled(mat._MULTI_VALUES)) {
            spec0 = spec0 * mat._SpecularColor;
            spec1 = spec1 * mat._SpecularColor1;
        }
    }
    let spec = mix(spec0, spec1, l);
    let f0 = spec.rgb;
    let smoothness = clamp(spec.a, 0.0, 1.0);
    let roughness = clamp(1.0 - smoothness, 0.045, 1.0);
    let one_minus_reflectivity = 1.0 - max(max(f0.r, f0.g), f0.b);

    let n = sample_normal_world(uv_main0, uv_main1, world_n, front_facing, l);

    let cam = rg::frame.camera_world_pos.xyz;
    let v = normalize(cam - world_pos);

#ifdef MULTIVIEW
    let vi = view_idx;
#else
    let vi = 0u;
#endif
    let zc = pcls::select_eye_view_space_z_coeffs(
        vi,
        rg::frame.view_space_z_coeffs,
        rg::frame.view_space_z_coeffs_right,
        rg::frame.stereo_cluster_layers,
    );
    let cluster_id = pcls::cluster_id_from_frag(
        frag_pos.xy,
        world_pos,
        zc,
        rg::frame.viewport_width,
        rg::frame.viewport_height,
        rg::frame.cluster_count_x,
        rg::frame.cluster_count_y,
        rg::frame.cluster_count_z,
        rg::frame.near_clip,
        rg::frame.far_clip,
        vi,
        rg::frame.stereo_cluster_layers,
    );

    let count = rg::cluster_light_counts[cluster_id];
    let base_idx = cluster_id * pcls::MAX_LIGHTS_PER_TILE;
    var lo = vec3<f32>(0.0);
    let i_max = min(count, pcls::MAX_LIGHTS_PER_TILE);
    for (var i = 0u; i < i_max; i++) {
        let li = rg::cluster_light_indices[base_idx + i];
        if (li >= rg::frame.light_count) {
            continue;
        }
        let light = rg::lights[li];
        lo = lo + brdf::direct_radiance_specular(
            light,
            world_pos,
            n,
            v,
            roughness,
            base_color,
            f0,
            one_minus_reflectivity,
        );
    }

    let amb = vec3<f32>(0.03);
    let color = (amb * base_color * occlusion + lo * occlusion) + emission;
    let fg_anchor = (dot(rg::frame.view_space_z_coeffs_right, vec4<f32>(1.0, 1.0, 1.0, 1.0)) * 1e-10 + f32(rg::frame.stereo_cluster_layers) * 1e-10);
    return vec4<f32>(color + vec3<f32>(fg_anchor * 1e-30), alpha);
}
