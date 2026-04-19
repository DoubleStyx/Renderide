//! Unity PBS multi-UV specular (`Shader "PBSMultiUVSpecular"`).
//!
//! Supports dual albedo/emission maps plus per-map UV selectors (`0 -> uv0`, `>=1 -> uv1`).
// unity-shader-name: PBSMultiUVSpecular

#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::pbs::brdf as brdf
#import renderide::pbs::cluster as pcls
#import renderide::alpha_clip_sample as acs
#import renderide::uv_utils as uvu
#import renderide::normal_decode as nd

struct PbsMultiUvSpecularMaterial {
    _Color: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _SecondaryEmissionColor: vec4<f32>,
    _SpecularColor: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _SecondaryAlbedo_ST: vec4<f32>,
    _EmissionMap_ST: vec4<f32>,
    _SecondaryEmissionMap_ST: vec4<f32>,
    _NormalMap_ST: vec4<f32>,
    _OcclusionMap_ST: vec4<f32>,
    _SpecularMap_ST: vec4<f32>,
    _AlphaClip: f32,
    _NormalScale: f32,
    _AlbedoUV: f32,
    _SecondaryAlbedoUV: f32,
    _EmissionUV: f32,
    _SecondaryEmissionUV: f32,
    _NormalUV: f32,
    _OcclusionUV: f32,
    _SpecularUV: f32,
    _DUAL_ALBEDO: f32,
    _EMISSIONTEX: f32,
    _DUAL_EMISSIONTEX: f32,
    _NORMALMAP: f32,
    _SPECULARMAP: f32,
    _OCCLUSION: f32,
    _ALPHACLIP: f32,
}

@group(1) @binding(0)  var<uniform> mat: PbsMultiUvSpecularMaterial;
@group(1) @binding(1)  var _MainTex: texture_2d<f32>;
@group(1) @binding(2)  var _MainTex_sampler: sampler;
@group(1) @binding(3)  var _SecondaryAlbedo: texture_2d<f32>;
@group(1) @binding(4)  var _SecondaryAlbedo_sampler: sampler;
@group(1) @binding(5)  var _NormalMap: texture_2d<f32>;
@group(1) @binding(6)  var _NormalMap_sampler: sampler;
@group(1) @binding(7)  var _EmissionMap: texture_2d<f32>;
@group(1) @binding(8)  var _EmissionMap_sampler: sampler;
@group(1) @binding(9)  var _SecondaryEmissionMap: texture_2d<f32>;
@group(1) @binding(10) var _SecondaryEmissionMap_sampler: sampler;
@group(1) @binding(11) var _OcclusionMap: texture_2d<f32>;
@group(1) @binding(12) var _OcclusionMap_sampler: sampler;
@group(1) @binding(13) var _SpecularMap: texture_2d<f32>;
@group(1) @binding(14) var _SpecularMap_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) uv1: vec2<f32>,
    @location(4) uv2: vec2<f32>,
    @location(5) uv3: vec2<f32>,
    @location(6) @interpolate(flat) view_layer: u32,
}

fn pick_uv(selector: f32, uv0: vec2<f32>, uv1: vec2<f32>, uv2: vec2<f32>, uv3: vec2<f32>) -> vec2<f32> {
    let idx = i32(round(selector));
    if (idx <= 0) {
        return uv0;
    }
    if (idx == 1) {
        return uv1;
    }
    if (idx == 2) {
        return uv2;
    }
    return uv3;
}

fn sample_normal_world(uv: vec2<f32>, world_n: vec3<f32>) -> vec3<f32> {
    if (!uvu::kw_enabled(mat._NORMALMAP)) {
        return normalize(world_n);
    }
    let tbn = brdf::orthonormal_tbn(normalize(world_n));
    let ts_n = nd::decode_ts_normal_with_placeholder_sample(
        textureSample(_NormalMap, _NormalMap_sampler, uv),
        mat._NormalScale,
    );
    return normalize(tbn * ts_n);
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
    @location(5) uv1: vec2<f32>,
    @location(6) uv2: vec2<f32>,
    @location(7) uv3: vec2<f32>,
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
    let world_p = d.model * vec4<f32>(pos.xyz, 1.0);
    let wn = normalize(d.normal_matrix * n.xyz);
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
    out.uv0 = uv0;
    out.uv1 = uv1;
    out.uv2 = uv2;
    out.uv3 = uv3;
#ifdef MULTIVIEW
    out.view_layer = view_idx;
#else
    out.view_layer = 0u;
#endif
    return out;
}

@fragment
fn fs_main(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) uv1: vec2<f32>,
    @location(4) uv2: vec2<f32>,
    @location(5) uv3: vec2<f32>,
    @location(6) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let uv_albedo = uvu::apply_st(pick_uv(mat._AlbedoUV, uv0, uv1, uv2, uv3), mat._MainTex_ST);
    let uv_albedo2 = uvu::apply_st(
        pick_uv(mat._SecondaryAlbedoUV, uv0, uv1, uv2, uv3),
        mat._SecondaryAlbedo_ST,
    );
    let uv_emission = uvu::apply_st(
        pick_uv(mat._EmissionUV, uv0, uv1, uv2, uv3),
        mat._EmissionMap_ST,
    );
    let uv_emission2 = uvu::apply_st(
        pick_uv(mat._SecondaryEmissionUV, uv0, uv1, uv2, uv3),
        mat._SecondaryEmissionMap_ST,
    );
    let uv_normal = uvu::apply_st(pick_uv(mat._NormalUV, uv0, uv1, uv2, uv3), mat._NormalMap_ST);
    let uv_occ = uvu::apply_st(pick_uv(mat._OcclusionUV, uv0, uv1, uv2, uv3), mat._OcclusionMap_ST);
    let uv_spec = uvu::apply_st(pick_uv(mat._SpecularUV, uv0, uv1, uv2, uv3), mat._SpecularMap_ST);

    var albedo_s = textureSample(_MainTex, _MainTex_sampler, uv_albedo);
    if (uvu::kw_enabled(mat._DUAL_ALBEDO)) {
        albedo_s = albedo_s * textureSample(_SecondaryAlbedo, _SecondaryAlbedo_sampler, uv_albedo2);
    }

    let base_color = mat._Color.rgb * albedo_s.rgb;
    let alpha = mat._Color.a * albedo_s.a;
    if (uvu::kw_enabled(mat._ALPHACLIP)) {
        let clip_alpha = mat._Color.a * acs::texture_alpha_base_mip(_MainTex, _MainTex_sampler, uv_albedo);
        if (clip_alpha <= mat._AlphaClip) {
            discard;
        }
    }

    let spec_sample = select(
        mat._SpecularColor,
        textureSample(_SpecularMap, _SpecularMap_sampler, uv_spec),
        uvu::kw_enabled(mat._SPECULARMAP),
    );
    let f0 = clamp(spec_sample.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let smoothness = clamp(spec_sample.a, 0.0, 1.0);
    let roughness = clamp(1.0 - smoothness, 0.045, 1.0);
    let one_minus_reflectivity = 1.0 - max(max(f0.r, f0.g), f0.b);

    let occlusion = select(
        1.0,
        textureSample(_OcclusionMap, _OcclusionMap_sampler, uv_occ).r,
        uvu::kw_enabled(mat._OCCLUSION),
    );

    let n = sample_normal_world(uv_normal, world_n);

    var emission = mat._EmissionColor.rgb;
    if (uvu::kw_enabled(mat._EMISSIONTEX) || uvu::kw_enabled(mat._DUAL_EMISSIONTEX)) {
        emission = emission * textureSample(_EmissionMap, _EmissionMap_sampler, uv_emission).rgb;
    }
    if (uvu::kw_enabled(mat._DUAL_EMISSIONTEX)) {
        emission = emission + textureSample(_SecondaryEmissionMap, _SecondaryEmissionMap_sampler, uv_emission2).rgb
            * mat._SecondaryEmissionColor.rgb;
    }

    let cam = rg::frame.camera_world_pos.xyz;
    let v = normalize(cam - world_pos);

    let cluster_id = pcls::cluster_id_from_frag(
        frag_pos.xy,
        world_pos,
        rg::frame.view_space_z_coeffs,
        rg::frame.view_space_z_coeffs_right,
        view_layer,
        rg::frame.viewport_width,
        rg::frame.viewport_height,
        rg::frame.cluster_count_x,
        rg::frame.cluster_count_y,
        rg::frame.cluster_count_z,
        rg::frame.near_clip,
        rg::frame.far_clip,
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
    return vec4<f32>(color, alpha);
}
