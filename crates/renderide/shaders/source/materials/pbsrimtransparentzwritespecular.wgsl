//! Unity PBS rim transparent specular with ZWrite (`Shader "PBSRimTransparentZWriteSpecular"`):
//! same shading as [`pbsrimtransparentspecular`](super::pbsrimtransparentspecular), with a
//! depth-only prepass before the alpha-blended forward pass so the surface populates depth.
//!
//! Mirrors [`pbsrimtransparentzwrite`](super::pbsrimtransparentzwrite) but with the SpecularSetup
//! workflow.

// unity-shader-name: PBSRimTransparentZWriteSpecular

#import renderide::globals as rg
#import renderide::sh2_ambient as shamb
#import renderide::per_draw as pd
#import renderide::pbs::brdf as brdf
#import renderide::pbs::normal as pnorm
#import renderide::pbs::cluster as pcls
#import renderide::uv_utils as uvu
#import renderide::normal_decode as nd

struct PbsRimTransparentZWriteSpecularMaterial {
    _Color: vec4<f32>,
    _SpecularColor: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _RimColor: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _MainTex_StorageVInverted: f32,
    _NormalScale: f32,
    _RimPower: f32,
    _ALBEDOTEX: f32,
    _EMISSIONTEX: f32,
    _NORMALMAP: f32,
    _SPECULARMAP: f32,
    _OCCLUSION: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(1) @binding(0)  var<uniform> mat: PbsRimTransparentZWriteSpecularMaterial;
@group(1) @binding(1)  var _MainTex: texture_2d<f32>;
@group(1) @binding(2)  var _MainTex_sampler: sampler;
@group(1) @binding(3)  var _NormalMap: texture_2d<f32>;
@group(1) @binding(4)  var _NormalMap_sampler: sampler;
@group(1) @binding(5)  var _EmissionMap: texture_2d<f32>;
@group(1) @binding(6)  var _EmissionMap_sampler: sampler;
@group(1) @binding(7)  var _OcclusionMap: texture_2d<f32>;
@group(1) @binding(8)  var _OcclusionMap_sampler: sampler;
@group(1) @binding(9)  var _SpecularMap: texture_2d<f32>;
@group(1) @binding(10) var _SpecularMap_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
}

fn sample_normal_world(uv_main: vec2<f32>, world_n: vec3<f32>) -> vec3<f32> {
    var n = normalize(world_n);
    if (uvu::kw_enabled(mat._NORMALMAP)) {
        let tbn = pnorm::orthonormal_tbn(n);
        let ts_n = nd::decode_ts_normal_with_placeholder(
            textureSample(_NormalMap, _NormalMap_sampler, uv_main).xyz,
            mat._NormalScale,
        );
        return normalize(tbn * ts_n);
    }
    return n;
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
#ifdef MULTIVIEW
    out.view_layer = view_idx;
#else
    out.view_layer = 0u;
#endif
    return out;
}

//#pass depth_prepass
@fragment
fn fs_depth_only(
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let uv_main = uvu::apply_st_for_storage(uv0, mat._MainTex_ST, mat._MainTex_StorageVInverted);
    let albedo_s = textureSample(_MainTex, _MainTex_sampler, uv_main);
    let normal_s = textureSample(_NormalMap, _NormalMap_sampler, uv_main);
    let emit_s = textureSample(_EmissionMap, _EmissionMap_sampler, uv_main);
    let occ_s = textureSample(_OcclusionMap, _OcclusionMap_sampler, uv_main);
    let spec_s = textureSample(_SpecularMap, _SpecularMap_sampler, uv_main);
    let touch = (mat._Color.x + mat._SpecularColor.x + mat._EmissionColor.x + mat._RimColor.x
        + mat._NormalScale + mat._RimPower + mat._ALBEDOTEX + mat._EMISSIONTEX + mat._NORMALMAP
        + mat._SPECULARMAP + mat._OCCLUSION
        + albedo_s.x + normal_s.x + emit_s.x + occ_s.x + spec_s.x
        + world_pos.x + world_n.x + f32(view_layer)) * 0.0;
    return rg::retain_globals_additive(vec4<f32>(touch, touch, touch, 0.0));
}

//#pass forward
@fragment
fn fs_main(
    @builtin(position) frag_pos: vec4<f32>,
    @builtin(front_facing) front_facing: bool,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let uv_main = uvu::apply_st_for_storage(uv0, mat._MainTex_ST, mat._MainTex_StorageVInverted);

    var c0 = mat._Color;
    if (uvu::kw_enabled(mat._ALBEDOTEX)) {
        c0 = c0 * textureSample(_MainTex, _MainTex_sampler, uv_main);
    }
    let base_color = c0.rgb;
    let alpha = c0.a;

    var n = sample_normal_world(uv_main, world_n);
    if (!front_facing) {
        n = -n;
    }

    var occlusion = 1.0;
    if (uvu::kw_enabled(mat._OCCLUSION)) {
        occlusion = textureSample(_OcclusionMap, _OcclusionMap_sampler, uv_main).r;
    }

    var spec = mat._SpecularColor;
    if (uvu::kw_enabled(mat._SPECULARMAP)) {
        spec = textureSample(_SpecularMap, _SpecularMap_sampler, uv_main);
    }
    let f0 = clamp(spec.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let smoothness = clamp(spec.a, 0.0, 1.0);
    let roughness = clamp(1.0 - smoothness, 0.045, 1.0);
    let one_minus_reflectivity = 1.0 - max(max(f0.r, f0.g), f0.b);

    var emission = mat._EmissionColor.rgb;
    if (uvu::kw_enabled(mat._EMISSIONTEX)) {
        emission = emission * textureSample(_EmissionMap, _EmissionMap_sampler, uv_main).rgb;
    }

    let cam = rg::frame.camera_world_pos.xyz;
    let v = normalize(cam - world_pos);

    let rim = pow(max(1.0 - clamp(dot(v, n), 0.0, 1.0), 0.0), max(mat._RimPower, 1e-4));
    let rim_emission = mat._RimColor.rgb * rim;

    let aa_roughness = brdf::filter_perceptual_roughness(roughness, n);

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
    var lo = vec3<f32>(0.0);
    let i_max = min(count, pcls::MAX_LIGHTS_PER_TILE);
    for (var i = 0u; i < i_max; i++) {
        let li = pcls::cluster_light_index_at(cluster_id, i);
        if (li >= rg::frame.light_count) {
            continue;
        }
        let light = rg::lights[li];
        lo = lo + brdf::direct_radiance_specular(
            light, world_pos, n, v, aa_roughness, base_color, f0, one_minus_reflectivity,
        );
    }

    let amb = shamb::ambient_probe(n);
    let color = (amb * base_color * occlusion + lo) + emission + rim_emission;
    return vec4<f32>(color, alpha);
}
