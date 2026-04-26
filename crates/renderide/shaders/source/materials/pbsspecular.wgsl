//! Unity Standard **specular** PBS (`Shader "PBSSpecular"` / Standard SpecularSetup): clustered forward +
//! Cook–Torrance BRDF with tinted specular color and Unity-style diffuse energy (`oneMinusReflectivity`).
//!
//! Build emits `pbsspecular_default` / `pbsspecular_multiview`. `@group(1)` names match Unity material
//! properties. ForwardAdd / lightmaps / reflection probes are not implemented yet.
//!
//! Per-draw uniforms (`@group(2)`) use [`renderide::per_draw`].
//!
//! ## UV convention
//! `_MainTex_ST` applies to the primary texture set (albedo, spec/gloss, bump, occlusion, emission,
//! detail mask). `_DetailAlbedoMap_ST` applies to the detail set (detail albedo, detail normal).
//! `_UVSec` selects UV0 or UV1 for the detail set.

// unity-shader-name: PBSSpecular

#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::pbs::brdf as brdf
#import renderide::pbs::cluster as pcls
#import renderide::alpha_clip_sample as acs
#import renderide::uv_utils as uvu
#import renderide::normal_decode as nd

struct PbsSpecularMaterial {
    _Color: vec4<f32>,
    _SpecColor: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _DetailAlbedoMap_ST: vec4<f32>,
    _Cutoff: f32,
    _Glossiness: f32,
    _SmoothnessTextureChannel: f32,
    _BumpScale: f32,
    _Parallax: f32,
    _DetailNormalMapScale: f32,
    _ALPHATEST_ON: f32,
    _ALPHABLEND_ON: f32,
    _ALPHAPREMULTIPLY_ON: f32,
}

@group(1) @binding(0)  var<uniform> mat: PbsSpecularMaterial;
@group(1) @binding(1)  var _MainTex: texture_2d<f32>;
@group(1) @binding(2)  var _MainTex_sampler: sampler;
@group(1) @binding(3)  var _SpecGlossMap: texture_2d<f32>;
@group(1) @binding(4)  var _SpecGlossMap_sampler: sampler;
@group(1) @binding(5)  var _BumpMap: texture_2d<f32>;
@group(1) @binding(6)  var _BumpMap_sampler: sampler;
@group(1) @binding(7)  var _OcclusionMap: texture_2d<f32>;
@group(1) @binding(8)  var _OcclusionMap_sampler: sampler;
@group(1) @binding(9)  var _EmissionMap: texture_2d<f32>;
@group(1) @binding(10) var _EmissionMap_sampler: sampler;
@group(1) @binding(11) var _DetailAlbedoMap: texture_2d<f32>;
@group(1) @binding(12) var _DetailAlbedoMap_sampler: sampler;
@group(1) @binding(13) var _DetailNormalMap: texture_2d<f32>;
@group(1) @binding(14) var _DetailNormalMap_sampler: sampler;
@group(1) @binding(15) var _DetailMask: texture_2d<f32>;
@group(1) @binding(16) var _DetailMask_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) uv1: vec2<f32>,
    @location(4) @interpolate(flat) view_layer: u32,
}

fn kw(v: f32) -> bool {
    return v > 0.5;
}

fn alpha_test_enabled() -> bool {
    return kw(mat._ALPHATEST_ON);
}

fn alpha_premultiply_enabled() -> bool {
    return kw(mat._ALPHAPREMULTIPLY_ON);
}

fn apply_premultiply(color: vec3<f32>, alpha: f32) -> vec3<f32> {
    return select(color, color * alpha, alpha_premultiply_enabled());
}

/// Sample base + detail normal maps, blend them (UDN) and transform to world space.
fn sample_normal_world(
    uv_main: vec2<f32>,
    uv_det: vec2<f32>,
    world_n: vec3<f32>,
    detail_mask: f32,
) -> vec3<f32> {
    let tbn = brdf::orthonormal_tbn(world_n);
    var ts_n = nd::decode_ts_normal_with_placeholder_sample(
        textureSample(_BumpMap, _BumpMap_sampler, uv_main),
        mat._BumpScale,
    );

    if detail_mask > 0.001 {
        let ts_detail = nd::decode_ts_normal_with_placeholder_sample(
            textureSample(_DetailNormalMap, _DetailNormalMap_sampler, uv_det),
            mat._DetailNormalMapScale,
        );
        ts_n = normalize(vec3<f32>(ts_n.xy + ts_detail.xy * detail_mask, ts_n.z));
    }

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
    out.clip_pos  = vp * world_p;
    out.world_pos = world_p.xyz;
    out.world_n   = wn;
    out.uv0 = uv0;
    out.uv1 = uv0;
#ifdef MULTIVIEW
    out.view_layer = view_idx;
#else
    out.view_layer = 0u;
#endif
    return out;
}

//#pass forward
@fragment
fn fs_main(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) uv1: vec2<f32>,
    @location(4) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    // --- UV transforms ---
    let uv_main = uvu::apply_st(uv0, mat._MainTex_ST);
    let uv_det  = uvu::apply_st(uv0, mat._DetailAlbedoMap_ST);

    // --- Albedo ---
    let albedo_s   = textureSample(_MainTex, _MainTex_sampler, uv_main);
    var base_color = mat._Color.xyz * albedo_s.xyz;
    let alpha      = mat._Color.a * albedo_s.a;
    let clip_alpha = mat._Color.a * acs::texture_alpha_base_mip(_MainTex, _MainTex_sampler, uv_main);
    if (alpha_test_enabled() && clip_alpha <= mat._Cutoff) {
        discard;
    }

    // --- Specular / gloss (Unity SpecularSetup) ---
    let sg         = textureSample(_SpecGlossMap, _SpecGlossMap_sampler, uv_main);
    // _SpecColor tints the specular map RGB; this is f0 (reflectance at normal incidence).
    var spec_tint  = mat._SpecColor.rgb * sg.rgb;
    // Smoothness: driven by spec gloss map (W = smoothness) or albedo alpha per _SmoothnessTextureChannel.
    let smooth_src = select(sg.w, albedo_s.a, mat._SmoothnessTextureChannel < 0.5);
    let smoothness = mat._Glossiness * smooth_src;
    let roughness  = clamp(1.0 - smoothness, 0.045, 1.0);
    // oneMinusReflectivity: energy conserved by diffuse (albedo reduced by specular contribution).
    let one_minus_reflectivity = 1.0 - max(max(spec_tint.r, spec_tint.g), spec_tint.b);
    let f0 = spec_tint;

    // --- Occlusion ---
    let occlusion = textureSample(_OcclusionMap, _OcclusionMap_sampler, uv_main).x;

    // --- Detail mask (alpha channel) ---
    let detail_mask_s = textureSample(_DetailMask, _DetailMask_sampler, uv_main).a;

    // --- Normal ---
    var n = normalize(world_n);
    n = sample_normal_world(uv_main, uv_det, n, detail_mask_s);

    // --- Emission ---
    let em = textureSample(_EmissionMap, _EmissionMap_sampler, uv_main).xyz * mat._EmissionColor.xyz;

    // --- Detail albedo (Unity: LerpWhiteTo(detail × 2, mask)) ---
    let detail       = textureSample(_DetailAlbedoMap, _DetailAlbedoMap_sampler, uv_det).xyz;
    let detail_blend = mix(vec3<f32>(1.0), detail * 2.0, detail_mask_s);
    base_color = base_color * detail_blend;

    // --- Lighting ---
    let cam = rg::frame.camera_world_pos.xyz;
    let v   = normalize(cam - world_pos);

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

    let count    = rg::cluster_light_counts[cluster_id];
    var lo       = vec3<f32>(0.0);
    let i_max    = min(count, pcls::MAX_LIGHTS_PER_TILE);
    for (var i = 0u; i < i_max; i++) {
        let li = pcls::cluster_light_index_at(cluster_id, i);
        if li >= rg::frame.light_count {
            continue;
        }
        let light = rg::lights[li];
        lo = lo + brdf::direct_radiance_specular(light, world_pos, n, v, roughness, base_color, f0, one_minus_reflectivity);
    }

    let amb   = vec3<f32>(0.03);
    let color = (amb * base_color * occlusion + lo * occlusion) + em;
    return vec4<f32>(apply_premultiply(color, alpha), alpha);
}
