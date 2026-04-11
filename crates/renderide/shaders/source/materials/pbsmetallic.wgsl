//! Unity Standard metallic PBS (`Shader "PBSMetallic"`): clustered forward + Cook–Torrance BRDF.
//!
//! Build emits `pbsmetallic_default` / `pbsmetallic_multiview`. `@group(1)` names match Unity material
//! properties. ForwardAdd / lightmaps / reflection probes are not implemented yet.
//!
//! Per-draw uniforms (`@group(2)`) use [`renderide::per_draw`].
//!
//! ## UV convention
//! `_MainTex_ST` applies to the primary texture set (albedo, metallic/gloss, bump, occlusion, emission,
//! detail mask). `_DetailAlbedoMap_ST` applies to the secondary/detail set (detail albedo, detail normal).
//! `_UVSec` selects UV0 or UV1 for the detail set (`0` = UV0, `1` = UV1; mesh currently only binds UV0
//! so both map to the same stream until TEXCOORD1 is plumbed).

// unity-shader-name: PBSMetallic

#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::pbs::brdf as brdf
#import renderide::pbs::cluster as pcls
#import renderide::alpha_clip_sample as acs
#import renderide::unity_st as ust
#import renderide::view_proj as vp

struct PbsMetallicMaterial {
    _Color: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _DetailAlbedoMap_ST: vec4<f32>,
    _Cutoff: f32,
    _Glossiness: f32,
    _GlossMapScale: f32,
    _SmoothnessTextureChannel: f32,
    _Metallic: f32,
    _BumpScale: f32,
    _Parallax: f32,
    _OcclusionStrength: f32,
    _DetailNormalMapScale: f32,
    _UVSec: f32,
    _SpecularHighlights: f32,
    _GlossyReflections: f32,
    _SrcBlend: f32,
    _DstBlend: f32,
    _ZWrite: f32,
    _Mode: f32,
}

@group(1) @binding(0)  var<uniform> mat: PbsMetallicMaterial;
@group(1) @binding(1)  var _MainTex: texture_2d<f32>;
@group(1) @binding(2)  var _MainTex_sampler: sampler;
@group(1) @binding(3)  var _MetallicGlossMap: texture_2d<f32>;
@group(1) @binding(4)  var _MetallicGlossMap_sampler: sampler;
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
    // Secondary UV for `_UVSec` / detail; copied from uv0 until TEXCOORD1 is bound in the mesh pass.
    @location(3) uv1: vec2<f32>,
}

/// Sample base + detail normal maps, blend them (UDN) and transform to world space.
/// `detail_mask` gates the detail contribution (Unity `_DetailMask` alpha channel).
fn sample_normal_world(
    uv_main: vec2<f32>,
    uv_det: vec2<f32>,
    world_n: vec3<f32>,
    detail_mask: f32,
) -> vec3<f32> {
    let tbn = brdf::orthonormal_tbn(world_n);
    var ts_n = brdf::decode_ts_normal(textureSample(_BumpMap, _BumpMap_sampler, uv_main).xyz, mat._BumpScale);

    // Detail normal: UDN blend — add detail XY scaled by mask, preserve base Z.
    if detail_mask > 0.001 {
        let detail_raw = textureSample(_DetailNormalMap, _DetailNormalMap_sampler, uv_det).xyz;
        let ts_detail = brdf::decode_ts_normal(detail_raw, mat._DetailNormalMapScale);
        ts_n = normalize(vec3<f32>(ts_n.xy + ts_detail.xy * detail_mask, ts_n.z));
    }

    return normalize(tbn * ts_n);
}

/// Return (metallic, roughness) from `_MetallicGlossMap` and material scalars.
fn metallic_roughness(uv: vec2<f32>) -> vec2<f32> {
    let mg = textureSample(_MetallicGlossMap, _MetallicGlossMap_sampler, uv);
    var metallic   = mat._Metallic * mg.x;
    // `_SmoothnessTextureChannel`: 0 = metallic map alpha, 1 = albedo alpha (handled in caller).
    let smooth_src = select(mg.w, mg.y, mat._SmoothnessTextureChannel < 0.5);
    let smoothness = mat._Glossiness * mat._GlossMapScale * smooth_src;
    let roughness  = clamp(1.0 - smoothness, 0.045, 1.0);
    metallic = clamp(metallic, 0.0, 1.0);
    return vec2<f32>(metallic, roughness);
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
    out.clip_pos  = vpm * world_p;
    out.world_pos = world_p.xyz;
    out.world_n   = wn;
    out.uv0 = uv0;
    // Mesh forward only binds TEXCOORD0 (`@location(2)`). Duplicate until a second UV stream is plumbed.
    out.uv1 = uv0;
    return out;
}

@fragment
fn fs_main(
    @builtin(position) frag_pos: vec4<f32>,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) uv1: vec2<f32>,
) -> @location(0) vec4<f32> {
    // --- UV transforms (Unity ST convention + WebGPU V-flip) ---
    let uv_main = ust::apply_st(uv0, mat._MainTex_ST);
    let uv_sec  = select(uv0, uv1, mat._UVSec > 0.5);
    let uv_det  = ust::apply_st(uv_sec, mat._DetailAlbedoMap_ST);

    // --- Albedo ---
    let albedo_s   = textureSample(_MainTex, _MainTex_sampler, uv_main);
    var base_color = mat._Color.xyz * albedo_s.xyz;
    let alpha      = mat._Color.a * albedo_s.a;
    let clip_alpha = mat._Color.a * acs::texture_alpha_base_mip(_MainTex, _MainTex_sampler, uv_main);
    if clip_alpha < mat._Cutoff {
        discard;
    }

    // --- Metallic / roughness ---
    let mr        = metallic_roughness(uv_main);
    let metallic  = mr.x;
    let roughness = mr.y;

    // --- Occlusion ---
    let occ_s   = textureSample(_OcclusionMap, _OcclusionMap_sampler, uv_main).x;
    let occlusion = mix(1.0, occ_s, mat._OcclusionStrength);

    // --- Detail mask (Unity alpha channel, defaults to 1 = full detail) ---
    let detail_mask_s = textureSample(_DetailMask, _DetailMask_sampler, uv_main).a;

    // --- Normal (base bump + detail normal blended by mask) ---
    var n = normalize(world_n);
    n = sample_normal_world(uv_main, uv_det, n, detail_mask_s);

    // --- Emission ---
    let em = textureSample(_EmissionMap, _EmissionMap_sampler, uv_main).xyz * mat._EmissionColor.xyz;

    // --- Detail albedo (Unity: LerpWhiteTo(detail × 2, mask) = mix(1, detail×2, mask)) ---
    let detail       = textureSample(_DetailAlbedoMap, _DetailAlbedoMap_sampler, uv_det).xyz;
    let detail_blend = mix(vec3<f32>(1.0), detail * 2.0, detail_mask_s);
    base_color = base_color * detail_blend;

    // --- Lighting ---
    let cam = rg::frame.camera_world_pos.xyz;
    let v   = normalize(cam - world_pos);
    let f0  = mix(vec3<f32>(0.04), base_color, metallic);

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

    let count    = rg::cluster_light_counts[cluster_id];
    let base_idx = cluster_id * pcls::MAX_LIGHTS_PER_TILE;
    var lo       = vec3<f32>(0.0);
    let spec_on  = mat._SpecularHighlights > 0.5;
    let i_max    = min(count, pcls::MAX_LIGHTS_PER_TILE);
    for (var i = 0u; i < i_max; i++) {
        let li = rg::cluster_light_indices[base_idx + i];
        if li >= rg::frame.light_count {
            continue;
        }
        let light = rg::lights[li];
        if spec_on {
            lo = lo + brdf::direct_radiance_metallic(light, world_pos, n, v, roughness, metallic, base_color, f0);
        } else {
            lo = lo + brdf::diffuse_only_metallic(light, world_pos, n, base_color);
        }
    }

    let amb   = select(vec3<f32>(0.03), vec3<f32>(0.0), mat._GlossyReflections < 0.5);
    let color = (amb * base_color * occlusion + lo * occlusion) + em;
    let fg_anchor = (dot(rg::frame.view_space_z_coeffs_right, vec4<f32>(1.0, 1.0, 1.0, 1.0)) * 1e-10 + f32(rg::frame.stereo_cluster_layers) * 1e-10);
    return vec4<f32>(color + vec3<f32>(fg_anchor * 1e-30), alpha);
}
