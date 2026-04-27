//! Unity surface shader `Shader "PBSMultiUV"`: metallic Standard lighting where each texture
//! independently selects which mesh UV channel to sample and carries its own `_ST` tile/offset.
//!
//! Unity supports four UV channels (`texcoord` … `texcoord3`) selected by `_AlbedoUV`,
//! `_NormalUV`, `_EmissionUV`, etc. This renderer plumbs through UV0 and UV1; per-texture
//! `_*UV` values `< 1.0` resolve to UV0 and `>= 1.0` resolve to UV1, so meshes that author
//! against UV0/UV1 work end-to-end. UV2 / UV3 fall back to UV1 — supporting them requires
//! plumbing additional vertex streams through the per-draw layout, which is tracked separately.
//!
//! Mirrors the keyword surface (`_DUAL_ALBEDO`, `_EMISSIONTEX`, `_DUAL_EMISSIONTEX`, `_NORMALMAP`,
//! `_METALLICMAP`, `_OCCLUSION`, `_ALPHACLIP`).

// unity-shader-name: PBSMultiUV

#import renderide::globals as rg
#import renderide::sh2_ambient as shamb
#import renderide::per_draw as pd
#import renderide::pbs::brdf as brdf
#import renderide::pbs::normal as pnorm
#import renderide::pbs::cluster as pcls
#import renderide::alpha_clip_sample as acs
#import renderide::uv_utils as uvu
#import renderide::normal_decode as nd

/// Material uniforms for `PBSMultiUV`. Every texture has both a UV-channel selector (`_*UV`)
/// and a tile/offset (`_*_ST`), matching the Unity property block.
struct PbsMultiUVMaterial {
    /// Tint color (`Color`).
    _Color: vec4<f32>,
    /// Emission color (`EmissionColor`).
    _EmissionColor: vec4<f32>,
    /// Secondary emission color when `_DUAL_EMISSIONTEX` is enabled.
    _SecondaryEmissionColor: vec4<f32>,
    /// Albedo tile/offset.
    _MainTex_ST: vec4<f32>,
    _MainTex_StorageVInverted: f32,
    /// Secondary albedo tile/offset (used when `_DUAL_ALBEDO` is enabled).
    _SecondaryAlbedo_ST: vec4<f32>,
    /// Normal map tile/offset.
    _NormalMap_ST: vec4<f32>,
    /// Emission map tile/offset.
    _EmissionMap_ST: vec4<f32>,
    /// Secondary emission map tile/offset.
    _SecondaryEmissionMap_ST: vec4<f32>,
    /// Metallic map tile/offset.
    _MetallicMap_ST: vec4<f32>,
    /// Occlusion map tile/offset.
    _OcclusionMap_ST: vec4<f32>,
    /// Tangent-space normal scale (`Normal Scale`).
    _NormalScale: f32,
    /// Smoothness fallback when `_METALLICMAP` is disabled.
    _Glossiness: f32,
    /// Metallic fallback when `_METALLICMAP` is disabled.
    _Metallic: f32,
    /// Alpha-clip threshold; applied only when `_ALPHACLIP` is enabled.
    _AlphaClip: f32,
    /// UV-channel selector for `_MainTex` (Unity index, `>=1` rounds to UV1).
    _AlbedoUV: f32,
    /// UV-channel selector for `_SecondaryAlbedo`.
    _SecondaryAlbedoUV: f32,
    /// UV-channel selector for `_EmissionMap`.
    _EmissionUV: f32,
    /// UV-channel selector for `_SecondaryEmissionMap`.
    _SecondaryEmissionUV: f32,
    /// UV-channel selector for `_NormalMap`.
    _NormalUV: f32,
    /// UV-channel selector for `_OcclusionMap`.
    _OcclusionUV: f32,
    /// UV-channel selector for `_MetallicMap`.
    _MetallicUV: f32,
    /// Keyword: enable secondary albedo multiply.
    _DUAL_ALBEDO: f32,
    /// Keyword: enable emission texture multiply.
    _EMISSIONTEX: f32,
    /// Keyword: enable secondary emission texture additive contribution.
    _DUAL_EMISSIONTEX: f32,
    /// Keyword: enable normal map sampling.
    _NORMALMAP: f32,
    /// Keyword: read metallic + smoothness from `_MetallicMap` (R=metallic, A=smoothness).
    _METALLICMAP: f32,
    /// Keyword: read occlusion from `_OcclusionMap.r`.
    _OCCLUSION: f32,
    /// Keyword: enable alpha clipping against `_AlphaClip`.
    _ALPHACLIP: f32,
}

@group(1) @binding(0)  var<uniform> mat: PbsMultiUVMaterial;
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
@group(1) @binding(11) var _MetallicMap: texture_2d<f32>;
@group(1) @binding(12) var _MetallicMap_sampler: sampler;
@group(1) @binding(13) var _OcclusionMap: texture_2d<f32>;
@group(1) @binding(14) var _OcclusionMap_sampler: sampler;

/// Interpolated vertex output forwarded to both forward-base and forward-add fragments.
struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) uv1: vec2<f32>,
    @location(4) @interpolate(flat) view_layer: u32,
}

/// Resolved per-fragment shading inputs for the metallic Cook–Torrance path.
struct SurfaceData {
    base_color: vec3<f32>,
    alpha: f32,
    metallic: f32,
    roughness: f32,
    occlusion: f32,
    normal: vec3<f32>,
    emission: vec3<f32>,
}

/// Pick UV0 vs UV1 by a `_*UV` index uniform: `< 1.0` → UV0, `>= 1.0` → UV1. UV2 / UV3 are
/// not yet wired into this renderer, so any value above 1.0 collapses to UV1.
fn pick_uv(uv0: vec2<f32>, uv1: vec2<f32>, idx: f32) -> vec2<f32> {
    return select(uv0, uv1, idx >= 1.0);
}

/// Sample the normal map (when enabled) using its own UV channel + `_ST`, and place into world space.
fn sample_normal_world(uv0: vec2<f32>, uv1: vec2<f32>, world_n: vec3<f32>) -> vec3<f32> {
    let tbn = pnorm::orthonormal_tbn(normalize(world_n));
    var ts_n = vec3<f32>(0.0, 0.0, 1.0);
    if (uvu::kw_enabled(mat._NORMALMAP)) {
        let uv_n = uvu::apply_st(pick_uv(uv0, uv1, mat._NormalUV), mat._NormalMap_ST);
        ts_n = nd::decode_ts_normal_with_placeholder(
            textureSample(_NormalMap, _NormalMap_sampler, uv_n).xyz,
            mat._NormalScale,
        );
    }
    return normalize(tbn * ts_n);
}

/// Resolve the [`SurfaceData`] for a fragment, mirroring Unity's `surf` for `PBSMultiUV`.
fn sample_surface(uv0: vec2<f32>, uv1: vec2<f32>, world_n: vec3<f32>) -> SurfaceData {
    let uv_albedo = uvu::apply_st_for_storage(pick_uv(uv0, uv1, mat._AlbedoUV), mat._MainTex_ST, mat._MainTex_StorageVInverted);

    var c = mat._Color * textureSample(_MainTex, _MainTex_sampler, uv_albedo);
    if (uvu::kw_enabled(mat._DUAL_ALBEDO)) {
        let uv_albedo2 =
            uvu::apply_st(pick_uv(uv0, uv1, mat._SecondaryAlbedoUV), mat._SecondaryAlbedo_ST);
        c = c * textureSample(_SecondaryAlbedo, _SecondaryAlbedo_sampler, uv_albedo2);
    }
    let clip_alpha = mat._Color.a * acs::texture_alpha_base_mip(_MainTex, _MainTex_sampler, uv_albedo);
    if (uvu::kw_enabled(mat._ALPHACLIP) && clip_alpha <= mat._AlphaClip) {
        discard;
    }

    var metallic = mat._Metallic;
    var smoothness = mat._Glossiness;
    if (uvu::kw_enabled(mat._METALLICMAP)) {
        let uv_metal = uvu::apply_st(pick_uv(uv0, uv1, mat._MetallicUV), mat._MetallicMap_ST);
        let m = textureSample(_MetallicMap, _MetallicMap_sampler, uv_metal);
        metallic = m.r;
        smoothness = m.a;
    }
    metallic = clamp(metallic, 0.0, 1.0);
    let roughness = clamp(1.0 - smoothness, 0.045, 1.0);

    var occlusion = 1.0;
    if (uvu::kw_enabled(mat._OCCLUSION)) {
        let uv_occ = uvu::apply_st(pick_uv(uv0, uv1, mat._OcclusionUV), mat._OcclusionMap_ST);
        occlusion = textureSample(_OcclusionMap, _OcclusionMap_sampler, uv_occ).r;
    }

    var emission = mat._EmissionColor.rgb;
    if (uvu::kw_enabled(mat._EMISSIONTEX) || uvu::kw_enabled(mat._DUAL_EMISSIONTEX)) {
        let uv_em = uvu::apply_st(pick_uv(uv0, uv1, mat._EmissionUV), mat._EmissionMap_ST);
        emission = emission * textureSample(_EmissionMap, _EmissionMap_sampler, uv_em).rgb;
    }
    if (uvu::kw_enabled(mat._DUAL_EMISSIONTEX)) {
        let uv_em2 =
            uvu::apply_st(pick_uv(uv0, uv1, mat._SecondaryEmissionUV), mat._SecondaryEmissionMap_ST);
        let secondary =
            textureSample(_SecondaryEmissionMap, _SecondaryEmissionMap_sampler, uv_em2).rgb;
        emission = emission + secondary * mat._SecondaryEmissionColor.rgb;
    }

    return SurfaceData(
        c.rgb,
        c.a,
        metallic,
        roughness,
        occlusion,
        sample_normal_world(uv0, uv1, world_n),
        emission,
    );
}

/// Iterate the cluster's lights and accumulate Cook–Torrance radiance, gated by directional/local.
fn clustered_direct_lighting(
    frag_xy: vec2<f32>,
    world_pos: vec3<f32>,
    view_layer: u32,
    s: SurfaceData,
    include_directional: bool,
    include_local: bool,
) -> vec3<f32> {
    let cam = rg::frame.camera_world_pos.xyz;
    let v = normalize(cam - world_pos);
    let f0 = mix(vec3<f32>(0.04), s.base_color, s.metallic);

    let aa_roughness = brdf::filter_perceptual_roughness(s.roughness, s.normal);

    let cluster_id = pcls::cluster_id_from_frag(
        frag_xy,
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
    let i_max = min(count, pcls::MAX_LIGHTS_PER_TILE);
    var lo = vec3<f32>(0.0);
    for (var i = 0u; i < i_max; i++) {
        let li = pcls::cluster_light_index_at(cluster_id, i);
        if (li >= rg::frame.light_count) {
            continue;
        }
        let light = rg::lights[li];
        let is_directional = light.light_type == 1u;
        if ((is_directional && !include_directional) || (!is_directional && !include_local)) {
            continue;
        }
        lo = lo + brdf::direct_radiance_metallic(
            light,
            world_pos,
            s.normal,
            v,
            aa_roughness,
            s.metallic,
            s.base_color,
            f0,
        );
    }
    return lo;
}

/// Vertex stage: forward world position, world-space normal, and both UV0 and UV1 streams.
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
#ifdef MULTIVIEW
    out.view_layer = view_idx;
#else
    out.view_layer = 0u;
#endif
    return out;
}

/// Forward-base pass: ambient + directional lighting + emission.
//#pass forward
@fragment
fn fs_forward_base(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) uv1: vec2<f32>,
    @location(4) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let s = sample_surface(uv0, uv1, world_n);
    let direct = clustered_direct_lighting(frag_pos.xy, world_pos, view_layer, s, true, true);
    let ambient = shamb::ambient_diffuse(s.normal, s.base_color, s.occlusion);
    return vec4<f32>(ambient + direct + s.emission, s.alpha);
}

/// Forward-add pass: additive accumulation of local (point/spot) lights.
