//! Unity surface shader `Shader "PBSMultiUVSpecular"`: Standard SpecularSetup where each texture
//! independently selects its mesh UV channel and tile/offset.
//!
//! Mirrors [`pbsmultiuv`](super::pbsmultiuv) for the SpecularSetup workflow. All four Unity UV
//! channels (`texcoord` ... `texcoord3`) are wired through. Per-texture `_*UV` values `< 1.0`
//! resolve to UV0, `< 2.0` to UV1, `< 3.0` to UV2, and `>= 3.0` to UV3.


#import renderide::mesh::vertex as mv
#import renderide::pbs::normal as pnorm
#import renderide::pbs::lighting as plight
#import renderide::pbs::surface as psurf
#import renderide::alpha_clip_sample as acs
#import renderide::uv_utils as uvu
#import renderide::normal_decode as nd

/// Material uniforms for `PBSMultiUVSpecular`.
struct PbsMultiUVSpecularMaterial {
    /// Tint color (`Color`).
    _Color: vec4<f32>,
    /// Emission color (`EmissionColor`).
    _EmissionColor: vec4<f32>,
    /// Secondary emission color when `_DUAL_EMISSIONTEX` is enabled.
    _SecondaryEmissionColor: vec4<f32>,
    /// Tinted specular color when `_SPECULARMAP` is disabled (RGB = f0, A = smoothness).
    _SpecularColor: vec4<f32>,
    /// Albedo tile/offset.
    _MainTex_ST: vec4<f32>,
    /// Storage-V-inverted flag for `_MainTex` (non-zero => skip the final V-flip).
    /// Secondary albedo tile/offset.
    _SecondaryAlbedo_ST: vec4<f32>,
    /// Storage-V-inverted flag for `_SecondaryAlbedo`.
    /// Normal map tile/offset.
    _NormalMap_ST: vec4<f32>,
    /// Storage-V-inverted flag for `_NormalMap`.
    /// Emission map tile/offset.
    _EmissionMap_ST: vec4<f32>,
    /// Storage-V-inverted flag for `_EmissionMap`.
    /// Secondary emission map tile/offset.
    _SecondaryEmissionMap_ST: vec4<f32>,
    /// Storage-V-inverted flag for `_SecondaryEmissionMap`.
    /// Specular map tile/offset.
    _SpecularMap_ST: vec4<f32>,
    /// Storage-V-inverted flag for `_SpecularMap`.
    /// Occlusion map tile/offset.
    _OcclusionMap_ST: vec4<f32>,
    /// Storage-V-inverted flag for `_OcclusionMap`.
    /// Tangent-space normal scale.
    _NormalScale: f32,
    /// Alpha-clip threshold; applied only when `_ALPHACLIP` is enabled.
    _AlphaClip: f32,
    /// UV-channel selector for `_MainTex` (Unity index 0..3 => UV0..UV3).
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
    /// UV-channel selector for `_SpecularMap`.
    _SpecularUV: f32,
    /// Keyword: enable secondary albedo multiply.
    _DUAL_ALBEDO: f32,
    /// Keyword: enable emission texture multiply.
    _EMISSIONTEX: f32,
    /// Keyword: enable secondary emission texture additive contribution.
    _DUAL_EMISSIONTEX: f32,
    /// Keyword: enable normal map sampling.
    _NORMALMAP: f32,
    /// Keyword: read tinted f0 + smoothness from `_SpecularMap`.
    _SPECULARMAP: f32,
    /// Keyword: read occlusion from `_OcclusionMap.r`.
    _OCCLUSION: f32,
    /// Keyword: enable alpha clipping against `_AlphaClip`.
    _ALPHACLIP: f32,
}

@group(1) @binding(0)  var<uniform> mat: PbsMultiUVSpecularMaterial;
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
@group(1) @binding(11) var _SpecularMap: texture_2d<f32>;
@group(1) @binding(12) var _SpecularMap_sampler: sampler;
@group(1) @binding(13) var _OcclusionMap: texture_2d<f32>;
@group(1) @binding(14) var _OcclusionMap_sampler: sampler;

/// Resolved per-fragment shading inputs for the SpecularSetup path.
struct SurfaceData {
    base_color: vec3<f32>,
    alpha: f32,
    f0: vec3<f32>,
    roughness: f32,
    one_minus_reflectivity: f32,
    occlusion: f32,
    normal: vec3<f32>,
    emission: vec3<f32>,
}

/// Pick UV0..UV3 by a `_*UV` index uniform: `< 1.0` -> UV0, `< 2.0` -> UV1, `< 3.0` -> UV2,
/// `>= 3.0` -> UV3.
fn pick_uv(uv0: vec2<f32>, uv1: vec2<f32>, uv2: vec2<f32>, uv3: vec2<f32>, idx: f32) -> vec2<f32> {
    let lo = select(uv0, uv1, idx >= 1.0);
    let hi = select(uv2, uv3, idx >= 3.0);
    return select(lo, hi, idx >= 2.0);
}

/// Sample the normal map (when enabled) using its own UV channel + `_ST`, and place into world space.
fn sample_normal_world(
    uv0: vec2<f32>,
    uv1: vec2<f32>,
    uv2: vec2<f32>,
    uv3: vec2<f32>,
    world_n: vec3<f32>,
    world_t: vec4<f32>,
    front_facing: bool,
) -> vec3<f32> {
    let tbn = pnorm::orthonormal_tbn(world_n, world_t);
    var ts_n = vec3<f32>(0.0, 0.0, 1.0);
    if (uvu::kw_enabled(mat._NORMALMAP)) {
        let uv_n = uvu::apply_st(pick_uv(uv0, uv1, uv2, uv3, mat._NormalUV), mat._NormalMap_ST);
        ts_n = nd::decode_ts_normal_with_placeholder_sample(
            textureSample(_NormalMap, _NormalMap_sampler, uv_n),
            mat._NormalScale,
        );
    }
    if (!front_facing) {
        ts_n.z = -ts_n.z;
    }
    return normalize(tbn * ts_n);
}

/// Resolve the [`SurfaceData`] for a fragment, mirroring Unity's `surf` for `PBSMultiUVSpecular`.
fn sample_surface(
    uv0: vec2<f32>,
    uv1: vec2<f32>,
    uv2: vec2<f32>,
    uv3: vec2<f32>,
    world_n: vec3<f32>,
    world_t: vec4<f32>,
    front_facing: bool,
) -> SurfaceData {
    let uv_albedo = uvu::apply_st(pick_uv(uv0, uv1, uv2, uv3, mat._AlbedoUV), mat._MainTex_ST);

    var c = mat._Color * textureSample(_MainTex, _MainTex_sampler, uv_albedo);
    if (uvu::kw_enabled(mat._DUAL_ALBEDO)) {
        let uv_albedo2 = uvu::apply_st(pick_uv(uv0, uv1, uv2, uv3, mat._SecondaryAlbedoUV), mat._SecondaryAlbedo_ST);
        c = c * textureSample(_SecondaryAlbedo, _SecondaryAlbedo_sampler, uv_albedo2);
    }
    let clip_alpha = mat._Color.a * acs::texture_alpha_base_mip(_MainTex, _MainTex_sampler, uv_albedo);
    if (uvu::kw_enabled(mat._ALPHACLIP) && clip_alpha <= mat._AlphaClip) {
        discard;
    }

    var spec = mat._SpecularColor;
    if (uvu::kw_enabled(mat._SPECULARMAP)) {
        let uv_spec = uvu::apply_st(pick_uv(uv0, uv1, uv2, uv3, mat._SpecularUV), mat._SpecularMap_ST);
        spec = textureSample(_SpecularMap, _SpecularMap_sampler, uv_spec);
    }
    let f0 = clamp(spec.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let smoothness = clamp(spec.a, 0.0, 1.0);
    let roughness = clamp(1.0 - smoothness, 0.045, 1.0);
    let one_minus_reflectivity = 1.0 - max(max(f0.r, f0.g), f0.b);

    var occlusion = 1.0;
    if (uvu::kw_enabled(mat._OCCLUSION)) {
        let uv_occ = uvu::apply_st(pick_uv(uv0, uv1, uv2, uv3, mat._OcclusionUV), mat._OcclusionMap_ST);
        occlusion = textureSample(_OcclusionMap, _OcclusionMap_sampler, uv_occ).r;
    }

    var emission = mat._EmissionColor.rgb;
    if (uvu::kw_enabled(mat._EMISSIONTEX) || uvu::kw_enabled(mat._DUAL_EMISSIONTEX)) {
        let uv_em = uvu::apply_st(pick_uv(uv0, uv1, uv2, uv3, mat._EmissionUV), mat._EmissionMap_ST);
        emission = emission * textureSample(_EmissionMap, _EmissionMap_sampler, uv_em).rgb;
    }
    if (uvu::kw_enabled(mat._DUAL_EMISSIONTEX)) {
        let uv_em2 = uvu::apply_st(pick_uv(uv0, uv1, uv2, uv3, mat._SecondaryEmissionUV), mat._SecondaryEmissionMap_ST);
        let secondary =
            textureSample(_SecondaryEmissionMap, _SecondaryEmissionMap_sampler, uv_em2).rgb;
        emission = emission + secondary * mat._SecondaryEmissionColor.rgb;
    }

    return SurfaceData(
        c.rgb,
        c.a,
        f0,
        roughness,
        one_minus_reflectivity,
        occlusion,
        sample_normal_world(uv0, uv1, uv2, uv3, world_n, world_t, front_facing),
        emission,
    );
}

/// Vertex stage: forward world position, world-space normal, and all four UV streams.
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
    @location(5) uv1: vec2<f32>,
    @location(6) uv2: vec2<f32>,
    @location(7) uv3: vec2<f32>,
) -> mv::WorldUv4VertexOutput {
#ifdef MULTIVIEW
    return mv::world_uv4_vertex_main(instance_index, view_idx, pos, n, t, uv0, uv1, uv2, uv3);
#else
    return mv::world_uv4_vertex_main(instance_index, 0u, pos, n, t, uv0, uv1, uv2, uv3);
#endif
}

/// Forward-base pass: clustered lighting (ambient + directional + local lights) + emission.
//#pass forward
@fragment
fn fs_forward_base(
    @builtin(position) frag_pos: vec4<f32>,
    @builtin(front_facing) front_facing: bool,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) world_t: vec4<f32>,
    @location(3) uv0: vec2<f32>,
    @location(4) uv1: vec2<f32>,
    @location(5) uv2: vec2<f32>,
    @location(6) uv3: vec2<f32>,
    @location(7) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let s = sample_surface(uv0, uv1, uv2, uv3, world_n, world_t, front_facing);
    let surface = psurf::specular(
        s.base_color,
        s.alpha,
        s.f0,
        s.roughness,
        s.occlusion,
        s.normal,
        s.emission,
    );
    return vec4<f32>(
        plight::shade_specular_clustered(
            frag_pos.xy,
            world_pos,
            view_layer,
            surface,
            plight::default_lighting_options(),
        ),
        s.alpha,
    );
}
