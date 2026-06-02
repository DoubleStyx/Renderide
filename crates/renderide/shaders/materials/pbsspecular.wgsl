//! Unity Standard **specular** PBS (`Shader "PBSSpecular"` / Standard SpecularSetup): clustered
//! forward Cook-Torrance BRDF with specular f0 color and Unity-style diffuse energy conservation.
//!
//! Build emits `pbsspecular_default` / `pbsspecular_multiview`. `@group(1)` names match Unity
//! material properties. ForwardAdd, lightmaps, and reflection probes are not implemented yet.
//!
//! Froox variant bits populate `_RenderideVariantBits`; PBSSpecular's eleven keywords (sorted
//! alphabetically) occupy bits 0-10. `_ALPHABLEND_ON` is pipeline-affecting (blend mode), so it
//! reserves bit 0 but gets no shader-local constant.


//#texture_default _MainTex white
//#texture_default _SpecGlossMap white
//#texture_default _BumpMap bump
//#texture_default _ParallaxMap black
//#texture_default _OcclusionMap white
//#texture_default _EmissionMap white
//#texture_default _DetailMask white
//#texture_default _DetailAlbedoMap gray
//#texture_default _DetailNormalMap bump
//#mat_default _GlossMapScale float 1.0
//#mat_default _SmoothnessTextureChannel float 0.0
//#mat_default _OcclusionStrength float 1.0
//#mat_default _UVSec float 0.0
//#mat_default _BumpScale float 1.0
//#mat_default _Color vec4 1.0 1.0 1.0 1.0
//#mat_default _DetailNormalMapScale float 1.0
//#mat_default _EmissionColor vec4 0.0 0.0 0.0 1.0
//#mat_default _SpecColor vec4 0.2 0.2 0.2 1.0
//#mat_default _Cutoff float 0.5
//#mat_default _Glossiness float 0.5

#import renderide::mesh::vertex as mv
#import renderide::material::variant_bits as vb
#import renderide::pbs::detail as pdet
#import renderide::pbs::lighting as plight
#import renderide::pbs::surface as psurf
#import renderide::pbs::standard as pstd
#import renderide::core::uv as uvu
#import renderide::core::texture_sampling as ts

struct PbsSpecularMaterial {
    _Color: vec4<f32>,
    _SpecColor: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _DetailAlbedoMap_ST: vec4<f32>,
    _Cutoff: f32,
    _Glossiness: f32,
    _GlossMapScale: f32,
    _SmoothnessTextureChannel: f32,
    _BumpScale: f32,
    _Parallax: f32,
    _OcclusionStrength: f32,
    _DetailNormalMapScale: f32,
    _UVSec: f32,
    _RenderideVariantBits: u32,
    _MainTex_LodBias: f32,
    _SpecGlossMap_LodBias: f32,
    _BumpMap_LodBias: f32,
    _ParallaxMap_LodBias: f32,
    _OcclusionMap_LodBias: f32,
    _EmissionMap_LodBias: f32,
    _DetailMask_LodBias: f32,
    _DetailAlbedoMap_LodBias: f32,
    _DetailNormalMap_LodBias: f32,
}

const PBSSPECULAR_KW_ALPHABLEND_ON: u32 = 1u << 0u;
const PBSSPECULAR_KW_ALPHAPREMULTIPLY_ON: u32 = 1u << 1u;
const PBSSPECULAR_KW_ALPHATEST_ON: u32 = 1u << 2u;
const PBSSPECULAR_KW_DETAIL_MULX2: u32 = 1u << 3u;
const PBSSPECULAR_KW_EMISSION: u32 = 1u << 4u;
const PBSSPECULAR_KW_GLOSSYREFLECTIONS_OFF: u32 = 1u << 5u;
const PBSSPECULAR_KW_NORMALMAP: u32 = 1u << 6u;
const PBSSPECULAR_KW_PARALLAXMAP: u32 = 1u << 7u;
const PBSSPECULAR_KW_SMOOTHNESS_TEXTURE_ALBEDO_CHANNEL_A: u32 = 1u << 8u;
const PBSSPECULAR_KW_SPECGLOSSMAP: u32 = 1u << 9u;
const PBSSPECULAR_KW_SPECULARHIGHLIGHTS_OFF: u32 = 1u << 10u;

@group(1) @binding(0)  var<uniform> mat: PbsSpecularMaterial;
@group(1) @binding(1)  var _MainTex: texture_2d<f32>;
@group(1) @binding(2)  var _MainTex_sampler: sampler;
@group(1) @binding(3)  var _SpecGlossMap: texture_2d<f32>;
@group(1) @binding(4)  var _SpecGlossMap_sampler: sampler;
@group(1) @binding(5)  var _BumpMap: texture_2d<f32>;
@group(1) @binding(6)  var _BumpMap_sampler: sampler;
@group(1) @binding(7)  var _ParallaxMap: texture_2d<f32>;
@group(1) @binding(8)  var _ParallaxMap_sampler: sampler;
@group(1) @binding(9)  var _OcclusionMap: texture_2d<f32>;
@group(1) @binding(10) var _OcclusionMap_sampler: sampler;
@group(1) @binding(11) var _EmissionMap: texture_2d<f32>;
@group(1) @binding(12) var _EmissionMap_sampler: sampler;
@group(1) @binding(13) var _DetailMask: texture_2d<f32>;
@group(1) @binding(14) var _DetailMask_sampler: sampler;
@group(1) @binding(15) var _DetailAlbedoMap: texture_2d<f32>;
@group(1) @binding(16) var _DetailAlbedoMap_sampler: sampler;
@group(1) @binding(17) var _DetailNormalMap: texture_2d<f32>;
@group(1) @binding(18) var _DetailNormalMap_sampler: sampler;

struct SurfaceData {
    base_color: vec3<f32>,
    alpha: f32,
    specular_color: vec3<f32>,
    roughness: f32,
    occlusion: f32,
    normal: vec3<f32>,
    emission: vec3<f32>,
}

fn pbs_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn alpha_test_enabled() -> bool {
    return pbs_kw(PBSSPECULAR_KW_ALPHATEST_ON);
}

fn alpha_premultiply_enabled() -> bool {
    return pbs_kw(PBSSPECULAR_KW_ALPHAPREMULTIPLY_ON);
}

fn spec_gloss_map_enabled() -> bool {
    return pbs_kw(PBSSPECULAR_KW_SPECGLOSSMAP);
}

fn specular_highlights_enabled() -> bool {
    return !pbs_kw(PBSSPECULAR_KW_SPECULARHIGHLIGHTS_OFF);
}

fn glossy_reflections_enabled() -> bool {
    return !pbs_kw(PBSSPECULAR_KW_GLOSSYREFLECTIONS_OFF);
}

fn smoothness_from_albedo_alpha() -> bool {
    return pbs_kw(PBSSPECULAR_KW_SMOOTHNESS_TEXTURE_ALBEDO_CHANNEL_A);
}

fn sample_surface(uv0: vec2<f32>, uv1: vec2<f32>, world_pos: vec3<f32>, world_n: vec3<f32>, world_t: vec4<f32>, view_layer: u32) -> SurfaceData {
    let uv_base = uvu::apply_st(uv0, mat._MainTex_ST);
    let uv_main = pstd::apply_parallax(
        uv_base,
        pbs_kw(PBSSPECULAR_KW_PARALLAXMAP),
        mat._Parallax,
        world_pos,
        world_n,
        world_t,
        view_layer,
        _ParallaxMap,
        _ParallaxMap_sampler,
        mat._ParallaxMap_LodBias,
    );
    let uv_detail = pdet::detail_uv(uv0, uv1, mat._UVSec, mat._DetailAlbedoMap_ST);

    let albedo_sample = ts::sample_tex_2d(_MainTex, _MainTex_sampler, uv_main, mat._MainTex_LodBias);
    let base_alpha = pstd::standard_alpha(mat._Color.a, albedo_sample.a, smoothness_from_albedo_alpha());
    pstd::clip_standard_alpha(base_alpha, mat._Cutoff, alpha_test_enabled());

    var base_color = mat._Color.rgb * albedo_sample.rgb;

    var specular_color = mat._SpecColor.rgb;
    var smoothness = mat._Glossiness;
    let smoothness_scale = mat._GlossMapScale;
    if (spec_gloss_map_enabled()) {
        let spec_gloss = ts::sample_tex_2d(_SpecGlossMap, _SpecGlossMap_sampler, uv_main, mat._SpecGlossMap_LodBias);
        specular_color = spec_gloss.rgb;
        smoothness = spec_gloss.a * smoothness_scale;
    }
    if (smoothness_from_albedo_alpha()) {
        smoothness = albedo_sample.a * smoothness_scale;
    }
    let roughness = pstd::roughness_from_smoothness(smoothness);

    let occlusion_sample = ts::sample_tex_2d(_OcclusionMap, _OcclusionMap_sampler, uv_main, mat._OcclusionMap_LodBias).g;
    let occlusion = pstd::occlusion_from_sample(occlusion_sample, mat._OcclusionStrength);

    let detail_enabled = pbs_kw(PBSSPECULAR_KW_DETAIL_MULX2);
    let detail_mask = pdet::sample_detail_mask(
        detail_enabled,
        _DetailMask,
        _DetailMask_sampler,
        uv_main,
        mat._DetailMask_LodBias,
    );
    base_color = pdet::apply_detail_albedo(
        base_color,
        detail_enabled,
        detail_mask,
        _DetailAlbedoMap,
        _DetailAlbedoMap_sampler,
        uv_detail,
        mat._DetailAlbedoMap_LodBias,
    );

    let n = pstd::sample_world_normal(
        pbs_kw(PBSSPECULAR_KW_NORMALMAP),
        detail_enabled,
        detail_mask,
        _BumpMap,
        _BumpMap_sampler,
        uv_main,
        mat._BumpMap_LodBias,
        mat._BumpScale,
        _DetailNormalMap,
        _DetailNormalMap_sampler,
        uv_detail,
        mat._DetailNormalMap_LodBias,
        mat._DetailNormalMapScale,
        world_n,
        world_t,
        world_n,
    );

    let emission = pstd::sample_emission(
        pbs_kw(PBSSPECULAR_KW_EMISSION),
        mat._EmissionColor.rgb,
        _EmissionMap,
        _EmissionMap_sampler,
        uv_main,
        mat._EmissionMap_LodBias,
    );

    return SurfaceData(base_color, base_alpha, specular_color, roughness, occlusion, n, emission);
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
    @location(5) uv1: vec2<f32>,
) -> mv::WorldUv2VertexOutput {
#ifdef MULTIVIEW
    return mv::world_uv2_vertex_main(instance_index, view_idx, pos, n, t, uv0, uv1);
#else
    return mv::world_uv2_vertex_main(instance_index, 0u, pos, n, t, uv0, uv1);
#endif
}

//#pass type=forward
@fragment
fn fs_forward_base(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) world_t: vec4<f32>,
    @location(3) uv0: vec2<f32>,
    @location(4) uv1: vec2<f32>,
    @location(5) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let s = sample_surface(uv0, uv1, world_pos, world_n, world_t, view_layer);
    let surface = psurf::specular_with_geometric_normal(
        s.base_color,
        s.alpha,
        s.specular_color,
        s.roughness,
        s.occlusion,
        s.normal,
        world_n,
        s.emission,
    );
    let options = plight::ClusterLightingOptions(
        true,
        true,
        specular_highlights_enabled(),
        glossy_reflections_enabled(),
    );
    if (alpha_premultiply_enabled()) {
        return plight::shade_specular_transparent_clustered(
            frag_pos.xy,
            world_pos,
            view_layer,
            surface,
            options,
        );
    }
    let color = plight::shade_specular_clustered(
        frag_pos.xy,
        world_pos,
        view_layer,
        surface,
        options,
    );
    return vec4<f32>(color, s.alpha);
}
