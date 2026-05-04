//! Unity Standard **specular** PBS (`Shader "PBSSpecular"` / Standard SpecularSetup): clustered
//! forward Cook-Torrance BRDF with specular f0 color and Unity-style diffuse energy conservation.
//!
//! Build emits `pbsspecular_default` / `pbsspecular_multiview`. `@group(1)` names match Unity
//! material properties. ForwardAdd, lightmaps, and reflection probes are not implemented yet.


#import renderide::mesh::vertex as mv
#import renderide::pbs::normal as pnorm
#import renderide::pbs::parallax as ppar
#import renderide::pbs::lighting as plight
#import renderide::pbs::surface as psurf
#import renderide::alpha_clip_sample as acs
#import renderide::uv_utils as uvu
#import renderide::normal_decode as nd
#import renderide::texture_sampling as ts

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
    _NORMALMAP: f32,
    _ALPHATEST_ON: f32,
    _ALPHABLEND_ON: f32,
    _ALPHAPREMULTIPLY_ON: f32,
    _EMISSION: f32,
    _SPECGLOSSMAP: f32,
    _DETAIL_MULX2: f32,
    _SMOOTHNESS_TEXTURE_ALBEDO_CHANNEL_A: f32,
    _SPECULARHIGHLIGHTS_OFF: f32,
    _GLOSSYREFLECTIONS_OFF: f32,
    _PARALLAXMAP: f32,
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

fn kw(v: f32) -> bool {
    return v > 0.5;
}

fn alpha_test_enabled() -> bool {
    return kw(mat._ALPHATEST_ON);
}

fn alpha_premultiply_enabled() -> bool {
    return kw(mat._ALPHAPREMULTIPLY_ON);
}

fn spec_gloss_map_enabled() -> bool {
    return kw(mat._SPECGLOSSMAP);
}

fn specular_highlights_enabled() -> bool {
    return !kw(mat._SPECULARHIGHLIGHTS_OFF);
}

fn glossy_reflections_enabled() -> bool {
    return !kw(mat._GLOSSYREFLECTIONS_OFF);
}

fn smoothness_from_albedo_alpha() -> bool {
    return mat._SmoothnessTextureChannel > 0.5 || kw(mat._SMOOTHNESS_TEXTURE_ALBEDO_CHANNEL_A);
}

fn uv_with_parallax(uv: vec2<f32>, world_pos: vec3<f32>, world_n: vec3<f32>, world_t: vec4<f32>, view_layer: u32) -> vec2<f32> {
    if (!kw(mat._PARALLAXMAP)) {
        return uv;
    }
    let h = ts::sample_tex_2d(_ParallaxMap, _ParallaxMap_sampler, uv, mat._ParallaxMap_LodBias).g;
    return uv + ppar::unity_parallax_offset(h, mat._Parallax, world_pos, world_n, world_t, view_layer);
}

fn sample_normal_world(
    uv_main: vec2<f32>,
    uv_detail: vec2<f32>,
    world_n: vec3<f32>,
    world_t: vec4<f32>,
    detail_mask: f32,
) -> vec3<f32> {
    var n = world_n;
    if (!kw(mat._NORMALMAP)) {
        return n;
    }

    let tbn = pnorm::orthonormal_tbn(n, world_t);
    var ts_n = nd::decode_ts_normal_with_placeholder_sample(
        ts::sample_tex_2d(_BumpMap, _BumpMap_sampler, uv_main, mat._BumpMap_LodBias),
        mat._BumpScale,
    );

    if (kw(mat._DETAIL_MULX2) && detail_mask > 0.001) {
        let ts_detail = nd::decode_ts_normal_with_placeholder_sample(
            ts::sample_tex_2d(_DetailNormalMap, _DetailNormalMap_sampler, uv_detail, mat._DetailNormalMap_LodBias),
            mat._DetailNormalMapScale,
        );
        ts_n = normalize(vec3<f32>(ts_n.xy + ts_detail.xy * detail_mask, ts_n.z));
    }

    return normalize(tbn * ts_n);
}

fn detail_uv(uv0: vec2<f32>, uv1: vec2<f32>) -> vec2<f32> {
    return uvu::apply_st(select(uv0, uv1, mat._UVSec >= 0.5), mat._DetailAlbedoMap_ST);
}

fn sample_surface(uv0: vec2<f32>, uv1: vec2<f32>, world_pos: vec3<f32>, world_n: vec3<f32>, world_t: vec4<f32>, view_layer: u32) -> SurfaceData {
    let uv_base = uvu::apply_st(uv0, mat._MainTex_ST);
    let uv_main = uv_with_parallax(uv_base, world_pos, world_n, world_t, view_layer);
    let uv_detail = detail_uv(uv0, uv1);

    let albedo_sample = ts::sample_tex_2d(_MainTex, _MainTex_sampler, uv_main, mat._MainTex_LodBias);
    let base_alpha = mat._Color.a * albedo_sample.a;
    let clip_alpha = mat._Color.a * acs::texture_alpha_base_mip(_MainTex, _MainTex_sampler, uv_main);
    if (alpha_test_enabled() && clip_alpha <= mat._Cutoff) {
        discard;
    }

    var base_color = mat._Color.rgb * albedo_sample.rgb;

    let spec_gloss = ts::sample_tex_2d(_SpecGlossMap, _SpecGlossMap_sampler, uv_main, mat._SpecGlossMap_LodBias);
    var specular_color = mat._SpecColor.rgb;
    var smoothness = mat._Glossiness;
    if (spec_gloss_map_enabled()) {
        specular_color = spec_gloss.rgb;
        smoothness = spec_gloss.a * mat._GlossMapScale;
    }
    if (smoothness_from_albedo_alpha()) {
        smoothness = albedo_sample.a * mat._GlossMapScale;
    }
    let roughness = clamp(1.0 - clamp(smoothness, 0.0, 1.0), 0.045, 1.0);

    let occlusion_sample = ts::sample_tex_2d(_OcclusionMap, _OcclusionMap_sampler, uv_main, mat._OcclusionMap_LodBias).r;
    let occlusion = mix(1.0, occlusion_sample, clamp(mat._OcclusionStrength, 0.0, 1.0));

    var detail_mask = 0.0;
    if (kw(mat._DETAIL_MULX2)) {
        detail_mask = ts::sample_tex_2d(_DetailMask, _DetailMask_sampler, uv_main, mat._DetailMask_LodBias).a;
        let detail = ts::sample_tex_2d(_DetailAlbedoMap, _DetailAlbedoMap_sampler, uv_detail, mat._DetailAlbedoMap_LodBias).rgb;
        base_color = base_color * mix(vec3<f32>(1.0), detail * 2.0, detail_mask);
    }

    let n = sample_normal_world(uv_main, uv_detail, world_n, world_t, detail_mask);

    let emission_color = mat._EmissionColor.rgb;
    var emission = vec3<f32>(0.0);
    if (kw(mat._EMISSION) || dot(emission_color, emission_color) > 1e-8) {
        emission = ts::sample_tex_2d(_EmissionMap, _EmissionMap_sampler, uv_main, mat._EmissionMap_LodBias).rgb * emission_color;
    }

    return SurfaceData(base_color, base_alpha, specular_color, roughness, occlusion, n, emission);
}

fn apply_premultiply(color: vec3<f32>, alpha: f32) -> vec3<f32> {
    return select(color, color * alpha, alpha_premultiply_enabled());
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

//#pass forward
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
    let surface = psurf::specular(
        s.base_color,
        s.alpha,
        s.specular_color,
        s.roughness,
        s.occlusion,
        s.normal,
        s.emission,
    );
    let color = plight::shade_specular_clustered(
        frag_pos.xy,
        world_pos,
        view_layer,
        surface,
        plight::ClusterLightingOptions(
            true,
            true,
            specular_highlights_enabled(),
            glossy_reflections_enabled(),
        ),
    );
    return vec4<f32>(apply_premultiply(color, s.alpha), s.alpha);
}
