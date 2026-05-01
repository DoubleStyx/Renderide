//! Unity surface shader `Shader "PBSVertexColorTransparentSpecular"`: SpecularSetup lighting that
//! optionally multiplies albedo, emission, or specular color by the mesh vertex color.
//!
//! Sibling of [`pbsvertexcolortransparent`](super::pbsvertexcolortransparent); replaces the metallic
//! BRDF with the specular variant and reads tinted f0 + smoothness from `_SpecularColor` /
//! `_SpecularMap` instead of `_Metallic` / `_MetallicMap`.

#import renderide::alpha_clip_sample as acs
#import renderide::mesh::vertex as mv
#import renderide::pbs::lighting as plight
#import renderide::pbs::sampling as psamp
#import renderide::pbs::surface as psurf
#import renderide::uv_utils as uvu

struct PbsVertexColorTransparentSpecularMaterial {
    _Color: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _SpecularColor: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _MainTex_StorageVInverted: f32,
    _NormalScale: f32,
    _AlphaClip: f32,
    _ALPHACLIP: f32,
    _ALBEDOTEX: f32,
    _EMISSIONTEX: f32,
    _NORMALMAP: f32,
    _SPECULARMAP: f32,
    _OCCLUSION: f32,
    VCOLOR_ALBEDO: f32,
    VCOLOR_EMIT: f32,
    VCOLOR_SPECULAR: f32,
}

@group(1) @binding(0)  var<uniform> mat: PbsVertexColorTransparentSpecularMaterial;
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

fn sample_normal_world(uv_main: vec2<f32>, world_n: vec3<f32>, world_t: vec4<f32>) -> vec3<f32> {
    return psamp::sample_optional_world_normal(
        uvu::kw_enabled(mat._NORMALMAP),
        _NormalMap,
        _NormalMap_sampler,
        uv_main,
        0.0,
        mat._NormalScale,
        world_n,
        world_t,
    );
}

fn sample_surface(uv0: vec2<f32>, world_n: vec3<f32>, world_t: vec4<f32>, vertex_color: vec4<f32>) -> SurfaceData {
    let uv_main = uvu::apply_st_for_storage(uv0, mat._MainTex_ST, mat._MainTex_StorageVInverted);

    var albedo = mat._Color;
    if (uvu::kw_enabled(mat._ALBEDOTEX)) {
        albedo = albedo * textureSample(_MainTex, _MainTex_sampler, uv_main);
    }
    if (uvu::kw_enabled(mat.VCOLOR_ALBEDO)) {
        albedo = albedo * vertex_color;
    }
    let vertex_alpha = select(1.0, vertex_color.a, uvu::kw_enabled(mat.VCOLOR_ALBEDO));
    let clip_alpha = select(
        albedo.a,
        mat._Color.a
            * vertex_alpha
            * acs::texture_alpha_base_mip(_MainTex, _MainTex_sampler, uv_main),
        uvu::kw_enabled(mat._ALBEDOTEX),
    );
    if (uvu::kw_enabled(mat._ALPHACLIP) && clip_alpha <= mat._AlphaClip) {
        discard;
    }

    var spec = mat._SpecularColor;
    if (uvu::kw_enabled(mat._SPECULARMAP)) {
        spec = textureSample(_SpecularMap, _SpecularMap_sampler, uv_main);
    }
    if (uvu::kw_enabled(mat.VCOLOR_SPECULAR)) {
        spec = spec * vertex_color;
    }
    let f0 = clamp(spec.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let smoothness = clamp(spec.a, 0.0, 1.0);
    let roughness = psamp::roughness_from_smoothness(smoothness);

    var occlusion = 1.0;
    if (uvu::kw_enabled(mat._OCCLUSION)) {
        occlusion = textureSample(_OcclusionMap, _OcclusionMap_sampler, uv_main).r;
    }

    let emission_color = mat._EmissionColor.rgb;
    var emission = vec3<f32>(0.0);
    if (dot(emission_color, emission_color) > 1e-8) {
        emission = emission_color;
        if (uvu::kw_enabled(mat._EMISSIONTEX)) {
            emission = emission * textureSample(_EmissionMap, _EmissionMap_sampler, uv_main).rgb;
        }
    }
    if (uvu::kw_enabled(mat.VCOLOR_EMIT)) {
        emission = emission * vertex_color.rgb;
    }

    return SurfaceData(
        albedo.rgb,
        albedo.a,
        f0,
        roughness,
        1.0 - max(max(f0.r, f0.g), f0.b),
        occlusion,
        sample_normal_world(uv_main, world_n, world_t),
        emission,
    );
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
    @location(3) color: vec4<f32>,
    @location(4) t: vec4<f32>,
) -> mv::WorldColorVertexOutput {
#ifdef MULTIVIEW
    return mv::world_color_vertex_main(instance_index, view_idx, pos, n, t, uv0, color);
#else
    return mv::world_color_vertex_main(instance_index, 0u, pos, n, t, uv0, color);
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
    @location(4) color: vec4<f32>,
    @location(5) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let s = sample_surface(uv0, world_n, world_t, color);
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
        plight::shade_specular_clustered(frag_pos.xy, world_pos, view_layer, surface, plight::default_lighting_options()),
        s.alpha,
    );
}
