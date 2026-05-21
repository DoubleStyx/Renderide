//! Unity surface shader `Shader "PBSStencilSpecular"`: SpecularSetup lighting through the
//! standard forward path; host applies stencil ops driven by `_Stencil*` and `_ColorMask`.
//! Sibling of [`pbsstencil`](super::pbsstencil); swaps to the SpecularSetup BRDF and reads
//! tinted f0 + smoothness from `_SpecularColor` / `_SpecularMap` instead of metallic-gloss.
//!
//! Froox variant bits populate `_RenderideVariantBits`; this shader decodes
//! PBSStencilSpecular's shader-specific keyword bits locally.

//#texture_default _MainTex white
//#texture_default _NormalMap bump
//#texture_default _EmissionMap black
//#texture_default _OcclusionMap white
//#texture_default _SpecularMap white
//#mat_default _Color vec4 1.0 1.0 1.0 1.0
//#mat_default _NormalScale float 1.0
//#mat_default _SpecularColor vec4 1.0 1.0 1.0 0.5

#import renderide::mesh::vertex as mv
#import renderide::material::variant_bits as vb
#import renderide::pbs::lighting as plight
#import renderide::pbs::sampling as psamp
#import renderide::pbs::surface as psurf
#import renderide::core::texture_sampling as ts
#import renderide::core::uv as uvu

struct PbsStencilSpecularMaterial {
    _Color: vec4<f32>,
    _SpecularColor: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _NormalScale: f32,
    _RenderideVariantBits: u32,
    _MainTex_LodBias: f32,
    _NormalMap_LodBias: f32,
    _EmissionMap_LodBias: f32,
    _OcclusionMap_LodBias: f32,
    _SpecularMap_LodBias: f32,
    _pad0: f32,
}

const PBSSTENCILSPECULAR_KW_ALBEDOTEX: u32 = 1u << 0u;
const PBSSTENCILSPECULAR_KW_EMISSIONTEX: u32 = 1u << 1u;
const PBSSTENCILSPECULAR_KW_NORMALMAP: u32 = 1u << 2u;
const PBSSTENCILSPECULAR_KW_OCCLUSION: u32 = 1u << 3u;
const PBSSTENCILSPECULAR_KW_SPECULARMAP: u32 = 1u << 4u;

@group(1) @binding(0)  var<uniform> mat: PbsStencilSpecularMaterial;
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

fn pbs_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn sample_normal_world(uv_main: vec2<f32>, world_n: vec3<f32>, world_t: vec4<f32>) -> vec3<f32> {
    return psamp::sample_optional_world_normal(
        pbs_kw(PBSSTENCILSPECULAR_KW_NORMALMAP),
        _NormalMap,
        _NormalMap_sampler,
        uv_main,
        mat._NormalMap_LodBias,
        mat._NormalScale,
        world_n,
        world_t,
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
    @location(4) t: vec4<f32>,
) -> mv::WorldVertexOutput {
#ifdef MULTIVIEW
    return mv::world_vertex_main(instance_index, view_idx, pos, n, t, uv0);
#else
    return mv::world_vertex_main(instance_index, 0u, pos, n, t, uv0);
#endif
}

fn shade(
    frag_xy: vec2<f32>,
    world_pos: vec3<f32>,
    world_n: vec3<f32>,
    world_t: vec4<f32>,
    uv0: vec2<f32>,
    view_layer: u32,
    include_directional: bool,
    include_local: bool,
) -> vec4<f32> {
    let uv_main = uvu::apply_st(uv0, mat._MainTex_ST);
    var c = mat._Color;
    if (pbs_kw(PBSSTENCILSPECULAR_KW_ALBEDOTEX)) {
        c = c * ts::sample_tex_2d(_MainTex, _MainTex_sampler, uv_main, mat._MainTex_LodBias);
    }

    var spec = mat._SpecularColor;
    if (pbs_kw(PBSSTENCILSPECULAR_KW_SPECULARMAP)) {
        spec = ts::sample_tex_2d(_SpecularMap, _SpecularMap_sampler, uv_main, mat._SpecularMap_LodBias);
    }
    let f0 = clamp(spec.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let smoothness = clamp(spec.a, 0.0, 1.0);
    let roughness = psamp::roughness_from_smoothness(smoothness);

    var occlusion = 1.0;
    if (pbs_kw(PBSSTENCILSPECULAR_KW_OCCLUSION)) {
        occlusion = ts::sample_tex_2d(_OcclusionMap, _OcclusionMap_sampler, uv_main, mat._OcclusionMap_LodBias).r;
    }

    var emission = mat._EmissionColor.rgb;
    if (pbs_kw(PBSSTENCILSPECULAR_KW_EMISSIONTEX)) {
        emission = emission * ts::sample_tex_2d(_EmissionMap, _EmissionMap_sampler, uv_main, mat._EmissionMap_LodBias).rgb;
    }

    let n = sample_normal_world(uv_main, world_n, world_t);
    let base_color = c.rgb;
    let surface = psurf::specular_with_geometric_normal(base_color, c.a, f0, roughness, occlusion, n, world_n, emission);
    let options = plight::ClusterLightingOptions(include_directional, include_local, true, true);
    return vec4<f32>(
        plight::shade_specular_clustered(frag_xy, world_pos, view_layer, surface, options),
        c.a,
    );
}

//#pass type=forward color_mask=material(rgba) stencil=material offset=material(0,0)
@fragment
fn fs_forward_base(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) world_t: vec4<f32>,
    @location(3) uv0: vec2<f32>,
    @location(4) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    return shade(frag_pos.xy, world_pos, world_n, world_t, uv0, view_layer, true, true);
}
