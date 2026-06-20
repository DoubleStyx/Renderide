//! Unity surface shader `Shader "PBSDisplaceSpecular"`: SpecularSetup lighting with the same
//! VERTEX_OFFSET / UV_OFFSET / OBJECT_POS_OFFSET / VERTEX_POS_OFFSET displacement modes as
//! [`pbsdisplace`](super::pbsdisplace).
//!
//! Reads tinted f0 + smoothness from `_SpecularColor` / `_SpecularMap` instead of metallic-gloss.
//!
//! Froox variant bits populate `_RenderideVariantBits`; this shader decodes PBSDisplaceSpecular's
//! shader-specific keyword bits locally.

//#texture_default _MainTex white
//#texture_default _NormalMap bump
//#texture_default _EmissionMap black
//#texture_default _OcclusionMap white
//#texture_default _SpecularMap white
//#texture_default _VertexOffsetMap black
//#texture_default _UVOffsetMap black
//#texture_default _PositionOffsetMap black
//#mat_default _Color vec4 1.0 1.0 1.0 1.0
//#mat_default _NormalScale float 1.0
//#mat_default _PositionOffsetMagnitude vec4 1.0 1.0 0.0 0.0
//#mat_default _SpecularColor vec4 1.0 1.0 1.0 0.5
//#mat_default _UVOffsetMagnitude float 0.1
//#mat_default _VertexOffsetMagnitude float 0.1
//#mat_default _AlphaClip float 0.5

#import renderide::draw::per_draw as pd
#import renderide::pbs::displace as pdisp
#import renderide::pbs::lighting as plight
#import renderide::pbs::sampling as psamp
#import renderide::pbs::surface as psurf
#import renderide::material::variant_bits as vb
#import renderide::core::uv as uvu
#import renderide::core::texture_sampling as ts

struct PbsDisplaceSpecularMaterial {
    _Color: vec4<f32>,
    _SpecularColor: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _VertexOffsetMap_ST: vec4<f32>,
    _UVOffsetMap_ST: vec4<f32>,
    _PositionOffsetMap_ST: vec4<f32>,
    _PositionOffsetMagnitude: vec4<f32>,
    _NormalScale: f32,
    _AlphaClip: f32,
    _VertexOffsetMagnitude: f32,
    _VertexOffsetBias: f32,
    _UVOffsetMagnitude: f32,
    _UVOffsetBias: f32,
    _RenderideVariantBits: u32,
    _UVOffsetMap_LodBias: f32,
    _NormalMap_LodBias: f32,
    _EmissionMap_LodBias: f32,
    _MainTex_LodBias: f32,
    _OcclusionMap_LodBias: f32,
    _SpecularMap_LodBias: f32,
}

const PBSDISPSPEC_KW_ALBEDOTEX: u32 = 1u << 0u;
const PBSDISPSPEC_KW_ALPHACLIP: u32 = 1u << 1u;
const PBSDISPSPEC_KW_EMISSIONTEX: u32 = 1u << 2u;
const PBSDISPSPEC_KW_NORMALMAP: u32 = 1u << 3u;
const PBSDISPSPEC_KW_OCCLUSION: u32 = 1u << 4u;
const PBSDISPSPEC_KW_SPECULARMAP: u32 = 1u << 5u;
const PBSDISPSPEC_KW_OBJECT_POS_OFFSET: u32 = 1u << 6u;
const PBSDISPSPEC_KW_UV_OFFSET: u32 = 1u << 7u;
const PBSDISPSPEC_KW_VERTEX_OFFSET: u32 = 1u << 8u;
const PBSDISPSPEC_KW_VERTEX_POS_OFFSET: u32 = 1u << 9u;

@group(1) @binding(0)  var<uniform> mat: PbsDisplaceSpecularMaterial;
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
@group(1) @binding(11) var _VertexOffsetMap: texture_2d<f32>;
@group(1) @binding(12) var _VertexOffsetMap_sampler: sampler;
@group(1) @binding(13) var _UVOffsetMap: texture_2d<f32>;
@group(1) @binding(14) var _UVOffsetMap_sampler: sampler;
@group(1) @binding(15) var _PositionOffsetMap: texture_2d<f32>;
@group(1) @binding(16) var _PositionOffsetMap_sampler: sampler;

fn pbsdispspec_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn kw_ALBEDOTEX() -> bool { return pbsdispspec_kw(PBSDISPSPEC_KW_ALBEDOTEX); }
fn kw_ALPHACLIP() -> bool { return pbsdispspec_kw(PBSDISPSPEC_KW_ALPHACLIP); }
fn kw_EMISSIONTEX() -> bool { return pbsdispspec_kw(PBSDISPSPEC_KW_EMISSIONTEX); }
fn kw_NORMALMAP() -> bool { return pbsdispspec_kw(PBSDISPSPEC_KW_NORMALMAP); }
fn kw_OCCLUSION() -> bool { return pbsdispspec_kw(PBSDISPSPEC_KW_OCCLUSION); }
fn kw_SPECULARMAP() -> bool { return pbsdispspec_kw(PBSDISPSPEC_KW_SPECULARMAP); }
fn kw_OBJECT_POS_OFFSET() -> bool { return pbsdispspec_kw(PBSDISPSPEC_KW_OBJECT_POS_OFFSET); }
fn kw_UV_OFFSET() -> bool { return pbsdispspec_kw(PBSDISPSPEC_KW_UV_OFFSET); }
fn kw_VERTEX_OFFSET() -> bool { return pbsdispspec_kw(PBSDISPSPEC_KW_VERTEX_OFFSET); }
fn kw_VERTEX_POS_OFFSET() -> bool { return pbsdispspec_kw(PBSDISPSPEC_KW_VERTEX_POS_OFFSET); }

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
) -> pdisp::VertexOutput {
    let d = pd::get_draw(instance_index);
    let displaced_uv = pdisp::apply_vertex_offsets(
        pos.xyz,
        n.xyz,
        uv0,
        d.model,
        kw_VERTEX_OFFSET(),
        kw_OBJECT_POS_OFFSET(),
        kw_VERTEX_POS_OFFSET(),
        mat._VertexOffsetMap_ST,
        mat._PositionOffsetMap_ST,
        mat._PositionOffsetMagnitude.xy,
        mat._VertexOffsetMagnitude,
        mat._VertexOffsetBias,
        _VertexOffsetMap,
        _VertexOffsetMap_sampler,
        _PositionOffsetMap,
        _PositionOffsetMap_sampler,
    );
    let displaced = displaced_uv.position;
    let uv = displaced_uv.uv;
#ifdef MULTIVIEW
    return pdisp::vertex_output(d, instance_index, view_idx, n, t, displaced, uv);
#else
    return pdisp::vertex_output(d, instance_index, 0u, n, t, displaced, uv);
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
    let uv_main_base = uvu::apply_st(uv0, mat._MainTex_ST);
    let uv_main = pdisp::apply_fragment_uv_offset(
        uv_main_base,
        uv0,
        kw_UV_OFFSET(),
        mat._UVOffsetMap_ST,
        mat._UVOffsetMagnitude,
        mat._UVOffsetBias,
        mat._UVOffsetMap_LodBias,
        _UVOffsetMap,
        _UVOffsetMap_sampler,
    );

    var c = mat._Color;
    if (kw_ALBEDOTEX()) {
        c = c * ts::sample_tex_2d(_MainTex, _MainTex_sampler, uv_main, mat._MainTex_LodBias);
    }
    if (kw_ALPHACLIP() && c.a < mat._AlphaClip) {
        discard;
    }

    var spec = mat._SpecularColor;
    if (kw_SPECULARMAP()) {
        spec = ts::sample_tex_2d(_SpecularMap, _SpecularMap_sampler, uv_main, mat._SpecularMap_LodBias);
    }
    let f0 = clamp(spec.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let smoothness = clamp(spec.a, 0.0, 1.0);
    let roughness = psamp::roughness_from_smoothness(smoothness);

    var occlusion = 1.0;
    if (kw_OCCLUSION()) {
        occlusion = ts::sample_tex_2d(_OcclusionMap, _OcclusionMap_sampler, uv_main, mat._OcclusionMap_LodBias).r;
    }

    var emission = mat._EmissionColor.rgb;
    if (kw_EMISSIONTEX()) {
        emission = emission * ts::sample_tex_2d(_EmissionMap, _EmissionMap_sampler, uv_main, mat._EmissionMap_LodBias).rgb;
    }

    let n = psamp::sample_optional_world_normal(
        kw_NORMALMAP(),
        _NormalMap,
        _NormalMap_sampler,
        uv_main,
        mat._NormalMap_LodBias,
        mat._NormalScale,
        world_n,
        world_t,
    );
    let base_color = c.rgb;
    let surface = psurf::specular_with_geometric_normal(
        base_color,
        c.a,
        f0,
        roughness,
        occlusion,
        n,
        world_n,
        emission,
    );
    let options = plight::ClusterLightingOptions(include_directional, include_local, true, true);
    return vec4<f32>(
        plight::shade_specular_clustered(frag_xy, world_pos, view_layer, surface, options),
        c.a,
    );
}

//#pass type=forward cull=material(off) a2c=cutout
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
