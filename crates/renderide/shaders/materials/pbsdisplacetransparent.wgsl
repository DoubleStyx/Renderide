//! Unity surface shader `Shader "PBSDisplaceTransparent"`: transparent metallic Standard
//! lighting with optional displacement modes.
//!
//! * `VERTEX_OFFSET`: scalar displacement along vertex normal (`_VertexOffsetMap.r`).
//! * `UV_OFFSET`: shifts the fragment main UV by `_UVOffsetMap.rg` x magnitude.
//! * `OBJECT_POS_OFFSET` / `VERTEX_POS_OFFSET`: shifts `_VertexOffsetMap` UVs from
//!   `_PositionOffsetMap.xy` sampled by object-origin or vertex world XZ.
//!
//! `_VertexOffsetMap` and `_PositionOffsetMap` sample from `vs_main` via
//! `textureSampleLevel(..., 0.0)`.
//!
//! Froox variant bits populate `_RenderideVariantBits`; this shader decodes
//! PBSDisplaceTransparent's shader-specific keyword bits locally.

//#render_queue Transparent
//#texture_default _MainTex white
//#texture_default _NormalMap bump
//#texture_default _EmissionMap black
//#texture_default _OcclusionMap white
//#texture_default _MetallicMap black
//#texture_default _VertexOffsetMap black
//#texture_default _UVOffsetMap black
//#texture_default _PositionOffsetMap black
//#mat_default _Color vec4 1.0 1.0 1.0 1.0
//#mat_default _NormalScale float 1.0
//#mat_default _PositionOffsetMagnitude vec4 1.0 1.0 0.0 0.0
//#mat_default _UVOffsetMagnitude float 0.1
//#mat_default _VertexOffsetMagnitude float 0.1
//#mat_default _AlphaClip float 0.5
//#mat_default _Glossiness float 0.5

#import renderide::draw::per_draw as pd
#import renderide::pbs::displace as pdisp
#import renderide::pbs::lighting as plight
#import renderide::pbs::sampling as psamp
#import renderide::pbs::surface as psurf
#import renderide::material::variant_bits as vb
#import renderide::core::uv as uvu
#import renderide::core::texture_sampling as ts

struct PbsDisplaceTransparentMaterial {
    _Color: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _VertexOffsetMap_ST: vec4<f32>,
    _UVOffsetMap_ST: vec4<f32>,
    _PositionOffsetMap_ST: vec4<f32>,
    _PositionOffsetMagnitude: vec4<f32>,
    _NormalScale: f32,
    _Glossiness: f32,
    _Metallic: f32,
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
    _MetallicMap_LodBias: f32,
    _OcclusionMap_LodBias: f32,
}

const PBSDISPT_KW_ALBEDOTEX: u32 = 1u << 0u;
const PBSDISPT_KW_ALPHACLIP: u32 = 1u << 1u;
const PBSDISPT_KW_EMISSIONTEX: u32 = 1u << 2u;
const PBSDISPT_KW_METALLICMAP: u32 = 1u << 3u;
const PBSDISPT_KW_NORMALMAP: u32 = 1u << 4u;
const PBSDISPT_KW_OCCLUSION: u32 = 1u << 5u;
const PBSDISPT_KW_OBJECT_POS_OFFSET: u32 = 1u << 6u;
const PBSDISPT_KW_UV_OFFSET: u32 = 1u << 7u;
const PBSDISPT_KW_VERTEX_OFFSET: u32 = 1u << 8u;
const PBSDISPT_KW_VERTEX_POS_OFFSET: u32 = 1u << 9u;

@group(1) @binding(0)  var<uniform> mat: PbsDisplaceTransparentMaterial;
@group(1) @binding(1)  var _MainTex: texture_2d<f32>;
@group(1) @binding(2)  var _MainTex_sampler: sampler;
@group(1) @binding(3)  var _NormalMap: texture_2d<f32>;
@group(1) @binding(4)  var _NormalMap_sampler: sampler;
@group(1) @binding(5)  var _EmissionMap: texture_2d<f32>;
@group(1) @binding(6)  var _EmissionMap_sampler: sampler;
@group(1) @binding(7)  var _OcclusionMap: texture_2d<f32>;
@group(1) @binding(8)  var _OcclusionMap_sampler: sampler;
@group(1) @binding(9)  var _MetallicMap: texture_2d<f32>;
@group(1) @binding(10) var _MetallicMap_sampler: sampler;
@group(1) @binding(11) var _VertexOffsetMap: texture_2d<f32>;
@group(1) @binding(12) var _VertexOffsetMap_sampler: sampler;
@group(1) @binding(13) var _UVOffsetMap: texture_2d<f32>;
@group(1) @binding(14) var _UVOffsetMap_sampler: sampler;
@group(1) @binding(15) var _PositionOffsetMap: texture_2d<f32>;
@group(1) @binding(16) var _PositionOffsetMap_sampler: sampler;

fn pbsdispt_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn kw_ALBEDOTEX() -> bool { return pbsdispt_kw(PBSDISPT_KW_ALBEDOTEX); }
fn kw_ALPHACLIP() -> bool { return pbsdispt_kw(PBSDISPT_KW_ALPHACLIP); }
fn kw_EMISSIONTEX() -> bool { return pbsdispt_kw(PBSDISPT_KW_EMISSIONTEX); }
fn kw_METALLICMAP() -> bool { return pbsdispt_kw(PBSDISPT_KW_METALLICMAP); }
fn kw_NORMALMAP() -> bool { return pbsdispt_kw(PBSDISPT_KW_NORMALMAP); }
fn kw_OCCLUSION() -> bool { return pbsdispt_kw(PBSDISPT_KW_OCCLUSION); }
fn kw_OBJECT_POS_OFFSET() -> bool { return pbsdispt_kw(PBSDISPT_KW_OBJECT_POS_OFFSET); }
fn kw_UV_OFFSET() -> bool { return pbsdispt_kw(PBSDISPT_KW_UV_OFFSET); }
fn kw_VERTEX_OFFSET() -> bool { return pbsdispt_kw(PBSDISPT_KW_VERTEX_OFFSET); }
fn kw_VERTEX_POS_OFFSET() -> bool { return pbsdispt_kw(PBSDISPT_KW_VERTEX_POS_OFFSET); }

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

    var metallic = mat._Metallic;
    var smoothness = mat._Glossiness;
    if (kw_METALLICMAP()) {
        let m = ts::sample_tex_2d(_MetallicMap, _MetallicMap_sampler, uv_main, mat._MetallicMap_LodBias);
        metallic = m.r;
        smoothness = m.a;
    }
    metallic = clamp(metallic, 0.0, 1.0);
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
    let surface = psurf::metallic_with_geometric_normal(
        base_color,
        c.a,
        metallic,
        roughness,
        occlusion,
        n,
        world_n,
        emission,
    );
    let options = plight::ClusterLightingOptions(include_directional, include_local, true, true);
    return plight::shade_metallic_transparent_clustered(frag_xy, world_pos, view_layer, surface, options);
}

//#pass type=forward name=forward_transparent blend=transparent_material zwrite=material(off) cull=material(off) color_mask=material(rgba)
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
