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

#import renderide::mesh::vertex as mv
#import renderide::draw::per_draw as pd
#import renderide::pbs::displace as pdisp
#import renderide::pbs::lighting as plight
#import renderide::pbs::normal as pnorm
#import renderide::pbs::sampling as psamp
#import renderide::pbs::surface as psurf
#import renderide::material::variant_bits as vb
#import renderide::core::normal_decode as nd
#import renderide::core::uv as uvu

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

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) world_t: vec4<f32>,
    @location(3) uv0: vec2<f32>,
    @location(4) @interpolate(flat) view_layer: u32,
}

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

fn sample_normal_world(uv_main: vec2<f32>, world_n: vec3<f32>, world_t: vec4<f32>, front_facing: bool) -> vec3<f32> {
    if (!kw_NORMALMAP()) {
        var n = normalize(world_n);
        if (!front_facing) {
            n = -n;
        }
        return n;
    }

    let tbn = pnorm::orthonormal_tbn(world_n, world_t);
    var ts_n = nd::decode_ts_normal_with_placeholder_sample(
        textureSample(_NormalMap, _NormalMap_sampler, uv_main),
        mat._NormalScale,
    );
    if (!front_facing) {
        ts_n.z = -ts_n.z;
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
    @location(4) t: vec4<f32>,
) -> VertexOutput {
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

    let world_p = d.model * vec4<f32>(displaced, 1.0);
    let wn = normalize(d.normal_matrix * n.xyz);
    let wt = mv::world_tangent(d, t);
#ifdef MULTIVIEW
    let vp = mv::select_view_proj(d, view_idx);
#else
    let vp = mv::select_view_proj(d, 0u);
#endif
    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.world_pos = world_p.xyz;
    out.world_n = wn;
    out.world_t = wt;
    out.uv0 = uv;
#ifdef MULTIVIEW
    out.view_layer = mv::packed_view_layer(instance_index, view_idx);
#else
    out.view_layer = mv::packed_view_layer(instance_index, 0u);
#endif
    return out;
}

fn shade(
    frag_xy: vec2<f32>,
    world_pos: vec3<f32>,
    world_n: vec3<f32>,
    world_t: vec4<f32>,
    uv0: vec2<f32>,
    view_layer: u32,
    front_facing: bool,
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
        _UVOffsetMap,
        _UVOffsetMap_sampler,
    );

    var c = mat._Color;
    if (kw_ALBEDOTEX()) {
        c = c * textureSample(_MainTex, _MainTex_sampler, uv_main);
    }
    if (kw_ALPHACLIP() && c.a <= mat._AlphaClip) {
        discard;
    }

    var metallic = mat._Metallic;
    var smoothness = mat._Glossiness;
    if (kw_METALLICMAP()) {
        let m = textureSample(_MetallicMap, _MetallicMap_sampler, uv_main);
        metallic = m.r;
        smoothness = m.a;
    }
    metallic = clamp(metallic, 0.0, 1.0);
    let roughness = psamp::roughness_from_smoothness(smoothness);

    var occlusion = 1.0;
    if (kw_OCCLUSION()) {
        occlusion = textureSample(_OcclusionMap, _OcclusionMap_sampler, uv_main).r;
    }

    var emission = mat._EmissionColor.rgb;
    if (kw_EMISSIONTEX()) {
        emission = emission * textureSample(_EmissionMap, _EmissionMap_sampler, uv_main).rgb;
    }

    let n = sample_normal_world(uv_main, world_n, world_t, front_facing);
    let base_color = c.rgb;
    let surface = psurf::metallic_with_geometric_normal(
        base_color,
        c.a,
        metallic,
        roughness,
        occlusion,
        n,
        psamp::two_sided_geometric_normal(world_n, front_facing),
        emission,
    );
    let options = plight::ClusterLightingOptions(include_directional, include_local, true, true);
    return plight::shade_metallic_transparent_clustered(frag_xy, world_pos, view_layer, surface, options);
}

//#pass forward_transparent
@fragment
fn fs_forward_base(
    @builtin(position) frag_pos: vec4<f32>,
    @builtin(front_facing) front_facing: bool,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) world_t: vec4<f32>,
    @location(3) uv0: vec2<f32>,
    @location(4) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    return shade(frag_pos.xy, world_pos, world_n, world_t, uv0, view_layer, front_facing, true, true);
}
