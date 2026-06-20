//! Unity surface shader `Shader "PBSDistanceLerp"`: metallic Standard lighting with vertex
//! displacement and emission driven by distance to a list of up to 16 reference points.
//!
//! Each vertex computes its distance to each active `_Points[i]` (after optional grid snap with
//! `_DistanceGridSize` / `_DistanceGridOffset`), and accumulates two lerps:
//! displacement magnitude (between `_DisplaceMagnitudeFrom`/`To` over the
//! `[_DisplaceDistanceFrom, _DisplaceDistanceTo]` band) and emission color (between
//! `_EmissionColorFrom`/`To` over the `[_EmissionDistanceFrom, _EmissionDistanceTo]` band, scaled
//! by the per-point `_TintColors[i]`). The displacement is applied along the surface normal
//! unless `OVERRIDE_DISPLACE_DIRECTION` is set, in which case it follows
//! `_DisplacementDirection.xyz`. Reference space defaults to world; `LOCAL_SPACE` switches to
//! object space.
//!
//! Precedent for the fixed-size 16-element arrays: `pbsslice.wgsl` ships with
//! `_Slicers: array<vec4<f32>, 8>` and the host CPU packing is known to support indexed array
//! material properties.
//!
//! Froox variant bits populate `_RenderideVariantBits`; this shader decodes PBSDistanceLerp's
//! shader-specific keyword bits locally.

//#texture_default _MainTex white
//#texture_default _NormalMap bump
//#texture_default _EmissionMap black
//#texture_default _OcclusionMap white
//#texture_default _MetallicMap black
//#mat_default _Color vec4 1.0 1.0 1.0 1.0
//#mat_default _DisplaceDistanceFrom float 1.0
//#mat_default _DisplaceMagnitudeTo float 0.1
//#mat_default _DisplacementDirection vec4 0.0 1.0 0.0 0.0
//#mat_default _EmissionColorTo vec4 1.5 1.5 1.5 0.0
//#mat_default _EmissionDistanceFrom float 1.0
//#mat_default _NormalScale float 1.0
//#mat_default _Glossiness float 0.5

#import renderide::draw::per_draw as pd
#import renderide::pbs::families::distance_lerp as pdist
#import renderide::pbs::lighting as plight
#import renderide::pbs::sampling as psamp
#import renderide::pbs::surface as psurf
#import renderide::material::variant_bits as vb
#import renderide::core::texture_sampling as ts

struct PbsDistanceLerpMaterial {
    _Color: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _DistanceGridSize: vec4<f32>,
    _DistanceGridOffset: vec4<f32>,
    _EmissionColorFrom: vec4<f32>,
    _EmissionColorTo: vec4<f32>,
    _DisplacementDirection: vec4<f32>,
    _NormalScale: f32,
    _Glossiness: f32,
    _Metallic: f32,
    _DisplaceDistanceFrom: f32,
    _DisplaceDistanceTo: f32,
    _DisplaceMagnitudeFrom: f32,
    _DisplaceMagnitudeTo: f32,
    _EmissionDistanceFrom: f32,
    _EmissionDistanceTo: f32,
    _PointCount: f32,
    _RenderideVariantBits: u32,
    _MainTex_LodBias: f32,
    _NormalMap_LodBias: f32,
    _EmissionMap_LodBias: f32,
    _OcclusionMap_LodBias: f32,
    _MetallicMap_LodBias: f32,
    _Points: array<vec4<f32>, 16>,
    _TintColors: array<vec4<f32>, 16>,
}

const PBSDL_KW_METALLICMAP: u32 = 1u << 0u;
const PBSDL_KW_NORMALMAP: u32 = 1u << 1u;
const PBSDL_KW_LOCAL_SPACE: u32 = 1u << 2u;
const PBSDL_KW_OVERRIDE_DISPLACE_DIRECTION: u32 = 1u << 3u;
const PBSDL_KW_WORLD_SPACE: u32 = 1u << 4u;

@group(1) @binding(0)  var<uniform> mat: PbsDistanceLerpMaterial;
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

fn pbsdl_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn kw_METALLICMAP() -> bool { return pbsdl_kw(PBSDL_KW_METALLICMAP); }
fn kw_NORMALMAP() -> bool { return pbsdl_kw(PBSDL_KW_NORMALMAP); }
fn kw_LOCAL_SPACE() -> bool { return pbsdl_kw(PBSDL_KW_LOCAL_SPACE); }
fn kw_OVERRIDE_DISPLACE_DIRECTION() -> bool { return pbsdl_kw(PBSDL_KW_OVERRIDE_DISPLACE_DIRECTION); }

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(4) t: vec4<f32>,
) -> pdist::VertexOutput {
    let d = pd::get_draw(instance_index);
#ifdef MULTIVIEW
    return pdist::vertex_main(
        d,
        instance_index,
        view_idx,
        pos,
        n,
        t,
        kw_LOCAL_SPACE(),
        kw_OVERRIDE_DISPLACE_DIRECTION(),
        mat._DistanceGridSize.xyz,
        mat._DistanceGridOffset.xyz,
        mat._DisplacementDirection.xyz,
        mat._PointCount,
        mat._Points,
        mat._TintColors,
        mat._DisplaceDistanceFrom,
        mat._DisplaceDistanceTo,
        mat._DisplaceMagnitudeFrom,
        mat._DisplaceMagnitudeTo,
        mat._EmissionDistanceFrom,
        mat._EmissionDistanceTo,
        mat._EmissionColorFrom,
        mat._EmissionColorTo,
    );
#else
    return pdist::vertex_main(
        d,
        instance_index,
        0u,
        pos,
        n,
        t,
        kw_LOCAL_SPACE(),
        kw_OVERRIDE_DISPLACE_DIRECTION(),
        mat._DistanceGridSize.xyz,
        mat._DistanceGridOffset.xyz,
        mat._DisplacementDirection.xyz,
        mat._PointCount,
        mat._Points,
        mat._TintColors,
        mat._DisplaceDistanceFrom,
        mat._DisplaceDistanceTo,
        mat._DisplaceMagnitudeFrom,
        mat._DisplaceMagnitudeTo,
        mat._EmissionDistanceFrom,
        mat._EmissionDistanceTo,
        mat._EmissionColorFrom,
        mat._EmissionColorTo,
    );
#endif
}

fn shade(
    frag_xy: vec2<f32>,
    world_pos: vec3<f32>,
    world_n: vec3<f32>,
    world_t: vec4<f32>,
    point_emission: vec3<f32>,
    view_layer: u32,
    front_facing: bool,
    include_directional: bool,
    include_local: bool,
) -> vec4<f32> {
    let uv_main = vec2<f32>(0.0);
    let albedo_s = ts::sample_tex_2d(_MainTex, _MainTex_sampler, uv_main, mat._MainTex_LodBias);
    let base_color = (mat._Color * albedo_s).rgb;
    let alpha = mat._Color.a * albedo_s.a;

    let occlusion = ts::sample_tex_2d(_OcclusionMap, _OcclusionMap_sampler, uv_main, mat._OcclusionMap_LodBias).r;

    var metallic = mat._Metallic;
    var smoothness = mat._Glossiness;
    if (kw_METALLICMAP()) {
        let m = ts::sample_tex_2d(_MetallicMap, _MetallicMap_sampler, uv_main, mat._MetallicMap_LodBias);
        metallic = m.r;
        smoothness = m.a;
    }
    metallic = clamp(metallic, 0.0, 1.0);
    let roughness = psamp::roughness_from_smoothness(smoothness);

    let n = psamp::sample_optional_two_sided_world_normal(
        kw_NORMALMAP(),
        _NormalMap,
        _NormalMap_sampler,
        uv_main,
        mat._NormalMap_LodBias,
        mat._NormalScale,
        world_n,
        world_t,
        front_facing,
    );

    let emission_tex = ts::sample_tex_2d(_EmissionMap, _EmissionMap_sampler, uv_main, mat._EmissionMap_LodBias).rgb;
    let emission = mat._EmissionColor.rgb * emission_tex + point_emission;
    let surface = psurf::metallic_with_geometric_normal(
        base_color,
        alpha,
        metallic,
        roughness,
        occlusion,
        n,
        psamp::two_sided_geometric_normal(world_n, front_facing),
        emission,
    );
    let options = plight::ClusterLightingOptions(include_directional, include_local, true, true);
    let color = plight::shade_metallic_clustered(frag_xy, world_pos, view_layer, surface, options);
    return vec4<f32>(color, alpha);
}

//#pass type=forward cull=material(off) offset=material(0,0)
@fragment
fn fs_forward_base(
    @builtin(position) frag_pos: vec4<f32>,
    @builtin(front_facing) front_facing: bool,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) world_t: vec4<f32>,
    @location(3) point_emission: vec3<f32>,
    @location(4) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    return shade(frag_pos.xy, world_pos, world_n, world_t, point_emission, view_layer, front_facing, true, true);
}
