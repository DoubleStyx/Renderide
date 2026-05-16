//! Unity surface shader `Shader "PBSDualSidedTransparent"`: metallic Standard lighting with
//! transparent two-pass dual-sided rendering.
//!
//! Unity renders this shader as back faces first (`Cull Front`) and front faces second
//! (`Cull Back`). The two WGSL fragment entries keep that pass topology while sharing the same
//! surface evaluation.
//!
//! Froox variant bits populate `_RenderideVariantBits`; this shader decodes
//! PBSDualSidedTransparent's shader-specific keyword bits locally.

//#texture_default _MainTex white
//#texture_default _NormalMap bump
//#texture_default _EmissionMap black
//#texture_default _OcclusionMap white
//#texture_default _MetallicMap black
//#mat_default _Color vec4 1.0 1.0 1.0 1.0
//#mat_default _NormalScale float 1.0

#import renderide::mesh::vertex as mv
#import renderide::pbs::lighting as plight
#import renderide::pbs::sampling as psamp
#import renderide::pbs::surface as psurf
#import renderide::material::variant_bits as vb
#import renderide::core::uv as uvu

struct PbsDualSidedTransparentMaterial {
    _Color: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _NormalScale: f32,
    _Glossiness: f32,
    _Metallic: f32,
    _RenderideVariantBits: u32,
}

const PBSDST_KW_ALBEDOTEX: u32 = 1u << 0u;
const PBSDST_KW_EMISSIONTEX: u32 = 1u << 1u;
const PBSDST_KW_METALLICMAP: u32 = 1u << 2u;
const PBSDST_KW_NORMALMAP: u32 = 1u << 3u;
const PBSDST_KW_OCCLUSION: u32 = 1u << 4u;
const PBSDST_KW_VCOLOR_ALBEDO: u32 = 1u << 5u;
const PBSDST_KW_VCOLOR_EMIT: u32 = 1u << 6u;
const PBSDST_KW_VCOLOR_METALLIC: u32 = 1u << 7u;

@group(1) @binding(0)  var<uniform> mat: PbsDualSidedTransparentMaterial;
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

fn pbsdst_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn kw_ALBEDOTEX() -> bool { return pbsdst_kw(PBSDST_KW_ALBEDOTEX); }
fn kw_EMISSIONTEX() -> bool { return pbsdst_kw(PBSDST_KW_EMISSIONTEX); }
fn kw_METALLICMAP() -> bool { return pbsdst_kw(PBSDST_KW_METALLICMAP); }
fn kw_NORMALMAP() -> bool { return pbsdst_kw(PBSDST_KW_NORMALMAP); }
fn kw_OCCLUSION() -> bool { return pbsdst_kw(PBSDST_KW_OCCLUSION); }
fn kw_VCOLOR_ALBEDO() -> bool { return pbsdst_kw(PBSDST_KW_VCOLOR_ALBEDO); }
fn kw_VCOLOR_EMIT() -> bool { return pbsdst_kw(PBSDST_KW_VCOLOR_EMIT); }
fn kw_VCOLOR_METALLIC() -> bool { return pbsdst_kw(PBSDST_KW_VCOLOR_METALLIC); }

struct SurfaceData {
    base_color: vec3<f32>,
    alpha: f32,
    metallic: f32,
    roughness: f32,
    occlusion: f32,
    normal: vec3<f32>,
    emission: vec3<f32>,
}

fn sample_normal_world(uv_main: vec2<f32>, world_n: vec3<f32>, world_t: vec4<f32>, front_facing: bool) -> vec3<f32> {
    return psamp::sample_optional_two_sided_world_normal(
        kw_NORMALMAP(),
        _NormalMap,
        _NormalMap_sampler,
        uv_main,
        0.0,
        mat._NormalScale,
        world_n,
        world_t,
        front_facing,
    );
}

fn sample_surface(
    uv0: vec2<f32>,
    world_n: vec3<f32>,
    world_t: vec4<f32>,
    front_facing: bool,
    vertex_color: vec4<f32>,
) -> SurfaceData {
    let uv_main = uvu::apply_st(uv0, mat._MainTex_ST);

    var albedo = mat._Color;
    if (kw_ALBEDOTEX()) {
        albedo = albedo * textureSample(_MainTex, _MainTex_sampler, uv_main);
    }
    if (kw_VCOLOR_ALBEDO()) {
        albedo = albedo * vertex_color;
    }

    var metallic = mat._Metallic;
    var smoothness = mat._Glossiness;
    if (kw_METALLICMAP()) {
        let m = textureSample(_MetallicMap, _MetallicMap_sampler, uv_main);
        metallic = m.r;
        smoothness = m.a;
    }
    if (kw_VCOLOR_METALLIC()) {
        metallic = metallic * dot(vertex_color.rgb, vec3<f32>(0.33333334));
        smoothness = smoothness * vertex_color.a;
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
    if (kw_VCOLOR_EMIT()) {
        emission = emission * vertex_color.rgb;
    }

    return SurfaceData(
        albedo.rgb,
        albedo.a,
        metallic,
        roughness,
        occlusion,
        sample_normal_world(uv_main, world_n, world_t, front_facing),
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

fn shade(
    frag_xy: vec2<f32>,
    world_pos: vec3<f32>,
    world_n: vec3<f32>,
    world_t: vec4<f32>,
    uv0: vec2<f32>,
    vertex_color: vec4<f32>,
    view_layer: u32,
    front_facing: bool,
) -> vec4<f32> {
    let s = sample_surface(uv0, world_n, world_t, front_facing, vertex_color);
    let surface = psurf::metallic_with_geometric_normal(
        s.base_color,
        s.alpha,
        s.metallic,
        s.roughness,
        s.occlusion,
        s.normal,
        psamp::two_sided_geometric_normal(world_n, front_facing),
        s.emission,
    );
    return plight::shade_metallic_transparent_clustered(
        frag_xy,
        world_pos,
        view_layer,
        surface,
        plight::default_lighting_options(),
    );
}

//#pass forward_transparent_cull_front
@fragment
fn fs_back_faces(
    @builtin(position) frag_pos: vec4<f32>,
    @builtin(front_facing) front_facing: bool,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) world_t: vec4<f32>,
    @location(3) uv0: vec2<f32>,
    @location(4) color: vec4<f32>,
    @location(5) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    return shade(frag_pos.xy, world_pos, world_n, world_t, uv0, color, view_layer, front_facing);
}

//#pass forward_transparent_cull_back
@fragment
fn fs_front_faces(
    @builtin(position) frag_pos: vec4<f32>,
    @builtin(front_facing) front_facing: bool,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) world_t: vec4<f32>,
    @location(3) uv0: vec2<f32>,
    @location(4) color: vec4<f32>,
    @location(5) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    return shade(frag_pos.xy, world_pos, world_n, world_t, uv0, color, view_layer, front_facing);
}
