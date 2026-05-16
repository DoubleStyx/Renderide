//! Unity surface shader `Shader "PBSDualSidedSpecular"`: specular Standard lighting with two-sided normals.
//!
//! Unity's `#pragma surface surf StandardSpecular fullforwardshadows addshadow` generates forward
//! base, forward additive, and shadow caster passes. This renderer has a forward color path here, so
//! the shader declares the forward base + forward additive passes and keeps culling disabled.
//!
//! Froox variant bits populate `_RenderideVariantBits`; this shader decodes PBSDualSidedSpecular's
//! shader-specific keyword bits locally.

//#texture_default _MainTex white
//#texture_default _NormalMap bump
//#texture_default _EmissionMap black
//#texture_default _OcclusionMap white
//#texture_default _SpecularMap white
//#mat_default _Color vec4 1.0 1.0 1.0 1.0
//#mat_default _NormalScale float 1.0
//#mat_default _SpecularColor vec4 1.0 1.0 1.0 0.5

#import renderide::mesh::vertex as mv
#import renderide::pbs::lighting as plight
#import renderide::pbs::sampling as psamp
#import renderide::pbs::surface as psurf
#import renderide::material::variant_bits as vb
#import renderide::core::uv as uvu

struct PbsDualSidedSpecularMaterial {
    _Color: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _SpecularColor: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _NormalScale: f32,
    _AlphaClip: f32,
    _RenderideVariantBits: u32,
}

const PBSDS_KW_ALBEDOTEX: u32 = 1u << 0u;
const PBSDS_KW_ALPHACLIP: u32 = 1u << 1u;
const PBSDS_KW_EMISSIONTEX: u32 = 1u << 2u;
const PBSDS_KW_NORMALMAP: u32 = 1u << 3u;
const PBSDS_KW_OCCLUSION: u32 = 1u << 4u;
const PBSDS_KW_SPECULARMAP: u32 = 1u << 5u;
const PBSDS_KW_VCOLOR_ALBEDO: u32 = 1u << 6u;
const PBSDS_KW_VCOLOR_EMIT: u32 = 1u << 7u;
const PBSDS_KW_VCOLOR_SPECULAR: u32 = 1u << 8u;

@group(1) @binding(0)  var<uniform> mat: PbsDualSidedSpecularMaterial;
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

fn pbsds_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn kw_ALBEDOTEX() -> bool { return pbsds_kw(PBSDS_KW_ALBEDOTEX); }
fn kw_ALPHACLIP() -> bool { return pbsds_kw(PBSDS_KW_ALPHACLIP); }
fn kw_EMISSIONTEX() -> bool { return pbsds_kw(PBSDS_KW_EMISSIONTEX); }
fn kw_NORMALMAP() -> bool { return pbsds_kw(PBSDS_KW_NORMALMAP); }
fn kw_OCCLUSION() -> bool { return pbsds_kw(PBSDS_KW_OCCLUSION); }
fn kw_SPECULARMAP() -> bool { return pbsds_kw(PBSDS_KW_SPECULARMAP); }
fn kw_VCOLOR_ALBEDO() -> bool { return pbsds_kw(PBSDS_KW_VCOLOR_ALBEDO); }
fn kw_VCOLOR_EMIT() -> bool { return pbsds_kw(PBSDS_KW_VCOLOR_EMIT); }
fn kw_VCOLOR_SPECULAR() -> bool { return pbsds_kw(PBSDS_KW_VCOLOR_SPECULAR); }

struct SurfaceData {
    base_color: vec3<f32>,
    alpha: f32,
    f0: vec3<f32>,
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
    if (kw_ALPHACLIP() && albedo.a <= mat._AlphaClip) {
        discard;
    }

    var spec = mat._SpecularColor;
    if (kw_SPECULARMAP()) {
        spec = textureSample(_SpecularMap, _SpecularMap_sampler, uv_main);
    }
    if (kw_VCOLOR_SPECULAR()) {
        spec = spec * vertex_color;
    }
    let f0 = clamp(spec.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let smoothness = clamp(spec.a, 0.0, 1.0);
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
        f0,
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

//#pass forward_two_sided
@fragment
fn fs_forward_base(
    @builtin(position) frag_pos: vec4<f32>,
    @builtin(front_facing) front_facing: bool,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) world_t: vec4<f32>,
    @location(3) uv0: vec2<f32>,
    @location(4) color: vec4<f32>,
    @location(5) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let s = sample_surface(uv0, world_n, world_t, front_facing, color);
    let surface = psurf::specular_with_geometric_normal(
        s.base_color,
        s.alpha,
        s.f0,
        s.roughness,
        s.occlusion,
        s.normal,
        psamp::two_sided_geometric_normal(world_n, front_facing),
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
