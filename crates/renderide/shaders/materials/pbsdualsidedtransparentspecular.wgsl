//! Unity surface shader `Shader "PBSDualSidedTransparentSpecular"`: Standard SpecularSetup with
//! two-sided normals, authored for transparent draws.
//!
//! Transparent two-pass dual-sided rendering: back faces first (`Cull Front`) and front faces
//! second (`Cull Back`), matching the metallic transparent variant's pass topology.
//!
//! Froox variant bits populate `_RenderideVariantBits`; this shader decodes
//! PBSDualSidedTransparentSpecular's shader-specific keyword bits locally. Unlike
//! [`pbsdualsidedspecular`](super::pbsdualsidedspecular), this transparent variant has no
//! `_ALPHACLIP` keyword.


//#texture_default _MainTex white
//#texture_default _NormalMap bump
//#texture_default _EmissionMap black
//#texture_default _OcclusionMap white
//#texture_default _SpecularMap white

#import renderide::mesh::vertex as mv
#import renderide::pbs::normal as pnorm
#import renderide::pbs::lighting as plight
#import renderide::pbs::surface as psurf
#import renderide::material::variant_bits as vb
#import renderide::core::uv as uvu
#import renderide::core::normal_decode as nd

/// Material uniforms for `PBSDualSidedTransparentSpecular`.
struct PbsDualSidedTransparentSpecularMaterial {
    /// Tint color (`Color`).
    _Color: vec4<f32>,
    /// Emission color (`EmissionColor`).
    _EmissionColor: vec4<f32>,
    /// Tinted specular color when `_SPECULARMAP` is disabled (RGB = f0, A = smoothness).
    _SpecularColor: vec4<f32>,
    /// Albedo `_ST` (xy = scale, zw = offset).
    _MainTex_ST: vec4<f32>,
    /// Tangent-space normal scale.
    _NormalScale: f32,
    /// Raw Froox shader variant bitmask; decoded locally via the PBSDST_KW_* constants.
    _RenderideVariantBits: u32,
}

const PBSDSTSPEC_KW_ALBEDOTEX: u32 = 1u << 0u;
const PBSDSTSPEC_KW_EMISSIONTEX: u32 = 1u << 1u;
const PBSDSTSPEC_KW_NORMALMAP: u32 = 1u << 2u;
const PBSDSTSPEC_KW_OCCLUSION: u32 = 1u << 3u;
const PBSDSTSPEC_KW_SPECULARMAP: u32 = 1u << 4u;
const PBSDSTSPEC_KW_VCOLOR_ALBEDO: u32 = 1u << 5u;
const PBSDSTSPEC_KW_VCOLOR_EMIT: u32 = 1u << 6u;
const PBSDSTSPEC_KW_VCOLOR_SPECULAR: u32 = 1u << 7u;

@group(1) @binding(0)  var<uniform> mat: PbsDualSidedTransparentSpecularMaterial;
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

fn pbsdstspec_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn kw_ALBEDOTEX() -> bool { return pbsdstspec_kw(PBSDSTSPEC_KW_ALBEDOTEX); }
fn kw_EMISSIONTEX() -> bool { return pbsdstspec_kw(PBSDSTSPEC_KW_EMISSIONTEX); }
fn kw_NORMALMAP() -> bool { return pbsdstspec_kw(PBSDSTSPEC_KW_NORMALMAP); }
fn kw_OCCLUSION() -> bool { return pbsdstspec_kw(PBSDSTSPEC_KW_OCCLUSION); }
fn kw_SPECULARMAP() -> bool { return pbsdstspec_kw(PBSDSTSPEC_KW_SPECULARMAP); }
fn kw_VCOLOR_ALBEDO() -> bool { return pbsdstspec_kw(PBSDSTSPEC_KW_VCOLOR_ALBEDO); }
fn kw_VCOLOR_EMIT() -> bool { return pbsdstspec_kw(PBSDSTSPEC_KW_VCOLOR_EMIT); }
fn kw_VCOLOR_SPECULAR() -> bool { return pbsdstspec_kw(PBSDSTSPEC_KW_VCOLOR_SPECULAR); }

/// Resolved per-fragment shading inputs for the SpecularSetup path.
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

/// Sample tangent-space normal and place it in the fragment's visible-side world frame.
fn sample_normal_world(uv_main: vec2<f32>, world_n: vec3<f32>, world_t: vec4<f32>, front_facing: bool) -> vec3<f32> {
    let tbn = pnorm::orthonormal_tbn(world_n, world_t);
    var ts_n = vec3<f32>(0.0, 0.0, 1.0);
    if (kw_NORMALMAP()) {
        ts_n = nd::decode_ts_normal_with_placeholder_sample(
            textureSample(_NormalMap, _NormalMap_sampler, uv_main),
            mat._NormalScale,
        );
    }
    if (!front_facing) {
        ts_n.z = -ts_n.z;
    }
    return normalize(tbn * ts_n);
}

/// Resolve the [`SurfaceData`] for a fragment, mirroring Unity's `surf` for `PBSDualSidedTransparentSpecular`.
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

    var spec = mat._SpecularColor;
    if (kw_SPECULARMAP()) {
        spec = textureSample(_SpecularMap, _SpecularMap_sampler, uv_main);
    }
    if (kw_VCOLOR_SPECULAR()) {
        spec = spec * vertex_color;
    }
    let f0 = clamp(spec.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let smoothness = clamp(spec.a, 0.0, 1.0);
    let roughness = clamp(1.0 - smoothness, 0.0, 1.0);
    let one_minus_reflectivity = 1.0 - max(max(f0.r, f0.g), f0.b);

    var occlusion = 1.0;
    if (kw_OCCLUSION()) {
        occlusion = textureSample(_OcclusionMap, _OcclusionMap_sampler, uv_main).r;
    }

    let emission_color = mat._EmissionColor.rgb;
    var emission = vec3<f32>(0.0);
    if (dot(emission_color, emission_color) > 1e-8) {
        emission = emission_color;
        if (kw_EMISSIONTEX()) {
            emission = emission * textureSample(_EmissionMap, _EmissionMap_sampler, uv_main).rgb;
        }
    }
    if (kw_VCOLOR_EMIT()) {
        emission = emission * vertex_color.rgb;
    }

    return SurfaceData(
        albedo.rgb,
        albedo.a,
        f0,
        roughness,
        one_minus_reflectivity,
        occlusion,
        sample_normal_world(uv_main, world_n, world_t, front_facing),
        emission,
    );
}

/// Vertex stage: forward world position, world-space normal, primary UV, and vertex color.
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
    let surface = psurf::specular(
        s.base_color,
        s.alpha,
        s.f0,
        s.roughness,
        s.occlusion,
        s.normal,
        s.emission,
    );
    return plight::shade_specular_transparent_clustered(
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
