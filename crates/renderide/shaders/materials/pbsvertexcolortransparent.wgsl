//! Unity surface shader `Shader "PBSVertexColorTransparent"`: metallic Standard lighting that
//! optionally multiplies albedo, emission, or metallic/smoothness by the mesh vertex color.
//!
//! Mirrors [`pbsdualsided`](super::pbsdualsided) for the keyword + VCOLOR_* surface, dropping the
//! front-face flip (this shader is single-sided in Unity). Transparent alpha blending is used with
//! Unity's authored `Cull Back`.
//!
//! Froox variant bits populate `_RenderideVariantBits`; this shader decodes
//! PBSVertexColorTransparent's shader-specific keyword bits locally.

//#texture_default _MainTex white
//#texture_default _NormalMap bump
//#texture_default _EmissionMap black
//#texture_default _OcclusionMap white
//#texture_default _MetallicMap black
//#mat_default _Color vec4 1.0 1.0 1.0 1.0
//#mat_default _NormalScale float 1.0

#import renderide::material::variant_bits as vb
#import renderide::core::math as rmath
#import renderide::mesh::vertex as mv
#import renderide::pbs::lighting as plight
#import renderide::pbs::sampling as psamp
#import renderide::pbs::surface as psurf
#import renderide::core::uv as uvu

struct PbsVertexColorTransparentMaterial {
    _Color: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _NormalScale: f32,
    _Glossiness: f32,
    _Metallic: f32,
    _AlphaClip: f32,
    _RenderideVariantBits: u32,
}

const PBSVCT_KW_ALBEDOTEX: u32 = 1u << 0u;
const PBSVCT_KW_ALPHACLIP: u32 = 1u << 1u;
const PBSVCT_KW_EMISSIONTEX: u32 = 1u << 2u;
const PBSVCT_KW_METALLICMAP: u32 = 1u << 3u;
const PBSVCT_KW_NORMALMAP: u32 = 1u << 4u;
const PBSVCT_KW_OCCLUSION: u32 = 1u << 5u;
const PBSVCT_KW_VCOLOR_ALBEDO: u32 = 1u << 6u;
const PBSVCT_KW_VCOLOR_EMIT: u32 = 1u << 7u;
const PBSVCT_KW_VCOLOR_METALLIC: u32 = 1u << 8u;

@group(1) @binding(0)  var<uniform> mat: PbsVertexColorTransparentMaterial;
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

fn pbsvct_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn kw_ALBEDOTEX() -> bool {
    return pbsvct_kw(PBSVCT_KW_ALBEDOTEX);
}

fn kw_ALPHACLIP() -> bool {
    return pbsvct_kw(PBSVCT_KW_ALPHACLIP);
}

fn kw_EMISSIONTEX() -> bool {
    return pbsvct_kw(PBSVCT_KW_EMISSIONTEX);
}

fn kw_METALLICMAP() -> bool {
    return pbsvct_kw(PBSVCT_KW_METALLICMAP);
}

fn kw_NORMALMAP() -> bool {
    return pbsvct_kw(PBSVCT_KW_NORMALMAP);
}

fn kw_OCCLUSION() -> bool {
    return pbsvct_kw(PBSVCT_KW_OCCLUSION);
}

fn kw_VCOLOR_ALBEDO() -> bool {
    return pbsvct_kw(PBSVCT_KW_VCOLOR_ALBEDO);
}

fn kw_VCOLOR_EMIT() -> bool {
    return pbsvct_kw(PBSVCT_KW_VCOLOR_EMIT);
}

fn kw_VCOLOR_METALLIC() -> bool {
    return pbsvct_kw(PBSVCT_KW_VCOLOR_METALLIC);
}

struct SurfaceData {
    base_color: vec3<f32>,
    alpha: f32,
    metallic: f32,
    roughness: f32,
    occlusion: f32,
    normal: vec3<f32>,
    emission: vec3<f32>,
}

fn sample_normal_world(uv_main: vec2<f32>, world_n: vec3<f32>, world_t: vec4<f32>) -> vec3<f32> {
    return psamp::sample_optional_world_normal(
        kw_NORMALMAP(),
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

    var metallic = mat._Metallic;
    var smoothness = mat._Glossiness;
    if (kw_METALLICMAP()) {
        let m = textureSample(_MetallicMap, _MetallicMap_sampler, uv_main);
        metallic = m.r;
        smoothness = m.a;
    }
    if (kw_VCOLOR_METALLIC()) {
        metallic = metallic * rmath::luminance_rgb(vertex_color.rgb);
        smoothness = smoothness * vertex_color.a;
    }
    metallic = clamp(metallic, 0.0, 1.0);
    let roughness = psamp::roughness_from_smoothness(smoothness);

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
        metallic,
        roughness,
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

//#pass forward_transparent_cull_back
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
    let surface = psurf::metallic_with_geometric_normal(
        s.base_color,
        s.alpha,
        s.metallic,
        s.roughness,
        s.occlusion,
        s.normal,
        world_n,
        s.emission,
    );
    return plight::shade_metallic_transparent_clustered(frag_pos.xy, world_pos, view_layer, surface, plight::default_lighting_options());
}
