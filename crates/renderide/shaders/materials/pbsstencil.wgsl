//! Unity surface shader `Shader "PBSStencil"`: metallic Standard lighting that runs through the
//! standard forward path while the host applies stencil ops driven by `_Stencil`, `_StencilComp`,
//! `_StencilOp`, `_StencilReadMask`, `_StencilWriteMask`, and `_ColorMask` material properties.
//!
//! Stencil state is CPU-side (it lives in the wgpu pipeline descriptor, not a shader binding), so
//! the WGSL itself is effectively `pbsmetallic` with the keyword surface scoped down to this
//! material's smaller property block. It shares `pbsdualsided.wgsl` shading without the
//! front-face flip.
//!
//! Froox variant bits populate `_RenderideVariantBits`; this shader decodes PBSStencil's
//! shader-specific keyword bits locally.


//#texture_default _MainTex white
//#texture_default _NormalMap bump
//#texture_default _EmissionMap black
//#texture_default _OcclusionMap white
//#texture_default _MetallicMap black
//#mat_default _Color vec4 1.0 1.0 1.0 1.0
//#mat_default _NormalScale float 1.0

#import renderide::material::variant_bits as vb
#import renderide::mesh::vertex as mv
#import renderide::pbs::lighting as plight
#import renderide::pbs::sampling as psamp
#import renderide::pbs::surface as psurf
#import renderide::core::uv as uvu

struct PbsStencilMaterial {
    _Color: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _NormalScale: f32,
    _Glossiness: f32,
    _Metallic: f32,
    _RenderideVariantBits: u32,
}

const PBSSTENCIL_KW_ALBEDOTEX: u32 = 1u << 0u;
const PBSSTENCIL_KW_EMISSIONTEX: u32 = 1u << 1u;
const PBSSTENCIL_KW_METALLICMAP: u32 = 1u << 2u;
const PBSSTENCIL_KW_NORMALMAP: u32 = 1u << 3u;
const PBSSTENCIL_KW_OCCLUSION: u32 = 1u << 4u;

@group(1) @binding(0)  var<uniform> mat: PbsStencilMaterial;
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

fn pbsstencil_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn kw_ALBEDOTEX() -> bool {
    return pbsstencil_kw(PBSSTENCIL_KW_ALBEDOTEX);
}

fn kw_EMISSIONTEX() -> bool {
    return pbsstencil_kw(PBSSTENCIL_KW_EMISSIONTEX);
}

fn kw_METALLICMAP() -> bool {
    return pbsstencil_kw(PBSSTENCIL_KW_METALLICMAP);
}

fn kw_NORMALMAP() -> bool {
    return pbsstencil_kw(PBSSTENCIL_KW_NORMALMAP);
}

fn kw_OCCLUSION() -> bool {
    return pbsstencil_kw(PBSSTENCIL_KW_OCCLUSION);
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
    if (kw_ALBEDOTEX()) {
        c = c * textureSample(_MainTex, _MainTex_sampler, uv_main);
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

    let base_color = c.rgb;
    let n = sample_normal_world(uv_main, world_n, world_t);
    let surface = psurf::metallic_with_geometric_normal(base_color, c.a, metallic, roughness, occlusion, n, world_n, emission);
    let options = plight::ClusterLightingOptions(include_directional, include_local, true, true);
    return vec4<f32>(
        plight::shade_metallic_clustered(frag_xy, world_pos, view_layer, surface, options),
        c.a,
    );
}

//#pass forward
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
