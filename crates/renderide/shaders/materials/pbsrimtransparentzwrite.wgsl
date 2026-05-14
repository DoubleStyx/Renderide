//! Unity PBS rim transparent with ZWrite (`Shader "PBSRimTransparentZWrite"`):
//! same shading as [`pbsrimtransparent`], but emits a depth-only prepass before the alpha-blended
//! forward pass so the surface populates the depth buffer (matches Unity's `Pass { ColorMask 0 }`
//! prepass + `#pragma surface surf Standard alpha fullforwardshadows` color pass).
//!
//! Froox variant bits populate `_RenderideVariantBits`; PBSRimTransparentZWrite's keywords (sorted
//! alphabetically) occupy bits 0-5. `_ZWRITE` is pipeline-affecting only, so it reserves bit 5
//! but gets no shader-local constant.


//#texture_default _MainTex white
//#texture_default _NormalMap bump
//#texture_default _EmissionMap black
//#texture_default _OcclusionMap white
//#texture_default _MetallicMap black

#import renderide::frame::globals as rg
#import renderide::material::fresnel as mf
#import renderide::material::variant_bits as vb
#import renderide::mesh::vertex as mv
#import renderide::pbs::lighting as plight
#import renderide::pbs::sampling as psamp
#import renderide::pbs::surface as psurf
#import renderide::core::uv as uvu

struct PbsRimTransparentZWriteMaterial {
    _Color: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _RimColor: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _Glossiness: f32,
    _Metallic: f32,
    _NormalScale: f32,
    _RimPower: f32,
    _RenderideVariantBits: u32,
}

const PBSRIMTRANSPARENTZWRITE_KW_ALBEDOTEX: u32 = 1u << 0u;
const PBSRIMTRANSPARENTZWRITE_KW_EMISSIONTEX: u32 = 1u << 1u;
const PBSRIMTRANSPARENTZWRITE_KW_METALLICMAP: u32 = 1u << 2u;
const PBSRIMTRANSPARENTZWRITE_KW_NORMALMAP: u32 = 1u << 3u;
const PBSRIMTRANSPARENTZWRITE_KW_OCCLUSION: u32 = 1u << 4u;
const PBSRIMTRANSPARENTZWRITE_KW_ZWRITE: u32 = 1u << 5u;

@group(1) @binding(0)  var<uniform> mat: PbsRimTransparentZWriteMaterial;
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

fn pbs_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn sample_normal_world(uv_main: vec2<f32>, world_n: vec3<f32>, world_t: vec4<f32>) -> vec3<f32> {
    return psamp::sample_optional_world_normal(
        pbs_kw(PBSRIMTRANSPARENTZWRITE_KW_NORMALMAP),
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

/// Depth-only prepass: writes nothing to color (`write=none`) but populates depth so the alpha-blended
/// main pass below can self-occlude. Touches every binding so the prepass pipeline's auto-derived
/// bind-group layout matches the forward pass and the same material bind group binds for both.
//#pass depth_prepass
@fragment
fn fs_depth_only(
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) world_t: vec4<f32>,
    @location(3) uv0: vec2<f32>,
    @location(4) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let uv_main = uvu::apply_st(uv0, mat._MainTex_ST);
    let albedo_s = textureSample(_MainTex, _MainTex_sampler, uv_main);
    let normal_s = textureSample(_NormalMap, _NormalMap_sampler, uv_main);
    let emit_s = textureSample(_EmissionMap, _EmissionMap_sampler, uv_main);
    let occ_s = textureSample(_OcclusionMap, _OcclusionMap_sampler, uv_main);
    let metal_s = textureSample(_MetallicMap, _MetallicMap_sampler, uv_main);
    let touch = (mat._Color.x + mat._EmissionColor.x + mat._RimColor.x
        + mat._Glossiness + mat._Metallic + mat._NormalScale + mat._RimPower
        + f32(mat._RenderideVariantBits)
        + albedo_s.x + normal_s.x + emit_s.x + occ_s.x + metal_s.x
        + world_pos.x + world_n.x + world_t.x + f32(view_layer)) * 0.0;
    return rg::retain_globals_additive(vec4<f32>(touch, touch, touch, 0.0));
}

//#pass forward_transparent
@fragment
fn fs_main(
    @builtin(position) frag_pos: vec4<f32>,
    @builtin(front_facing) front_facing: bool,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_n: vec3<f32>,
    @location(2) world_t: vec4<f32>,
    @location(3) uv0: vec2<f32>,
    @location(4) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let uv_main = uvu::apply_st(uv0, mat._MainTex_ST);

    var c0 = mat._Color;
    if (pbs_kw(PBSRIMTRANSPARENTZWRITE_KW_ALBEDOTEX)) {
        c0 = c0 * textureSample(_MainTex, _MainTex_sampler, uv_main);
    }
    let base_color = c0.rgb;
    let alpha = c0.a;

    var n = sample_normal_world(uv_main, world_n, world_t);
    if (!front_facing) {
        n = -n;
    }

    var occlusion = 1.0;
    if (pbs_kw(PBSRIMTRANSPARENTZWRITE_KW_OCCLUSION)) {
        occlusion = textureSample(_OcclusionMap, _OcclusionMap_sampler, uv_main).r;
    }

    var metallic = mat._Metallic;
    var smoothness = mat._Glossiness;
    if (pbs_kw(PBSRIMTRANSPARENTZWRITE_KW_METALLICMAP)) {
        let m = textureSample(_MetallicMap, _MetallicMap_sampler, uv_main);
        metallic = m.r;
        smoothness = m.a;
    }
    metallic = clamp(metallic, 0.0, 1.0);
    smoothness = clamp(smoothness, 0.0, 1.0);
    let roughness = psamp::roughness_from_smoothness(smoothness);

    var emission = mat._EmissionColor.rgb;
    if (pbs_kw(PBSRIMTRANSPARENTZWRITE_KW_EMISSIONTEX)) {
        emission = emission * textureSample(_EmissionMap, _EmissionMap_sampler, uv_main).rgb;
    }

    let view_dir = rg::view_dir_for_world_pos(world_pos, view_layer);
    let rim = mf::rim_factor(n, view_dir, mat._RimPower);
    let rim_emission = mat._RimColor.rgb * rim;
    let surface = psurf::metallic(base_color, alpha, metallic, roughness, occlusion, n, emission + rim_emission);
    return plight::shade_metallic_transparent_clustered(
        frag_pos.xy,
        world_pos,
        view_layer,
        surface,
        plight::default_lighting_options(),
    );
}
