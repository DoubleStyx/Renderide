//! Unity PBS intersect specular (`Shader "Custom/PBSIntersectSpecular"`): specular workflow with
//! intersection tint/emission parameters sampled from the scene-depth snapshot copied after the
//! opaque forward pass.
//!
//! Froox variant bits populate `_RenderideVariantBits`; this shader decodes PBSIntersectSpecular's
//! shader-specific keyword bits locally.

//#texture_default _MainTex white
//#texture_default _NormalMap bump
//#texture_default _EmissionMap black
//#texture_default _OcclusionMap white
//#texture_default _SpecularMap white
//#mat_default _Color vec4 1.0 1.0 1.0 1.0
//#mat_default _EndTransitionEnd float 0.1
//#mat_default _EndTransitionStart float 0.1
//#mat_default _IntersectColor vec4 1.0 1.0 1.0 1.0
//#mat_default _IntersectEmissionColor vec4 1.0 0.0 0.0 1.0
//#mat_default _NormalScale float 1.0
//#mat_default _SpecularColor vec4 1.0 1.0 1.0 0.5

#import renderide::material::variant_bits as vb
#import renderide::mesh::vertex as mv
#import renderide::pbs::families::intersect as pint
#import renderide::pbs::lighting as plight
#import renderide::pbs::sampling as psamp
#import renderide::pbs::surface as psurf
#import renderide::core::uv as uvu

struct PbsIntersectSpecularMaterial {
    _Color: vec4<f32>,
    _IntersectColor: vec4<f32>,
    _IntersectEmissionColor: vec4<f32>,
    _SpecularColor: vec4<f32>,
    _EmissionColor: vec4<f32>,
    _MainTex_ST: vec4<f32>,
    _BeginTransitionStart: f32,
    _BeginTransitionEnd: f32,
    _EndTransitionStart: f32,
    _EndTransitionEnd: f32,
    _NormalScale: f32,
    _RenderideVariantBits: u32,
}

const PBSINTERSECTSPECULAR_KW_ALBEDOTEX: u32 = 1u << 0u;
const PBSINTERSECTSPECULAR_KW_EMISSIONTEX: u32 = 1u << 1u;
const PBSINTERSECTSPECULAR_KW_NORMALMAP: u32 = 1u << 2u;
const PBSINTERSECTSPECULAR_KW_OCCLUSION: u32 = 1u << 3u;
const PBSINTERSECTSPECULAR_KW_SPECULARMAP: u32 = 1u << 4u;

@group(1) @binding(0)  var<uniform> mat: PbsIntersectSpecularMaterial;
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

fn pbs_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
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

//#pass type=forward name=forward_transparent blend=transparent_material zwrite=material(off) cull=material(off) color_mask=material(rgba) offset=material(0,0)
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
    let intersect_lerp = pint::intersection_lerp(
        frag_pos,
        world_pos,
        view_layer,
        mat._BeginTransitionStart,
        mat._BeginTransitionEnd,
        mat._EndTransitionStart,
        mat._EndTransitionEnd,
    );

    var c0 = mix(mat._Color, mat._IntersectColor, intersect_lerp);
    if (pbs_kw(PBSINTERSECTSPECULAR_KW_ALBEDOTEX)) {
        c0 = c0 * textureSample(_MainTex, _MainTex_sampler, uv_main);
    }
    let base_color = c0.rgb;
    let alpha = c0.a;

    let n = psamp::sample_optional_two_sided_world_normal(
        pbs_kw(PBSINTERSECTSPECULAR_KW_NORMALMAP),
        _NormalMap,
        _NormalMap_sampler,
        uv_main,
        0.0,
        mat._NormalScale,
        world_n,
        world_t,
        front_facing,
    );

    var occlusion = 1.0;
    if (pbs_kw(PBSINTERSECTSPECULAR_KW_OCCLUSION)) {
        occlusion = textureSample(_OcclusionMap, _OcclusionMap_sampler, uv_main).r;
    }

    var spec_sample = mat._SpecularColor;
    if (pbs_kw(PBSINTERSECTSPECULAR_KW_SPECULARMAP)) {
        spec_sample = textureSample(_SpecularMap, _SpecularMap_sampler, uv_main);
    }
    let f0 = spec_sample.rgb;
    let smoothness = clamp(spec_sample.a, 0.0, 1.0);
    let roughness = psamp::roughness_from_smoothness(smoothness);

    var emission = mat._EmissionColor.xyz;
    if (pbs_kw(PBSINTERSECTSPECULAR_KW_EMISSIONTEX)) {
        emission = emission * textureSample(_EmissionMap, _EmissionMap_sampler, uv_main).rgb;
    }
    emission = emission + mat._IntersectEmissionColor.rgb * intersect_lerp;

    let surface = psurf::specular_with_geometric_normal(
        base_color,
        alpha,
        f0,
        roughness,
        occlusion,
        n,
        psamp::two_sided_geometric_normal(world_n, front_facing),
        emission,
    );
    return plight::shade_specular_transparent_clustered(
        frag_pos.xy,
        world_pos,
        view_layer,
        surface,
        plight::default_lighting_options(),
    );
}
