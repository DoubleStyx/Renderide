//! Unity PBS intersect specular (`Shader "Custom/PBSIntersectSpecular"`): specular workflow with
//! intersection tint/emission parameters sampled from the scene-depth snapshot copied between the
//! opaque and intersection subpasses.
//!
//! Keyword-style float fields mirror Unity `#pragma multi_compile` values:
//! `_ALBEDOTEX`, `_EMISSIONTEX`, `_NORMALMAP`, `_SPECULARMAP`, `_OCCLUSION`.

#import renderide::math as rmath
#import renderide::mesh::vertex as mv
#import renderide::pbs::lighting as plight
#import renderide::pbs::normal as pnorm
#import renderide::pbs::sampling as psamp
#import renderide::pbs::surface as psurf
#import renderide::scene_depth_sample as sds
#import renderide::uv_utils as uvu
#import renderide::normal_decode as nd

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
    _ALBEDOTEX: f32,
    _EMISSIONTEX: f32,
    _NORMALMAP: f32,
    _SPECULARMAP: f32,
    _OCCLUSION: f32,
    _pad0: f32,
}

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

fn sample_normal_world(uv_main: vec2<f32>, world_n: vec3<f32>, world_t: vec4<f32>, front_facing: bool) -> vec3<f32> {
    var n = world_n;
    if (uvu::kw_enabled(mat._NORMALMAP)) {
        let tbn = pnorm::orthonormal_tbn(n, world_t);
        let ts_n = nd::decode_ts_normal_with_placeholder_sample(
            textureSample(_NormalMap, _NormalMap_sampler, uv_main),
            mat._NormalScale,
        );
        n = normalize(tbn * ts_n);
    }
    if (!front_facing) {
        n = -n;
    }
    return n;
}

fn intersection_lerp(frag_pos: vec4<f32>, world_pos: vec3<f32>, view_layer: u32) -> f32 {
    let diff = sds::scene_linear_depth(frag_pos, view_layer) - sds::fragment_linear_depth(world_pos, view_layer);
    if (diff < mat._EndTransitionStart) {
        return rmath::safe_linear_factor(mat._BeginTransitionStart, mat._BeginTransitionEnd, diff);
    }
    return 1.0 - rmath::safe_linear_factor(mat._EndTransitionStart, mat._EndTransitionEnd, diff);
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
    let intersect_lerp = intersection_lerp(frag_pos, world_pos, view_layer);

    var c0 = mix(mat._Color, mat._IntersectColor, intersect_lerp);
    if (uvu::kw_enabled(mat._ALBEDOTEX)) {
        c0 = c0 * textureSample(_MainTex, _MainTex_sampler, uv_main);
    }
    let base_color = c0.rgb;
    let alpha = c0.a;

    let n = sample_normal_world(uv_main, world_n, world_t, front_facing);

    var occlusion = 1.0;
    if (uvu::kw_enabled(mat._OCCLUSION)) {
        occlusion = textureSample(_OcclusionMap, _OcclusionMap_sampler, uv_main).r;
    }

    var spec_sample = mat._SpecularColor;
    if (uvu::kw_enabled(mat._SPECULARMAP)) {
        spec_sample = textureSample(_SpecularMap, _SpecularMap_sampler, uv_main);
    }
    let f0 = spec_sample.rgb;
    let smoothness = clamp(spec_sample.a, 0.0, 1.0);
    let roughness = psamp::roughness_from_smoothness(smoothness);

    var emission = mat._EmissionColor.xyz;
    if (uvu::kw_enabled(mat._EMISSIONTEX)) {
        emission = emission * textureSample(_EmissionMap, _EmissionMap_sampler, uv_main).rgb;
    }
    emission = emission + mat._IntersectEmissionColor.rgb * intersect_lerp;

    let surface = psurf::specular(base_color, alpha, f0, roughness, occlusion, n, emission);
    let color = plight::shade_specular_clustered(
        frag_pos.xy,
        world_pos,
        view_layer,
        surface,
        plight::default_lighting_options(),
    );
    return vec4<f32>(color, alpha);
}
