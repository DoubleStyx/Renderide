//! Per-object grab-pass refraction filter (`Shader "Filters/Refract_PerObject"`).

// unity-shader-name: Filters/Refract_PerObject

#import renderide::filter_math as fm
#import renderide::filter_vertex as fv
#import renderide::globals as rg
#import renderide::grab_pass as gp
#import renderide::normal_decode as nd
#import renderide::scene_depth_sample as sds
#import renderide::uv_utils as uvu

struct FiltersRefractPerObjectMaterial {
    _NormalMap_ST: vec4<f32>,
    _RefractionStrength: f32,
    _DepthBias: f32,
    _DepthDivisor: f32,
    _GrabPass: f32,
    _NORMALMAP: f32,
    _pad0: vec3<f32>,
}

@group(1) @binding(0) var<uniform> mat: FiltersRefractPerObjectMaterial;
@group(1) @binding(1) var _NormalMap: texture_2d<f32>;
@group(1) @binding(2) var _NormalMap_sampler: sampler;

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> fv::VertexOutput {
#ifdef MULTIVIEW
    return fv::vertex_main(instance_index, view_idx, pos, n, uv0);
#else
    return fv::vertex_main(instance_index, 0u, pos, n, uv0);
#endif
}

fn refract_offset(uv0: vec2<f32>, world_n: vec3<f32>) -> vec2<f32> {
    var n = normalize(world_n);
    if (uvu::kw_enabled(mat._NORMALMAP)) {
        let ts = nd::decode_ts_normal_with_placeholder_sample(
            textureSample(_NormalMap, _NormalMap_sampler, uvu::apply_st(uv0, mat._NormalMap_ST)),
            1.0,
        );
        n = normalize(vec3<f32>(n.xy + ts.xy, n.z));
    }
    return n.xy * mat._RefractionStrength;
}

//#pass forward
@fragment
fn fs_main(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) uv0: vec2<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) world_n: vec3<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let screen_uv = gp::frag_screen_uv(frag_pos);
    let fade = sds::depth_fade(frag_pos, world_pos, view_layer, max(mat._DepthDivisor, 1.0));
    let offset = refract_offset(uv0, world_n) * fade * fm::screen_vignette(screen_uv);
    let color = gp::sample_scene_color(screen_uv - offset, view_layer);
    return rg::retain_globals_additive(color);
}
