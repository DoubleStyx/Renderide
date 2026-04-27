//! Per-object grab-pass 3D LUT filter (`Shader "Filters/LUT_PerObject"`).

// unity-shader-name: Filters/LUT_PerObject

#import renderide::filter_vertex as fv
#import renderide::globals as rg
#import renderide::grab_pass as gp
#import renderide::texture_sampling as ts

struct FiltersLutPerObjectMaterial {
    _Lerp: f32,
    _GrabPass: f32,
    _LUT_LodBias: f32,
    _SecondaryLUT_LodBias: f32,
}

@group(1) @binding(0) var<uniform> mat: FiltersLutPerObjectMaterial;
@group(1) @binding(1) var _LUT: texture_3d<f32>;
@group(1) @binding(2) var _LUT_sampler: sampler;
@group(1) @binding(3) var _SecondaryLUT: texture_3d<f32>;
@group(1) @binding(4) var _SecondaryLUT_sampler: sampler;

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

//#pass forward
@fragment
fn fs_main(
    @builtin(position) frag_pos: vec4<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let c = gp::sample_scene_color(gp::frag_screen_uv(frag_pos), view_layer);
    let coords = clamp(c.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let primary = ts::sample_tex_3d(_LUT, _LUT_sampler, coords, mat._LUT_LodBias).rgb;
    let secondary = ts::sample_tex_3d(_SecondaryLUT, _SecondaryLUT_sampler, coords, mat._SecondaryLUT_LodBias).rgb;
    let filtered = mix(primary, secondary, clamp(mat._Lerp, 0.0, 1.0));
    return rg::retain_globals_additive(vec4<f32>(filtered, c.a));
}
