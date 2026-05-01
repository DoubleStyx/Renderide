//! Grab-pass pixelation filter (`Shader "Filters/Pixelate"`).


#import renderide::filter_math as fm
#import renderide::filter_vertex as fv
#import renderide::globals as rg
#import renderide::grab_pass as gp
#import renderide::uv_utils as uvu

struct FiltersPixelateMaterial {
    _Resolution: vec4<f32>,
    _ResolutionTex_ST: vec4<f32>,
}

@group(1) @binding(0) var<uniform> mat: FiltersPixelateMaterial;
@group(1) @binding(1) var _ResolutionTex: texture_2d<f32>;
@group(1) @binding(2) var _ResolutionTex_sampler: sampler;

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
) -> fv::VertexOutput {
#ifdef MULTIVIEW
    return fv::vertex_main(instance_index, view_idx, pos, n, t, uv0);
#else
    return fv::vertex_main(instance_index, 0u, pos, n, t, uv0);
#endif
}

//#pass forward
@fragment
fn fs_main(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) uv0: vec2<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    let texel_scale = textureSample(_ResolutionTex, _ResolutionTex_sampler, uvu::apply_st(uv0, mat._ResolutionTex_ST)).rg;
    let resolution = max(mat._Resolution.xy * texel_scale, vec2<f32>(1.0));
    let uv = fm::safe_div_vec2(round(gp::frag_screen_uv(frag_pos) * resolution), resolution);
    return rg::retain_globals_additive(gp::sample_scene_color(uv, view_layer));
}
