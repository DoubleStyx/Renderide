//! Grab-pass grayscale filter (`Shader "Filters/Grayscale"`).


#import renderide::filter_vertex as fv
#import renderide::globals as rg
#import renderide::grab_pass as gp
#import renderide::ui::rect_clip as uirc
#import renderide::uv_utils as uvu

struct FiltersGrayscaleMaterial {
    _Rect: vec4<f32>,
    _RatioR: f32,
    _RatioG: f32,
    _RatioB: f32,
    _Lerp: f32,
    GRADIENT: f32,
    _RectClip: f32,
    _pad0: vec2<f32>,
}

@group(1) @binding(0) var<uniform> mat: FiltersGrayscaleMaterial;
@group(1) @binding(1) var _Gradient: texture_2d<f32>;
@group(1) @binding(2) var _Gradient_sampler: sampler;

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
) -> fv::RectVertexOutput {
#ifdef MULTIVIEW
    return fv::rect_vertex_main(instance_index, view_idx, pos, n, t, uv0);
#else
    return fv::rect_vertex_main(instance_index, 0u, pos, n, t, uv0);
#endif
}

//#pass forward
@fragment
fn fs_main(in: fv::RectVertexOutput) -> @location(0) vec4<f32> {
    if (uirc::should_clip_rect(in.obj_xy, mat._Rect, mat._RectClip)) {
        discard;
    }

    let c = gp::sample_scene_color(gp::frag_screen_uv(in.clip_pos), in.view_layer);
    let grayscale = dot(c.rgb, vec3<f32>(mat._RatioR, mat._RatioG, mat._RatioB));
    var new_color = vec3<f32>(grayscale);
    if (uvu::kw_enabled(mat.GRADIENT)) {
        new_color = textureSampleLevel(_Gradient, _Gradient_sampler, vec2<f32>(grayscale, 0.0), 0.0).rgb;
    }
    let filtered = mix(c.rgb, new_color, mat._Lerp);
    return rg::retain_globals_additive(vec4<f32>(filtered, c.a));
}
