//! Grab-pass HSV offset/multiply filter (`Shader "Filters/HSV"`).


#import renderide::filter_math as fm
#import renderide::filter_vertex as fv
#import renderide::globals as rg
#import renderide::grab_pass as gp
#import renderide::ui::rect_clip as uirc

struct FiltersHsvMaterial {
    _Rect: vec4<f32>,
    _HSVOffset: vec4<f32>,
    _HSVMul: vec4<f32>,
    _RectClip: f32,
    _pad0: f32,
    _pad1: vec2<f32>,
}

@group(1) @binding(0) var<uniform> mat: FiltersHsvMaterial;

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
) -> fv::RectVertexOutput {
#ifdef MULTIVIEW
    return fv::rect_vertex_main(instance_index, view_idx, pos, n, uv0);
#else
    return fv::rect_vertex_main(instance_index, 0u, pos, n, uv0);
#endif
}

//#pass forward
@fragment
fn fs_main(in: fv::RectVertexOutput) -> @location(0) vec4<f32> {
    if (uirc::should_clip_rect(in.obj_xy, mat._Rect, mat._RectClip)) {
        discard;
    }

    let c = gp::sample_scene_color(gp::frag_screen_uv(in.clip_pos), in.view_layer);
    var hsv = fm::rgb_to_hsv_no_clip(c.rgb);
    hsv = hsv * mat._HSVMul.xyz + mat._HSVOffset.xyz;
    hsv.x = fract(hsv.x);
    hsv.y = clamp(hsv.y, 0.0, 1.0);
    let filtered = fm::hsv_to_rgb(hsv);
    return rg::retain_globals_additive(vec4<f32>(filtered, c.a));
}
