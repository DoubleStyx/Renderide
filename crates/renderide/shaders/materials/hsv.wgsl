//! Grab-pass HSV offset/multiply filter (`Shader "Filters/HSV"`).


//#mat_default _HSVMul vec4 1.0 1.0 1.0 1.0
//#mat_default _HSVOffset vec4 0.2 0.2 0.2 0.0

#import renderide::post::filter_math as fm
#import renderide::post::filter_vertex as fv
#import renderide::post::filter_common as fc
#import renderide::material::variant_bits as vb

struct FiltersHsvMaterial {
    _Rect: vec4<f32>,
    _HSVOffset: vec4<f32>,
    _HSVMul: vec4<f32>,
    _RenderideVariantBits: u32,
    _pad0: u32,
    _pad1: vec2<u32>,
}

const HSV_KW_RECTCLIP: u32 = 1u << 0u;

@group(1) @binding(0) var<uniform> mat: FiltersHsvMaterial;

fn hsv_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn kw_RECTCLIP() -> bool {
    return hsv_kw(HSV_KW_RECTCLIP);
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
) -> fv::RectVertexOutput {
#ifdef MULTIVIEW
    return fv::rect_vertex_main(instance_index, view_idx, pos, n, t, uv0);
#else
    return fv::rect_vertex_main(instance_index, 0u, pos, n, t, uv0);
#endif
}

//#pass forward_filter
@fragment
fn fs_main(in: fv::RectVertexOutput) -> @location(0) vec4<f32> {
    fc::discard_rect_if_enabled(in.obj_xy, mat._Rect, kw_RECTCLIP());

    let c = fc::sample_scene_color_at_clip(in.clip_pos, in.view_layer);
    var hsv = fm::rgb_to_hsv_no_clip(c.rgb);
    hsv = hsv * mat._HSVMul.xyz + mat._HSVOffset.xyz;
    hsv.x = fract(hsv.x);
    hsv.y = clamp(hsv.y, 0.0, 1.0);
    let filtered = fm::hsv_to_rgb(hsv);
    return fc::retain_globals(vec4<f32>(filtered, c.a));
}
