//! Grab-pass grayscale filter (`Shader "Filters/Grayscale"`).
//!
//! Froox variant bits populate `_RenderideVariantBits`; this shader decodes Grayscale's
//! shader-specific keyword bits locally.

//#texture_default _Gradient black
//#mat_default _Lerp float 1.0
//#mat_default _RatioB float 0.11
//#mat_default _RatioG float 0.59
//#mat_default _RatioR float 0.3

#import renderide::post::filter_vertex as fv
#import renderide::post::filter_common as fc
#import renderide::material::variant_bits as vb

struct FiltersGrayscaleMaterial {
    _Rect: vec4<f32>,
    _RatioR: f32,
    _RatioG: f32,
    _RatioB: f32,
    _Lerp: f32,
    _RenderideVariantBits: u32,
    _pad0: u32,
    _pad1: vec2<u32>,
}

const GRAYSCALE_KW_GRADIENT: u32 = 1u << 0u;
const GRAYSCALE_KW_RECTCLIP: u32 = 1u << 1u;

@group(1) @binding(0) var<uniform> mat: FiltersGrayscaleMaterial;
@group(1) @binding(1) var _Gradient: texture_2d<f32>;
@group(1) @binding(2) var _Gradient_sampler: sampler;

fn grayscale_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn kw_GRADIENT() -> bool {
    return grayscale_kw(GRAYSCALE_KW_GRADIENT);
}

fn kw_RECTCLIP() -> bool {
    return grayscale_kw(GRAYSCALE_KW_RECTCLIP);
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
    let grayscale = dot(c.rgb, vec3<f32>(mat._RatioR, mat._RatioG, mat._RatioB));
    var new_color = vec3<f32>(grayscale);
    if (kw_GRADIENT()) {
        new_color = textureSampleLevel(_Gradient, _Gradient_sampler, vec2<f32>(grayscale, 0.0), 0.0).rgb;
    }
    let filtered = mix(c.rgb, new_color, mat._Lerp);
    return fc::retain_globals(vec4<f32>(filtered, c.a));
}
