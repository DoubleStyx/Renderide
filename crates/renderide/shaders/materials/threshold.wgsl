//! Grab-pass threshold filter (`Shader "Filters/Threshold"`).
//!
//! Reads the scene color, remaps `[_Threshold - _Transition/2, _Threshold + _Transition/2]`
//! to `[0, 1]`, and clamps the result. The `RECTCLIP` variant bit
//! (Unity `#pragma multi_compile _ RECTCLIP`) decides whether `_Rect` discards
//! fragments outside the object-space rectangle.


//#mat_default _Threshold float 0.5
//#mat_default _Transition float 0.01

#import renderide::post::filter_common as fc
#import renderide::mesh::vertex as mv
#import renderide::draw::per_draw as pd
#import renderide::material::variant_bits as vb

struct FiltersThresholdMaterial {
    _Threshold: f32,
    _Transition: f32,
    _Rect: vec4<f32>,
    _RenderideVariantBits: u32,
}

const THRESHOLD_KW_RECTCLIP: u32 = 1u << 0u;

@group(1) @binding(0) var<uniform> mat: FiltersThresholdMaterial;

fn threshold_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) obj_xy: vec2<f32>,
    @location(1) @interpolate(flat) view_layer: u32,
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
    let world_p = mv::world_position(d, pos);
#ifdef MULTIVIEW
    let vp = mv::select_view_proj(d, view_idx);
    let layer = view_idx;
#else
    let vp = mv::select_view_proj(d, 0u);
    let layer = 0u;
#endif

    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.obj_xy = pos.xy;
    out.view_layer = layer;
    return out;
}

//#pass forward_filter
@fragment
fn fs_main(
    @builtin(position) frag_pos: vec4<f32>,
    @location(0) obj_xy: vec2<f32>,
    @location(1) @interpolate(flat) view_layer: u32,
) -> @location(0) vec4<f32> {
    fc::discard_rect_if_enabled(obj_xy, mat._Rect, threshold_kw(THRESHOLD_KW_RECTCLIP));

    let c = fc::sample_scene_color_at_clip(frag_pos, view_layer);
    let transition = max(abs(mat._Transition), 1e-6);
    let filtered = clamp(((c.rgb - vec3<f32>(mat._Threshold)) / transition) + vec3<f32>(mat._Transition * 0.5), vec3<f32>(0.0), vec3<f32>(1.0));
    return fc::retain_globals(vec4<f32>(filtered, c.a));
}
