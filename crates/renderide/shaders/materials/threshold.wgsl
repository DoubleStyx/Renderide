//! Grab-pass threshold filter (`Shader "Filters/Threshold"`).
//!
//! Reads the scene color, remaps `[_Threshold - _Transition/2, _Threshold + _Transition/2]`
//! to `[0, 1]`, and clamps the result. The `RECTCLIP` variant bit
//! (Unity `#pragma multi_compile _ RECTCLIP`) decides whether `_Rect` discards
//! fragments outside the object-space rectangle.


//#render_queue Transparent+500
//#mat_default _Threshold float 0.5
//#mat_default _Transition float 0.01

#import renderide::post::filter_common as fc
#import renderide::post::filter_vertex as fv
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

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
) -> fv::PositionRectVertexOutput {
#ifdef MULTIVIEW
    return fv::position_rect_vertex_main(instance_index, view_idx, pos);
#else
    return fv::position_rect_vertex_main(instance_index, 0u, pos);
#endif
}

//#pass type=forward name=forward_filter blend=material_filter
@fragment
fn fs_main(in: fv::PositionRectVertexOutput) -> @location(0) vec4<f32> {
    let c = fc::sample_clipped_scene_color_at_clip(
        in.obj_xy,
        mat._Rect,
        threshold_kw(THRESHOLD_KW_RECTCLIP),
        in.clip_pos,
        in.view_layer,
    );
    let transition = select(
        mat._Transition,
        select(-1e-6, 1e-6, mat._Transition >= 0.0),
        abs(mat._Transition) < 1e-6,
    );
    let filtered = clamp(((c.rgb - vec3<f32>(mat._Threshold)) / transition) + vec3<f32>(mat._Transition * 0.5), vec3<f32>(0.0), vec3<f32>(1.0));
    return fc::retain_scene_alpha(c, filtered);
}
