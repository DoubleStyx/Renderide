//! Scene-depth visualization filter (`Shader "Filters/Get Depth"`).
//!
//! Samples the renderer-produced scene-depth snapshot at the fragment position,
//! optionally rescales by `(depth - _ClipMin) / (_ClipMax - _ClipMin)`, applies
//! `_Multiply * depth + _Offset`, saturates, and writes RGB grayscale.
//!
//! **Rect clip (Unity `RECTCLIP` keyword):** When `_RectClip > 0.5` and `_Rect`
//! has non-zero area, fragments outside the rect in object XY are discarded.
//! Missing `_RectClip` defaults to off.
//!
//! **Clip range (Unity `CLIP` keyword):** Missing `CLIP` defaults to off, matching
//! the host's default even when `_ClipMax` is present.


#import renderide::filter_vertex as fv
#import renderide::globals as rg
#import renderide::scene_depth_sample as sds
#import renderide::ui::rect_clip as uirc
#import renderide::uv_utils as uvu

struct FiltersGetDepthMaterial {
    _Rect: vec4<f32>,
    _Multiply: f32,
    _Offset: f32,
    _ClipMin: f32,
    _ClipMax: f32,
    _RectClip: f32,
    CLIP: f32,
    _pad0: vec2<f32>,
}

@group(1) @binding(0) var<uniform> mat: FiltersGetDepthMaterial;

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
fn fs_main(vout: fv::RectVertexOutput) -> @location(0) vec4<f32> {
    if (uirc::should_clip_rect(vout.obj_xy, mat._Rect, mat._RectClip)) {
        discard;
    }

    var depth = sds::scene_linear_depth(vout.clip_pos, vout.view_layer);
    if (uvu::kw_enabled(mat.CLIP)) {
        let range = max(mat._ClipMax - mat._ClipMin, 1e-6);
        depth = (depth - mat._ClipMin) / range;
    }
    depth = depth * mat._Multiply + mat._Offset;
    depth = clamp(depth, 0.0, 1.0);

    return rg::retain_globals_additive(vec4<f32>(vec3<f32>(depth), 1.0));
}
