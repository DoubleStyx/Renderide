//! Scene-depth visualization filter (`Shader "Filters/Get Depth"`).
//!
//! Samples the renderer-produced scene-depth snapshot at the fragment position,
//! rescales by `(depth - _ClipMin) / (_ClipMax - _ClipMin)`, applies
//! `_Multiply * depth + _Offset`, saturates, and writes RGB grayscale.
//!
//! **Rect clip (Unity `RECTCLIP` keyword):** When `_RectClip > 0.5` and `_Rect`
//! has non-zero area, fragments outside the rect in object XY are discarded.
//! Missing `_RectClip` defaults to off.
//!
//! **Clip range (Unity `CLIP` keyword):** Defaults `_ClipMin = 0`, `_ClipMax = 1`
//! make the rescale a no-op, so the keyword need not be a separate uniform.


#import renderide::filter_vertex as fv
#import renderide::globals as rg
#import renderide::scene_depth_sample as sds
#import renderide::ui::rect_clip as uirc

struct FiltersGetDepthMaterial {
    _Rect: vec4<f32>,
    _Multiply: f32,
    _Offset: f32,
    _ClipMin: f32,
    _ClipMax: f32,
    _RectClip: f32,
    _pad0: f32,
    _pad1: vec2<f32>,
}

@group(1) @binding(0) var<uniform> mat: FiltersGetDepthMaterial;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) primary_uv: vec2<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) world_n: vec3<f32>,
    @location(3) @interpolate(flat) view_layer: u32,
    @location(4) view_n: vec3<f32>,
    @location(5) obj_xy: vec2<f32>,
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
) -> VertexOutput {
#ifdef MULTIVIEW
    let inner = fv::vertex_main(instance_index, view_idx, pos, n, t, uv0);
#else
    let inner = fv::vertex_main(instance_index, 0u, pos, n, t, uv0);
#endif
    var out: VertexOutput;
    out.clip_pos = inner.clip_pos;
    out.primary_uv = inner.primary_uv;
    out.world_pos = inner.world_pos;
    out.world_n = inner.world_n;
    out.view_layer = inner.view_layer;
    out.view_n = inner.view_n;
    out.obj_xy = pos.xy;
    return out;
}

//#pass forward
@fragment
fn fs_main(vout: VertexOutput) -> @location(0) vec4<f32> {
    if (uirc::should_clip_rect(vout.obj_xy, mat._Rect, mat._RectClip)) {
        discard;
    }

    var depth = sds::scene_linear_depth(vout.clip_pos, vout.view_layer);
    let range = max(mat._ClipMax - mat._ClipMin, 1e-6);
    depth = (depth - mat._ClipMin) / range;
    depth = depth * mat._Multiply + mat._Offset;
    depth = clamp(depth, 0.0, 1.0);

    return rg::retain_globals_additive(vec4<f32>(vec3<f32>(depth), 1.0));
}
