//! Scene-depth visualization filter (`Shader "Filters/Get Depth"`).
//!
//! Samples the renderer-produced scene-depth snapshot at the fragment position,
//! optionally rescales by `(depth - _ClipMin) / (_ClipMax - _ClipMin)`, applies
//! `_Multiply * depth + _Offset`, saturates, and writes RGB grayscale.
//!
//! Froox variant bits populate `_RenderideVariantBits`; this shader decodes GetDepth's
//! shader-specific keyword bits locally (Unity `CLIP` selects clip-range rescaling,
//! `RECTCLIP` discards fragments outside `_Rect` in object XY).

//#mat_default _ClipMax float 1.0
//#mat_default _Multiply float 1.0

#import renderide::post::filter_vertex as fv
#import renderide::post::filter_common as fc
#import renderide::frame::scene_depth_sample as sds
#import renderide::material::variant_bits as vb

struct FiltersGetDepthMaterial {
    _Rect: vec4<f32>,
    _Multiply: f32,
    _Offset: f32,
    _ClipMin: f32,
    _ClipMax: f32,
    _RenderideVariantBits: u32,
    _pad0: u32,
    _pad1: vec2<u32>,
}

const GETDEPTH_KW_CLIP: u32 = 1u << 0u;
const GETDEPTH_KW_RECTCLIP: u32 = 1u << 1u;

@group(1) @binding(0) var<uniform> mat: FiltersGetDepthMaterial;

fn getdepth_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn kw_CLIP() -> bool {
    return getdepth_kw(GETDEPTH_KW_CLIP);
}

fn kw_RECTCLIP() -> bool {
    return getdepth_kw(GETDEPTH_KW_RECTCLIP);
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
fn fs_main(vout: fv::RectVertexOutput) -> @location(0) vec4<f32> {
    fc::discard_rect_if_enabled(vout.obj_xy, mat._Rect, kw_RECTCLIP());

    var depth = sds::scene_linear_depth(vout.clip_pos, vout.view_layer);
    if (kw_CLIP()) {
        depth = depth - mat._ClipMin;
        depth = depth / (mat._ClipMax - mat._ClipMin);
    }
    depth = depth * mat._Multiply + mat._Offset;
    depth = clamp(depth, 0.0, 1.0);

    return fc::retain_globals(vec4<f32>(vec3<f32>(depth), 1.0));
}
