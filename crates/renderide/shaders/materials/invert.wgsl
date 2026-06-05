//! Grab-pass color inversion filter (`Shader "Filters/Invert"`).


//#render_queue Transparent+500
//#mat_default _Lerp float 1.0

#import renderide::billboard::vertex as bv
#import renderide::post::filter_vertex as fv
#import renderide::post::filter_common as fc
#import renderide::material::variant_bits as vb

struct FiltersInvertMaterial {
    _Rect: vec4<f32>,
    _Lerp: f32,
    _RenderideVariantBits: u32,
    _pad0: vec2<u32>,
}

const INVERT_KW_RECTCLIP: u32 = 1u << 0u;

@group(1) @binding(0) var<uniform> mat: FiltersInvertMaterial;

fn invert_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn kw_RECTCLIP() -> bool {
    return invert_kw(INVERT_KW_RECTCLIP);
}

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
    @builtin(vertex_index) vertex_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) n: vec4<f32>,
    @location(2) uv0: vec2<f32>,
    @location(4) t: vec4<f32>,
    @location(5) uv1: vec2<f32>,
) -> fv::RectVertexOutput {
#ifdef MULTIVIEW
    let view_layer = view_idx;
#else
    let view_layer = 0u;
#endif
    if (bv::kw_RENDER_BUFFER(mat._RenderideVariantBits)) {
        return fv::billboard_rect_vertex_main(instance_index, view_layer, pos, n, t, uv0, vertex_index, uv1);
    } else {
        return fv::rect_vertex_main(instance_index, view_layer, pos, n, t, uv0);
    }
}

//#pass type=forward name=forward_filter blend=material_filter
@fragment
fn fs_main(in: fv::RectVertexOutput) -> @location(0) vec4<f32> {
    let c = fc::sample_clipped_scene_color_at_clip(in.obj_xy, mat._Rect, kw_RECTCLIP(), in.clip_pos, in.view_layer);
    let filtered = mix(c.rgb, 1.0 - c.rgb, mat._Lerp);
    return fc::retain_scene_alpha(c, filtered);
}
