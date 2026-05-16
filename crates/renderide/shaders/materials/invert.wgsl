//! Grab-pass color inversion filter (`Shader "Filters/Invert"`).


//#mat_default _Lerp float 1.0

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
    let filtered = mix(c.rgb, 1.0 - c.rgb, mat._Lerp);
    return fc::retain_globals(vec4<f32>(filtered, c.a));
}
