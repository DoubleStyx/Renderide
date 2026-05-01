//! UV Rect (Unity shader asset `UVRect`): colors inside/outside a UV-space rect.
//!
//! `_RectClip` is an explicit helper field; missing values default to clipping disabled.

#import renderide::globals as rg
#import renderide::math as rmath
#import renderide::mesh::vertex as mv

struct UvRectMaterial {
    _Rect: vec4<f32>,
    _ClipRect: vec4<f32>,
    _OuterColor: vec4<f32>,
    _InnerColor: vec4<f32>,
    _RectClip: f32,
    _pad0: vec2<f32>,
}

@group(1) @binding(0) var<uniform> mat: UvRectMaterial;

@vertex
fn vs_main(
    @builtin(instance_index) instance_index: u32,
#ifdef MULTIVIEW
    @builtin(view_index) view_idx: u32,
#endif
    @location(0) pos: vec4<f32>,
    @location(1) _n: vec4<f32>,
    @location(2) uv: vec2<f32>,
) -> mv::UvVertexOutput {
#ifdef MULTIVIEW
    return mv::uv_vertex_main(instance_index, view_idx, pos, uv);
#else
    return mv::uv_vertex_main(instance_index, 0u, pos, uv);
#endif
}

//#pass forward
@fragment
fn fs_main(in: mv::UvVertexOutput) -> @location(0) vec4<f32> {
    if (mat._RectClip > 0.5 && rmath::outside_rect(in.uv, mat._ClipRect)) {
        discard;
    }

    let inner = rmath::inside_rect_mask(in.uv, mat._Rect);
    let color = mix(mat._OuterColor, mat._InnerColor, inner);

    return rg::retain_globals_additive(color);
}
