//! UV Rect (Unity shader asset `UVRect`): colors inside/outside a UV-space rect.
//!
//! Froox variant bits populate `_RenderideVariantBits`; the `RECTCLIP` keyword (host driver
//! at `UV_RectMaterial.UpdateKeywords`) gates the clip-rect discard.

//#mat_default _ClipRect vec4 0.0 0.0 1.0 1.0
//#mat_default _InnerColor vec4 1.0 1.0 1.0 1.0
//#mat_default _OuterColor vec4 0.0 0.0 0.0 1.0
//#mat_default _Rect vec4 0.25 0.25 0.75 0.75

#import renderide::frame::globals as rg
#import renderide::core::math as rmath
#import renderide::material::variant_bits as vb
#import renderide::mesh::vertex as mv

struct UvRectMaterial {
    _Rect: vec4<f32>,
    _ClipRect: vec4<f32>,
    _OuterColor: vec4<f32>,
    _InnerColor: vec4<f32>,
    _RenderideVariantBits: u32,
}

const UVRECT_KW_RECTCLIP: u32 = 1u << 0u;

@group(1) @binding(0) var<uniform> mat: UvRectMaterial;

fn uvrect_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

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
    if (uvrect_kw(UVRECT_KW_RECTCLIP) && rmath::outside_rect(in.uv, mat._ClipRect)) {
        discard;
    }

    let inner = rmath::inside_rect_mask(in.uv, mat._Rect);
    let color = mix(mat._OuterColor, mat._InnerColor, inner);

    return rg::retain_globals_additive(color);
}
