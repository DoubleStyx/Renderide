//! Grab-pass pixelation filter (`Shader "Filters/Pixelate"`).
//!
//! Snaps the grab-screen UV onto a discrete pixel grid sized by `_Resolution` (optionally scaled
//! per-pixel by `_ResolutionTex` when the `RESOLUTION_TEX` variant bit is set), then resamples the
//! grab-pass scene color at that quantized UV. The `RECTCLIP` variant bit gates an object-space
//! `_Rect` clip; both keywords come from Resonite's variant bitmask.


//#render_queue Transparent+500
//#texture_default _ResolutionTex white
//#mat_default _Resolution vec4 100.0 100.0 0.0 0.0

#import renderide::post::filter_math as fm
#import renderide::post::filter_vertex as fv
#import renderide::post::filter_common as fc
#import renderide::frame::grab_pass as gp
#import renderide::material::variant_bits as vb
#import renderide::core::texture_sampling as ts
#import renderide::core::uv as uvu

struct FiltersPixelateMaterial {
    _Resolution: vec4<f32>,
    _ResolutionTex_ST: vec4<f32>,
    _Rect: vec4<f32>,
    _RenderideVariantBits: u32,
    _ResolutionTex_LodBias: f32,
    _pad1: vec2<f32>,
}

const PIXELATE_KW_RECTCLIP: u32 = 1u << 0u;
const PIXELATE_KW_RESOLUTION_TEX: u32 = 1u << 1u;

@group(1) @binding(0) var<uniform> mat: FiltersPixelateMaterial;
@group(1) @binding(1) var _ResolutionTex: texture_2d<f32>;
@group(1) @binding(2) var _ResolutionTex_sampler: sampler;

fn pixelate_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

fn kw_RECTCLIP() -> bool {
    return pixelate_kw(PIXELATE_KW_RECTCLIP);
}

fn kw_RESOLUTION_TEX() -> bool {
    return pixelate_kw(PIXELATE_KW_RESOLUTION_TEX);
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

//#pass type=forward name=forward_filter blend=material_filter
@fragment
fn fs_main(vout: fv::RectVertexOutput) -> @location(0) vec4<f32> {
    fc::discard_rect_if_enabled(vout.obj_xy, mat._Rect, kw_RECTCLIP());

    var resolution = max(mat._Resolution.xy, vec2<f32>(1.0));
    if (kw_RESOLUTION_TEX()) {
        let texel_scale = ts::sample_tex_2d(_ResolutionTex, _ResolutionTex_sampler, uvu::apply_st(vout.primary_uv, mat._ResolutionTex_ST), mat._ResolutionTex_LodBias).rg;
        resolution = max(mat._Resolution.xy * texel_scale, vec2<f32>(1.0));
    }
    let uv = fm::safe_div_vec2(round(fc::screen_uv(vout.clip_pos) * resolution), resolution);
    return fc::retain_globals(gp::sample_scene_color(uv, vout.view_layer));
}
