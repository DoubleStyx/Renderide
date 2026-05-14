//! Grab-pass pixelation filter (`Shader "Filters/Pixelate"`).
//!
//! Snaps the grab-screen UV onto a discrete pixel grid sized by `_Resolution` (optionally scaled
//! per-pixel by `_ResolutionTex` when the `RESOLUTION_TEX` variant bit is set), then resamples the
//! grab-pass scene color at that quantized UV. The `RECTCLIP` variant bit gates an object-space
//! `_Rect` clip; both keywords come from Resonite's variant bitmask.


//#texture_default _ResolutionTex white

#import renderide::post::filter_math as fm
#import renderide::post::filter_vertex as fv
#import renderide::frame::globals as rg
#import renderide::frame::grab_pass as gp
#import renderide::material::variant_bits as vb
#import renderide::ui::rect_clip as uirc
#import renderide::core::uv as uvu

struct FiltersPixelateMaterial {
    _Resolution: vec4<f32>,
    _ResolutionTex_ST: vec4<f32>,
    _Rect: vec4<f32>,
    _RenderideVariantBits: u32,
    _pad0: f32,
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

//#pass forward_filter
@fragment
fn fs_main(vout: VertexOutput) -> @location(0) vec4<f32> {
    if (uirc::should_clip_rect_kw(vout.obj_xy, mat._Rect, kw_RECTCLIP())) {
        discard;
    }

    var resolution = max(mat._Resolution.xy, vec2<f32>(1.0));
    if (kw_RESOLUTION_TEX()) {
        let texel_scale = textureSample(_ResolutionTex, _ResolutionTex_sampler, uvu::apply_st(vout.primary_uv, mat._ResolutionTex_ST)).rg;
        resolution = max(mat._Resolution.xy * texel_scale, vec2<f32>(1.0));
    }
    let uv = fm::safe_div_vec2(round(gp::frag_screen_uv(vout.clip_pos) * resolution), resolution);
    return rg::retain_globals_additive(gp::sample_scene_color(uv, vout.view_layer));
}
