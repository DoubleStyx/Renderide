//! Canvas UI Unlit (`Shader "UI/Unlit"`): sprite texture, tint, alpha clip, mask, rect clip, overlay.
//!
//! Build emits `ui_unlit_default` / `ui_unlit_multiview` via [`MULTIVIEW`](https://docs.rs/naga_oil).
//! `@group(1)` field names match Unity `UI_Unlit.shader` material property names for host reflection.
//!
//! Vertex color: Unity multiplies `vertex_color * _Tint`. The mesh pass provides a dense
//! float4 color stream at `@location(3)` with opaque-white fallback when the host mesh lacks color.
//!
//! Froox `#pragma multi_compile` keywords (`ALPHACLIP`, `RECTCLIP`, `OVERLAY`,
//! `TEXTURE_NORMALMAP`/`TEXTURE_LERPCOLOR`, `_MASK_TEXTURE_MUL`/`_MASK_TEXTURE_CLIP`) are decoded
//! from the renderer-reserved `_RenderideVariantBits` uniform; bit positions match Froox's
//! sorted `UniqueKeywords` list (underscore-prefixed keywords sort before letters).
//!
//! Per-draw uniforms (`@group(2)`) use [`renderide::draw::per_draw`].


//#texture_default _MainTex white
//#texture_default _MaskTex white

#import renderide::core::texture_sampling as ts
#import renderide::frame::globals as rg
#import renderide::material::alpha as ma
#import renderide::material::variant_bits as vb
#import renderide::mesh::vertex as mv
#import renderide::core::normal_decode as nd
#import renderide::core::uv as uvu
#import renderide::draw::per_draw as pd
#import renderide::ui::overlay_tint as uiot
#import renderide::ui::rect_clip as uirc

struct UiUnlitMaterial {
    _MainTex_ST: vec4<f32>,
    _MaskTex_ST: vec4<f32>,
    _Tint: vec4<f32>,
    _OverlayTint: vec4<f32>,
    _Rect: vec4<f32>,
    _Cutoff: f32,
    _RenderideVariantBits: u32,
    _MainTex_LodBias: f32,
    _MaskTex_LodBias: f32,
}

const UIUNLIT_KW_MASK_TEXTURE_CLIP: u32 = 1u << 0u;
const UIUNLIT_KW_MASK_TEXTURE_MUL: u32 = 1u << 1u;
const UIUNLIT_KW_ALPHACLIP: u32 = 1u << 2u;
const UIUNLIT_KW_OVERLAY: u32 = 1u << 3u;
const UIUNLIT_KW_RECTCLIP: u32 = 1u << 4u;
const UIUNLIT_KW_TEXTURE_LERPCOLOR: u32 = 1u << 5u;
const UIUNLIT_KW_TEXTURE_NORMALMAP: u32 = 1u << 6u;

@group(1) @binding(0) var<uniform> mat: UiUnlitMaterial;
@group(1) @binding(1) var _MainTex: texture_2d<f32>;
@group(1) @binding(2) var _MainTex_sampler: sampler;
@group(1) @binding(3) var _MaskTex: texture_2d<f32>;
@group(1) @binding(4) var _MaskTex_sampler: sampler;

fn ui_unlit_kw(mask: u32) -> bool {
    return vb::enabled(mat._RenderideVariantBits, mask);
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) lerp_color: vec4<f32>,
    @location(3) obj_xy: vec2<f32>,
    @location(4) world_pos: vec3<f32>,
    @location(5) @interpolate(flat) view_layer: u32,
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
    @location(3) color: vec4<f32>,
    @location(4) tangent: vec4<f32>,
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
    let world_p = mv::world_position(d, pos);
#ifdef MULTIVIEW
    let vp = mv::select_view_proj(d, view_idx);
    let layer = mv::packed_view_layer(instance_index, view_idx);
#else
    let vp = mv::select_view_proj(d, 0u);
    let layer = mv::packed_view_layer(instance_index, 0u);
#endif

    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.uv = uv;
    out.color = color * mat._Tint;
    out.lerp_color = tangent * mat._Tint;
    out.obj_xy = pos.xy;
    out.world_pos = world_p.xyz;
    out.view_layer = layer;
    return out;
}

//#pass forward
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if (uirc::should_clip_rect_kw(in.obj_xy, mat._Rect, ui_unlit_kw(UIUNLIT_KW_RECTCLIP))) {
        discard;
    }

    let uv_main = uvu::apply_st(in.uv, mat._MainTex_ST);
    var tex_color = ts::sample_tex_2d(_MainTex, _MainTex_sampler, uv_main, mat._MainTex_LodBias);
    if (ui_unlit_kw(UIUNLIT_KW_TEXTURE_NORMALMAP)) {
        tex_color = vec4<f32>(nd::decode_ts_normal_with_placeholder_sample(tex_color, 1.0) * 0.5 + vec3<f32>(0.5), 1.0);
    }

    var color: vec4<f32>;
    if (ui_unlit_kw(UIUNLIT_KW_TEXTURE_LERPCOLOR)) {
        let l = dot(tex_color.rgb, vec3<f32>(0.3333333333));
        let lerp_color = mix(in.color, in.lerp_color, l);
        color = vec4<f32>(lerp_color.rgb, lerp_color.a * tex_color.a);
    } else {
        color = in.color * tex_color;
    }

    let mask_mul = ui_unlit_kw(UIUNLIT_KW_MASK_TEXTURE_MUL);
    let mask_clip = ui_unlit_kw(UIUNLIT_KW_MASK_TEXTURE_CLIP);
    if (mask_mul || mask_clip) {
        let uv_mask = uvu::apply_st(in.uv, mat._MaskTex_ST);
        let mask_sample = ts::sample_tex_2d(_MaskTex, _MaskTex_sampler, uv_mask, mat._MaskTex_LodBias);
        let mul = ma::mask_luminance(mask_sample);
        if (mask_mul) {
            color.a = color.a * mul;
        }
        if (mask_clip && mul <= mat._Cutoff) {
            discard;
        }
    }

    if (ui_unlit_kw(UIUNLIT_KW_ALPHACLIP) && !mask_clip && color.a <= mat._Cutoff) {
        discard;
    }

    color = uiot::apply_overlay_tint(
        color,
        mat._OverlayTint,
        in.clip_pos,
        in.world_pos,
        in.view_layer,
        ui_unlit_kw(UIUNLIT_KW_OVERLAY),
    );

    return rg::retain_globals_additive(color);
}
