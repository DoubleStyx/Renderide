//! Canvas UI Unlit (`Shader "UI/Unlit"`): sprite texture, tint, alpha clip, mask, rect clip, overlay.
//!
//! Build emits `ui_unlit_default` / `ui_unlit_multiview` via [`MULTIVIEW`](https://docs.rs/naga_oil).
//! `@group(1)` global names match Unity `UI_Unlit.shader` material property names for host reflection.
//!
//! **Vertex color:** Unity multiplies `vertex_color * _Tint`. The mesh pass provides a dense
//! float4 color stream at `@location(3)` with opaque-white fallback when the host mesh lacks color.
//!
//! Unity keyword-only modes are exposed as explicit scalar gates (`_ALPHACLIP`,
//! `_TEXTURE_NORMALMAP`, `_TEXTURE_LERPCOLOR`, `_MASK_TEXTURE_MUL`, `_MASK_TEXTURE_CLIP`).
//! Missing gates default off; `_ALPHATEST_ON` / `_ALPHABLEND_ON` remain as best-effort fallbacks
//! inferred from on-wire render state.
//!
//! Per-draw uniforms (`@group(2)`) use [`renderide::per_draw`].


#import renderide::texture_sampling as ts
#import renderide::globals as rg
#import renderide::alpha_clip_sample as acs
#import renderide::material::alpha as ma
#import renderide::mesh::vertex as mv
#import renderide::normal_decode as nd
#import renderide::per_draw as pd
#import renderide::scene_depth_sample as sds
#import renderide::ui::rect_clip as uirc
#import renderide::uv_utils as uvu

struct UiUnlitMaterial {
    _MainTex_ST: vec4<f32>,
    _MaskTex_ST: vec4<f32>,
    _Tint: vec4<f32>,
    _OverlayTint: vec4<f32>,
    _Rect: vec4<f32>,
    _MainTex_StorageVInverted: f32,
    _MaskTex_StorageVInverted: f32,
    _Cutoff: f32,
    _ALPHACLIP: f32,
    _ALPHATEST_ON: f32,
    _ALPHABLEND_ON: f32,
    _TEXTURE_NORMALMAP: f32,
    _TEXTURE_LERPCOLOR: f32,
    _MASK_TEXTURE_MUL: f32,
    _MASK_TEXTURE_CLIP: f32,
    _RectClip: f32,
    _OVERLAY: f32,
    _MainTex_LodBias: f32,
    _MaskTex_LodBias: f32,
    _pad0: vec2<f32>,
}

@group(1) @binding(0) var<uniform> mat: UiUnlitMaterial;
@group(1) @binding(1) var _MainTex: texture_2d<f32>;
@group(1) @binding(2) var _MainTex_sampler: sampler;
@group(1) @binding(3) var _MaskTex: texture_2d<f32>;
@group(1) @binding(4) var _MaskTex_sampler: sampler;

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
    let layer = view_idx;
#else
    let vp = mv::select_view_proj(d, 0u);
    let layer = 0u;
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

fn main_uv_for_storage(uv_main: vec2<f32>) -> vec2<f32> {
    return uvu::flip_v_for_storage(uv_main, mat._MainTex_StorageVInverted);
}

fn mask_uv_for_storage(uv_main: vec2<f32>) -> vec2<f32> {
    let uv_mask = uv_main * mat._MaskTex_ST.xy + mat._MaskTex_ST.zw;
    return uvu::flip_v_for_storage(uv_mask, mat._MaskTex_StorageVInverted);
}

fn main_uv_no_storage_flip(uv: vec2<f32>) -> vec2<f32> {
    return uv * mat._MainTex_ST.xy + mat._MainTex_ST.zw;
}

fn mask_mul_from_sample(mask: vec4<f32>) -> f32 {
    return dot(mask.rgb, vec3<f32>(0.3333333)) * mask.a;
}

//#pass forward
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if (uirc::should_clip_rect(in.obj_xy, mat._Rect, mat._RectClip)) {
        discard;
    }

    let uv_main_unflipped = main_uv_no_storage_flip(in.uv);
    let uv_s = main_uv_for_storage(uv_main_unflipped);
    var tex_color = ts::sample_tex_2d(_MainTex, _MainTex_sampler, uv_s, mat._MainTex_LodBias);
    if (uvu::kw_enabled(mat._TEXTURE_NORMALMAP)) {
        tex_color = vec4<f32>(nd::decode_ts_normal_with_placeholder_sample(tex_color, 1.0) * 0.5 + vec3<f32>(0.5), 1.0);
    }

    var color: vec4<f32>;
    if (uvu::kw_enabled(mat._TEXTURE_LERPCOLOR)) {
        let l = dot(tex_color.rgb, vec3<f32>(0.3333333333));
        let lerp_color = mix(in.color, in.lerp_color, l);
        color = vec4<f32>(lerp_color.rgb, lerp_color.a * tex_color.a);
    } else {
        color = in.color * tex_color;
    }

    let uv_mask = mask_uv_for_storage(uv_main_unflipped);
    let mask_mul_enabled = uvu::kw_enabled(mat._MASK_TEXTURE_MUL);
    let mask_clip_enabled = uvu::kw_enabled(mat._MASK_TEXTURE_CLIP);
    let alpha_test = uvu::kw_enabled(mat._ALPHATEST_ON);
    let alpha_blend = uvu::kw_enabled(mat._ALPHABLEND_ON);
    let alpha_clip = uvu::kw_enabled(mat._ALPHACLIP) || alpha_test;
    var alpha_clip_done = false;

    if (mask_mul_enabled || mask_clip_enabled) {
        let mask = ts::sample_tex_2d(_MaskTex, _MaskTex_sampler, uv_mask, mat._MaskTex_LodBias);
        let mul = mask_mul_from_sample(mask);
        if (mask_mul_enabled) {
            color.a = color.a * mul;
        }
        if (mask_clip_enabled && mul <= mat._Cutoff) {
            discard;
        }
    } else if (alpha_blend) {
        let mask_sample = ts::sample_tex_2d(_MaskTex, _MaskTex_sampler, uv_mask, mat._MaskTex_LodBias);
        color = ma::apply_alpha_mask(color, mask_sample);
    } else if (alpha_test) {
        let mask_clip_alpha = acs::mask_luminance_mul_base_mip(_MaskTex, _MaskTex_sampler, uv_mask);
        if (color.a * mask_clip_alpha <= mat._Cutoff) {
            discard;
        }
        alpha_clip_done = true;
    }

    if (!alpha_clip_done && alpha_clip && !mask_clip_enabled) {
        if (color.a <= mat._Cutoff) {
            discard;
        }
    }

    if (uvu::kw_enabled(mat._OVERLAY)) {
        let scene_z = sds::scene_linear_depth(in.clip_pos, in.view_layer);
        let part_z = sds::fragment_linear_depth(in.world_pos, in.view_layer);
        if (part_z > scene_z) {
            color = color * mat._OverlayTint;
        }
    }

    return rg::retain_globals_additive(color);
}
