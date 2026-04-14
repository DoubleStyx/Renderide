//! Canvas UI Unlit (`Shader "UI/Unlit"`): sprite texture, tint, optional rect/mask/alpha paths.
//!
//! Build emits `ui_unlit_default` / `ui_unlit_multiview` via [`MULTIVIEW`](https://docs.rs/naga_oil).
//! `@group(1)` global names match Unity `UI_Unlit.shader` material property names for host reflection.
//!
//! **Vertex color:** Unity multiplies `vertex_color * _Tint`. The mesh pass now provides a dense
//! float4 color stream at `@location(3)` with opaque-white fallback when the host mesh lacks color.
//!
//! **`flags` bits (host / material):** bit0 = sample `_MainTex`; bit1 = alpha clip on final alpha;
//! bit2 = rect clip using `_Rect` (xy = min, zw = size in object XY); bit3 = overlay tint stub
//! (multiplies by `_OverlayTint.a` as a stand-in; no scene depth); bit4 = mask multiply alpha;
//! bit5 = mask alpha clip vs `_Cutoff`. The manifest CPU path also sets bit0/bit1 from texture presence and `_Cutoff` when `_Flags` is absent.
//!
//! Per-draw uniforms (`@group(2)`) use [`renderide::per_draw`].

#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::alpha_clip_sample as acs
#import renderide::uv_utils as uvu

struct UiUnlitMaterial {
    _MainTex_ST: vec4<f32>,
    _MaskTex_ST: vec4<f32>,
    _Tint: vec4<f32>,
    _OverlayTint: vec4<f32>,
    _Rect: vec4<f32>,
    _Cutoff: f32,
    _SrcBlend: f32,
    _DstBlend: f32,
    _ZWrite: f32,
    _Cull: f32,
    _StencilComp: f32,
    _Stencil: f32,
    _StencilOp: f32,
    _StencilWriteMask: f32,
    _StencilReadMask: f32,
    _ColorMask: f32,
    flags: u32,
    _pad_end: vec2<f32>,
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
    @location(2) obj_xy: vec2<f32>,
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
) -> VertexOutput {
    let d = pd::get_draw(instance_index);
    let world_p = d.model * vec4<f32>(pos.xyz, 1.0);
#ifdef MULTIVIEW
    var vp: mat4x4<f32>;
    if (view_idx == 0u) {
        vp = d.view_proj_left;
    } else {
        vp = d.view_proj_right;
    }
#else
    let vp = d.view_proj_left;
#endif
    var out: VertexOutput;
    out.clip_pos = vp * world_p;
    out.uv = uv;
    out.color = color * mat._Tint;
    out.obj_xy = pos.xy;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var color = in.color;
    var clip_a = in.color.a;

    if ((mat.flags & 1u) != 0u) {
        let uv_s = uvu::apply_st(in.uv, mat._MainTex_ST);
        let t = textureSample(_MainTex, _MainTex_sampler, uv_s);
        clip_a = in.color.a * acs::texture_alpha_base_mip(_MainTex, _MainTex_sampler, uv_s);
        color = color * t;
    }

    if ((mat.flags & 4u) != 0u) {
        let r = mat._Rect;
        let min_v = r.xy;
        let max_v = r.xy + r.zw;
        if (in.obj_xy.x < min_v.x || in.obj_xy.x > max_v.x || in.obj_xy.y < min_v.y || in.obj_xy.y > max_v.y) {
            discard;
        }
    }

    if ((mat.flags & 48u) != 0u) {
        let uv_m = uvu::apply_st(in.uv, mat._MaskTex_ST);
        let mask = textureSample(_MaskTex, _MaskTex_sampler, uv_m);
        let mul = (mask.r + mask.g + mask.b) * 0.33333334 * mask.a;
        let mul_clip = acs::mask_luminance_mul_base_mip(_MaskTex, _MaskTex_sampler, uv_m);
        if ((mat.flags & 16u) != 0u) {
            color.a = color.a * mul;
            clip_a = clip_a * mul_clip;
        }
        if ((mat.flags & 32u) != 0u) {
            // Unity: `if (mul - _Cutoff <= 0) discard`
            if (mul_clip <= mat._Cutoff) {
                discard;
            }
        }
    }

    // Alpha clip — skipped when mask clip is already active (mirrors Unity #pragma).
    if ((mat.flags & 2u) != 0u && (mat.flags & 32u) == 0u) {
        if (clip_a <= mat._Cutoff) {
            discard;
        }
    }

    if ((mat.flags & 8u) != 0u) {
        let o = mat._OverlayTint;
        color = vec4<f32>(color.rgb * mix(vec3<f32>(1.0), o.rgb, o.a), color.a);
    }

    return rg::retain_globals_additive(color);
}
