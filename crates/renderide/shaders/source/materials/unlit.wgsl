//! World Unlit (`Shader "Unlit"`): texture × tint, optional alpha test, mask and offset textures.
//!
//! Build emits `unlit_default` / `unlit_multiview` targets via [`MULTIVIEW`](https://docs.rs/naga_oil).
//! `@group(1)` identifiers match Unity material property names (`_Color`, `_Tex`, …) for host binding by reflection.
//!
//! Per-frame bindings (`@group(0)`) are imported from `globals.wgsl` so composed targets match the frame bind group layout used by the renderer.
//! Per-draw uniforms (`@group(2)`) use [`renderide::per_draw`].
//!
//! ## `flags` bits (host-set u32)
//! | Bit | Mask | Meaning |
//! |-----|------|---------|
//! | 0   | 0x01 | Sample `_Tex` and multiply into color (`_TEXTURE`) |
//! | 1   | 0x02 | Alpha clip vs `_Cutoff` (`_ALPHATEST`) |
//! | 2   | 0x04 | Apply `_OffsetTex` UV offset (`_OFFSET_TEXTURE`) |
//! | 3   | 0x08 | Multiply alpha by `_MaskTex` luminance×alpha (`_MASK_TEXTURE_MUL`) |
//! | 4   | 0x10 | Clip when `_MaskTex` luminance×alpha < `_Cutoff` (`_MASK_TEXTURE_CLIP`) |
//! | 5   | 0x20 | Premultiply: multiply RGB by alpha after all ops (`_MUL_RGB_BY_ALPHA`) |
//! | 6   | 0x40 | Additive alpha: multiply output alpha by luminance (`_MUL_ALPHA_INTENSITY`) |

#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::alpha_clip_sample as acs
#import renderide::uv_utils as uvu

struct UnlitMaterial {
    _Color: vec4<f32>,
    _Tex_ST: vec4<f32>,
    _MaskTex_ST: vec4<f32>,
    _OffsetTex_ST: vec4<f32>,
    // Unity declares _OffsetMagnitude as a Vector property (vec4); only xy are used.
    _OffsetMagnitude: vec4<f32>,
    _Cutoff: f32,
    _PolarPow: f32,
    flags: u32,
    _SrcBlend: f32,
    _DstBlend: f32,
    _ZWrite: f32,
    _Cull: f32,
    _ZTest: f32,
}

@group(1) @binding(0) var<uniform> mat: UnlitMaterial;
@group(1) @binding(1) var _Tex: texture_2d<f32>;
@group(1) @binding(2) var _Tex_sampler: sampler;
@group(1) @binding(3) var _OffsetTex: texture_2d<f32>;
@group(1) @binding(4) var _OffsetTex_sampler: sampler;
@group(1) @binding(5) var _MaskTex: texture_2d<f32>;
@group(1) @binding(6) var _MaskTex_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
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
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var albedo = mat._Color;
    // Alpha for `_ALPHATEST` vs `_Cutoff` (base mip — stable coverage).
    var clip_a = mat._Color.a;

    // --- Main texture ---
    if ((mat.flags & 1u) != 0u) {
        var uv_main = uvu::apply_st(in.uv, mat._Tex_ST);

        // Offset texture: shift UV by (offsetSample.xy * _OffsetMagnitude).
        if ((mat.flags & 4u) != 0u) {
            let uv_off = uvu::apply_st(in.uv, mat._OffsetTex_ST);
            let offset_s = textureSample(_OffsetTex, _OffsetTex_sampler, uv_off);
            uv_main = uv_main + offset_s.xy * mat._OffsetMagnitude.xy;
        }

        let t = textureSample(_Tex, _Tex_sampler, uv_main);
        clip_a = mat._Color.a * acs::texture_alpha_base_mip(_Tex, _Tex_sampler, uv_main);
        albedo = albedo * t;
    }

    // --- Mask texture (MUL and/or CLIP) ---
    if ((mat.flags & 24u) != 0u) {   // bits 3 or 4
        let uv_mask = uvu::apply_st(in.uv, mat._MaskTex_ST);
        let mask = textureSample(_MaskTex, _MaskTex_sampler, uv_mask);
        // Unity: mul = (r+g+b) * 0.3333 * a  (luminance × alpha)
        let mul = (mask.r + mask.g + mask.b) * 0.33333334 * mask.a;
        let mul_clip = acs::mask_luminance_mul_base_mip(_MaskTex, _MaskTex_sampler, uv_mask);

        if ((mat.flags & 8u) != 0u) {     // _MASK_TEXTURE_MUL
            albedo.a = albedo.a * mul;
            clip_a = clip_a * mul_clip;
        }
        if ((mat.flags & 16u) != 0u) {    // _MASK_TEXTURE_CLIP
            // Unity: `if (mul - _Cutoff <= 0) discard` → discard when mul <= _Cutoff.
            if (mul_clip <= mat._Cutoff) {
                discard;
            }
        }
    }

    // --- Alpha clip — skipped when mask clip is already active (mirrors Unity #pragma) ---
    if ((mat.flags & 2u) != 0u && (mat.flags & 16u) == 0u) {
        // Unity: `if (col.a - _Cutoff <= 0) discard` → discard when a <= _Cutoff.
        if (clip_a <= mat._Cutoff) {
            discard;
        }
    }

    // --- Premultiply RGB by alpha (_MUL_RGB_BY_ALPHA) ---
    if ((mat.flags & 32u) != 0u) {
        albedo = vec4<f32>(albedo.rgb * albedo.a, albedo.a);
    }

    // --- Additive alpha: replace alpha with luminance (_MUL_ALPHA_INTENSITY) ---
    if ((mat.flags & 64u) != 0u) {
        let lum = (albedo.r + albedo.g + albedo.b) * 0.33333334;
        albedo.a = albedo.a * lum;
    }

    return rg::retain_globals_additive(albedo);
}
