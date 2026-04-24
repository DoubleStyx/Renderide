//! World Unlit (`Shader "Unlit"`): texture × tint, optional alpha test,
//! optional UV-shift from a packed offset texture and alpha mask.
//!
//! Build emits `unlit_default` / `unlit_multiview` targets via [`MULTIVIEW`](https://docs.rs/naga_oil).
//! `@group(1)` identifiers match Unity material property names (`_Color`, `_Tex`, `_MaskTex`, `_OffsetTex`, …)
//! so host binding picks them up by reflection.
//!
//! Per-frame bindings (`@group(0)`) are imported from `globals.wgsl` so composed targets match the frame bind group layout used by the renderer.
//! Per-draw uniforms (`@group(2)`) use [`renderide::per_draw`].
//!
//! Mask-mode caveat: Unity's Unlit shader gates mask application on
//! `_MASK_TEXTURE_MUL` / `_MASK_TEXTURE_CLIP` multi-compile keywords that FrooxEngine sets
//! through `ShaderKeywords.SetKeyword`, which the renderer never receives — decoding them
//! would require plumbing `ShaderKeywords.Variant` plus each shader's keyword-index table
//! through IPC (see `ShaderKeywords.cs`). The shader instead infers the mode from signals
//! that *are* on the wire: alpha-test active (`_Cutoff ∈ (0, 1)`) → CLIP; transparent blend
//! (`(_SrcBlend, _DstBlend) ≠ (1, 0)`) → MUL; opaque no-cutoff → mask has no visible effect,
//! skipped. The default-white texture fallback keeps every branch inert when no host mask
//! is bound (`mask.a == 1.0`). The rare case of an opaque cutoff-less material with
//! `MaskMode = AlphaClip` can only be recovered by plumbing the keyword bitmask.

#import renderide::globals as rg
#import renderide::per_draw as pd
#import renderide::alpha_clip_sample as acs
#import renderide::uv_utils as uvu

struct UnlitMaterial {
    _Color: vec4<f32>,
    _Tex_ST: vec4<f32>,
    _MaskTex_ST: vec4<f32>,
    _OffsetTex_ST: vec4<f32>,
    _OffsetMagnitude: vec4<f32>,
    _Cutoff: f32,
    _PolarPow: f32,
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
    let uv_off = uvu::apply_st(in.uv, mat._OffsetTex_ST);
    let offset_s = textureSample(_OffsetTex, _OffsetTex_sampler, uv_off);
    let uv_main = uvu::apply_st(in.uv, mat._Tex_ST) + offset_s.xy * mat._OffsetMagnitude.xy;

    let t = textureSample(_Tex, _Tex_sampler, uv_main);
    var albedo = mat._Color * t;
    var clip_a = mat._Color.a * acs::texture_alpha_base_mip(_Tex, _Tex_sampler, uv_main);

    let uv_mask = uvu::apply_st(in.uv, mat._MaskTex_ST);
    let cutoff_active = mat._Cutoff > 0.0 && mat._Cutoff < 1.0;
    let is_opaque_blend = mat._SrcBlend == 1.0 && mat._DstBlend == 0.0;
    if (cutoff_active) {
        clip_a = clip_a * acs::texture_alpha_base_mip(_MaskTex, _MaskTex_sampler, uv_mask);
    } else if (!is_opaque_blend) {
        albedo.a = albedo.a * textureSample(_MaskTex, _MaskTex_sampler, uv_mask).a;
    }

    // Alpha test is active when `_Cutoff` is a meaningful value in (0, 1); otherwise every
    // alpha at exactly 0 or exactly 1 would either never discard or always discard.
    if (mat._Cutoff > 0.0 && mat._Cutoff < 1.0 && clip_a <= mat._Cutoff) {
        discard;
    }

    return rg::retain_globals_additive(albedo);
}
