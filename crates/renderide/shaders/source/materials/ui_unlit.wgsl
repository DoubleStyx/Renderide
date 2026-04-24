//! Canvas UI Unlit (`Shader "UI/Unlit"`): sprite texture, tint, optional alpha clip, optional alpha mask.
//!
//! Build emits `ui_unlit_default` / `ui_unlit_multiview` via [`MULTIVIEW`](https://docs.rs/naga_oil).
//! `@group(1)` global names match Unity `UI_Unlit.shader` material property names for host reflection.
//!
//! **Vertex color:** Unity multiplies `vertex_color * _Tint`. The mesh pass provides a dense
//! float4 color stream at `@location(3)` with opaque-white fallback when the host mesh lacks color.
//!
//! Mask-mode caveat: Unity's UI_Unlit shader gates mask handling on
//! `_MASK_TEXTURE_MUL` / `_MASK_TEXTURE_CLIP` multi-compile keywords that FrooxEngine sets
//! through `ShaderKeywords.SetKeyword`, which the renderer never receives — decoding them
//! would require plumbing `ShaderKeywords.Variant` and each shader's keyword-index table
//! through IPC. The shader instead infers the mode from signals that *are* on the wire:
//! alpha-test active (`_Cutoff ∈ (0, 1)`) → CLIP; transparent blend
//! (`(_SrcBlend, _DstBlend) ≠ (1, 0)`) → MUL; opaque no-cutoff → mask skipped. The
//! default-white texture fallback keeps every branch inert when no host mask is bound
//! (`mask.a == 1.0`).
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
    return out;
}

//#material forward_base
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv_s = uvu::apply_st(in.uv, mat._MainTex_ST);
    let t = textureSample(_MainTex, _MainTex_sampler, uv_s);
    var color = in.color * t;
    var clip_a = in.color.a * acs::texture_alpha_base_mip(_MainTex, _MainTex_sampler, uv_s);

    let uv_mask = uvu::apply_st(in.uv, mat._MaskTex_ST);
    let cutoff_active = mat._Cutoff > 0.0 && mat._Cutoff < 1.0;
    let is_opaque_blend = mat._SrcBlend == 1.0 && mat._DstBlend == 0.0;
    if (cutoff_active) {
        clip_a = clip_a * acs::texture_alpha_base_mip(_MaskTex, _MaskTex_sampler, uv_mask);
    } else if (!is_opaque_blend) {
        color.a = color.a * textureSample(_MaskTex, _MaskTex_sampler, uv_mask).a;
    }

    if (mat._Cutoff > 0.0 && mat._Cutoff < 1.0 && clip_a <= mat._Cutoff) {
        discard;
    }

    return rg::retain_globals_additive(color);
}
