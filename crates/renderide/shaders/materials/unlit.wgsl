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
//! through `ShaderKeywords.SetKeyword`, which the renderer never receives. The
//! `_ALPHATEST_ON`, `_ALPHABLEND_ON`, and `_MUL_RGB_BY_ALPHA` keyword fields below are populated by
//! [`crate::backend::embedded::uniform_pack::inferred_keyword_float_f32`] from the on-wire
//! `MaterialRenderType` tag (Cutout enables `_ALPHATEST_ON`; Transparent enables
//! `_ALPHABLEND_ON`). When neither is set the material is treated as Opaque and the mask /
//! cutoff branches stay inert. The default-white texture fallback keeps each mask branch a
//! no-op when no host mask is bound (`mask.a == 1.0`).

#import renderide::texture_sampling as ts
#import renderide::globals as rg
#import renderide::alpha_clip_sample as acs
#import renderide::material::alpha as ma
#import renderide::mesh::vertex as mv
#import renderide::uv_utils as uvu

struct UnlitMaterial {
    _Color: vec4<f32>,
    _Tex_ST: vec4<f32>,
    _Tex_StorageVInverted: f32,
    _MaskTex_ST: vec4<f32>,
    _OffsetTex_ST: vec4<f32>,
    _OffsetMagnitude: vec4<f32>,
    _Cutoff: f32,
    _PolarPow: f32,
    _MUL_RGB_BY_ALPHA: f32,
    _ALPHATEST_ON: f32,
    _ALPHABLEND_ON: f32,
    _Tex_LodBias: f32,
    _OffsetTex_LodBias: f32,
    _MaskTex_LodBias: f32,
}

@group(1) @binding(0) var<uniform> mat: UnlitMaterial;
@group(1) @binding(1) var _Tex: texture_2d<f32>;
@group(1) @binding(2) var _Tex_sampler: sampler;
@group(1) @binding(3) var _OffsetTex: texture_2d<f32>;
@group(1) @binding(4) var _OffsetTex_sampler: sampler;
@group(1) @binding(5) var _MaskTex: texture_2d<f32>;
@group(1) @binding(6) var _MaskTex_sampler: sampler;

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
) -> mv::UvColorVertexOutput {
#ifdef MULTIVIEW
    return mv::uv_color_vertex_main(instance_index, view_idx, pos, uv, color * mat._Color);
#else
    return mv::uv_color_vertex_main(instance_index, 0u, pos, uv, color * mat._Color);
#endif
}

//#pass forward
@fragment
fn fs_main(in: mv::UvColorVertexOutput) -> @location(0) vec4<f32> {
    let uv_off = uvu::apply_st(in.uv, mat._OffsetTex_ST);
    let offset_s = ts::sample_tex_2d(_OffsetTex, _OffsetTex_sampler, uv_off, mat._OffsetTex_LodBias);
    let uv_main = uvu::apply_st_for_storage(in.uv, mat._Tex_ST, mat._Tex_StorageVInverted) + offset_s.xy * mat._OffsetMagnitude.xy;

    let t = ts::sample_tex_2d(_Tex, _Tex_sampler, uv_main, mat._Tex_LodBias);
    var color = in.color * t;

    let alpha_test = uvu::kw_enabled(mat._ALPHATEST_ON);
    let alpha_blend = uvu::kw_enabled(mat._ALPHABLEND_ON);
    let mul_rgb_by_alpha = uvu::kw_enabled(mat._MUL_RGB_BY_ALPHA);

    let uv_mask = uvu::apply_st(in.uv, mat._MaskTex_ST);

    if (alpha_test) {
        let tex_clip_alpha = mat._Color.a * acs::texture_alpha_base_mip(_Tex, _Tex_sampler, uv_main);
        let mask_clip_alpha = acs::mask_luminance_mul_base_mip(_MaskTex, _MaskTex_sampler, uv_mask);
        if (tex_clip_alpha * mask_clip_alpha <= mat._Cutoff) {
            discard;
        }
    } else if (alpha_blend) {
        let mask_sample = ts::sample_tex_2d(_MaskTex, _MaskTex_sampler, uv_mask, mat._MaskTex_LodBias);
        color = ma::apply_alpha_mask(color, mask_sample);
    }

    if (mul_rgb_by_alpha) {
        color = vec4<f32>(ma::apply_premultiply(color.rgb, color.a, true), color.a);
    }

    return rg::retain_globals_additive(color);
}
