//! Variant-bit decoding for the Xiexe Toon 2.0 family.
//!
//! The Froox shader-internal-name suffix produces a `u32` bitmask whose bits index into
//! the alphabetically-sorted `UniqueKeywords` list built from the Unity `XSToon2.0` /
//! `XSToon2.0_Outlined` `#pragma multi_compile` groups. Built-in pragmas such as
//! `multi_compile_fog` / `_fwdbase` / `_fwdadd_fullshadows` / `_shadowcaster` are not in
//! `VariantGroups`, so they consume no bits.
//!
//! Two Unity tokens are composite:
//!   * `OCCLUSION_METALLIC` enables both `OCCLUSION_MAP` and `METALLICGLOSS_MAP`.
//!   * `RAMPMASK_OUTLINEMASK_THICKNESS` enables `RAMP_MASK`, `OUTLINE_MASK`, and `THICKNESS_MAP`.
//!
//! Static XSToon shader assets in the `Xiexe/Toon2.0` namespace only declare the
//! `VERTEXLIGHT_ON` material keyword. They use the static-vertexlight layout below so bit 0
//! is not misread as generic `AlphaBlend`.

#define_import_path renderide::xiexe::toon2::variant_bits

#import renderide::material::variant_bits as vb
#import renderide::xiexe::toon2::base as xb

/// Keyword layout used by generic XSToon shaders with the full sorted keyword list.
const XTOON_KEYWORD_LAYOUT_GENERIC: u32 = 0u;
/// Keyword layout used by static XSToon shader assets whose only material keyword is `VERTEXLIGHT_ON`.
const XTOON_KEYWORD_LAYOUT_STATIC_VERTEXLIGHT: u32 = 1u;

/// `AlphaBlend` keyword bit (straight-alpha blending).
const XTOON_KW_ALPHABLEND: u32 = 1u << 0u;
/// `Cutout` keyword bit (alpha-test).
const XTOON_KW_CUTOUT: u32 = 1u << 1u;
/// `EMISSION_MAP` keyword bit.
const XTOON_KW_EMISSION_MAP: u32 = 1u << 2u;
/// `MATCAP` keyword bit.
const XTOON_KW_MATCAP: u32 = 1u << 3u;
/// `NORMAL_MAP` keyword bit.
const XTOON_KW_NORMAL_MAP: u32 = 1u << 4u;
/// `OCCLUSION_METALLIC` keyword bit (drives both metallic-gloss and occlusion maps).
const XTOON_KW_OCCLUSION_METALLIC: u32 = 1u << 5u;
/// `RAMPMASK_OUTLINEMASK_THICKNESS` keyword bit (drives ramp-mask, outline-mask, and thickness).
const XTOON_KW_RAMPMASK_OUTLINEMASK_THICKNESS: u32 = 1u << 6u;
/// `Transparent` keyword bit (premultiplied transparent blending).
const XTOON_KW_TRANSPARENT: u32 = 1u << 7u;
/// `VERTEX_COLOR_ALBEDO` keyword bit.
const XTOON_KW_VERTEX_COLOR_ALBEDO: u32 = 1u << 8u;
/// `VERTEXLIGHT_ON` keyword bit. Present for Froox-side parity; the clustered renderer
/// does not require this keyword to gate per-vertex point-light evaluation.
const XTOON_KW_VERTEXLIGHT_ON: u32 = 1u << 9u;
/// Static XSToon `VERTEXLIGHT_ON` keyword bit.
const XTOON_STATIC_KW_VERTEXLIGHT_ON: u32 = 1u << 0u;

/// Tests one keyword bit against the material's `_RenderideVariantBits`.
fn xtoon_kw(mask: u32) -> bool {
    return vb::enabled(xb::mat._RenderideVariantBits, mask);
}

/// Tests one static-layout keyword bit against the material's `_RenderideVariantBits`.
fn xtoon_static_kw(mask: u32) -> bool {
    return vb::enabled(xb::mat._RenderideVariantBits, mask);
}

/// True when the shader uses the static XSToon keyword layout.
fn static_vertexlight_layout(keyword_layout: u32) -> bool {
    return keyword_layout == XTOON_KEYWORD_LAYOUT_STATIC_VERTEXLIGHT;
}

/// `AlphaBlend` keyword on.
fn kw_AlphaBlend() -> bool {
    return kw_AlphaBlend_for_layout(XTOON_KEYWORD_LAYOUT_GENERIC);
}

/// `AlphaBlend` keyword on for a selected XSToon keyword layout.
fn kw_AlphaBlend_for_layout(keyword_layout: u32) -> bool {
    if (static_vertexlight_layout(keyword_layout)) {
        return false;
    }
    return xtoon_kw(XTOON_KW_ALPHABLEND);
}

/// `Cutout` keyword on.
fn kw_Cutout() -> bool {
    return kw_Cutout_for_layout(XTOON_KEYWORD_LAYOUT_GENERIC);
}

/// `Cutout` keyword on for a selected XSToon keyword layout.
fn kw_Cutout_for_layout(keyword_layout: u32) -> bool {
    if (static_vertexlight_layout(keyword_layout)) {
        return false;
    }
    return xtoon_kw(XTOON_KW_CUTOUT);
}

/// `EMISSION_MAP` keyword on.
fn kw_EMISSION_MAP() -> bool {
    return kw_EMISSION_MAP_for_layout(XTOON_KEYWORD_LAYOUT_GENERIC);
}

/// `EMISSION_MAP` keyword on for a selected XSToon keyword layout.
fn kw_EMISSION_MAP_for_layout(keyword_layout: u32) -> bool {
    if (static_vertexlight_layout(keyword_layout)) {
        return false;
    }
    return xtoon_kw(XTOON_KW_EMISSION_MAP);
}

/// `MATCAP` keyword on.
fn kw_MATCAP() -> bool {
    return kw_MATCAP_for_layout(XTOON_KEYWORD_LAYOUT_GENERIC);
}

/// `MATCAP` keyword on for a selected XSToon keyword layout.
fn kw_MATCAP_for_layout(keyword_layout: u32) -> bool {
    if (static_vertexlight_layout(keyword_layout)) {
        return false;
    }
    return xtoon_kw(XTOON_KW_MATCAP);
}

/// `NORMAL_MAP` keyword on.
fn kw_NORMAL_MAP() -> bool {
    return kw_NORMAL_MAP_for_layout(XTOON_KEYWORD_LAYOUT_GENERIC);
}

/// `NORMAL_MAP` keyword on for a selected XSToon keyword layout.
fn kw_NORMAL_MAP_for_layout(keyword_layout: u32) -> bool {
    if (static_vertexlight_layout(keyword_layout)) {
        return false;
    }
    return xtoon_kw(XTOON_KW_NORMAL_MAP);
}

/// `OCCLUSION_METALLIC` keyword on.
fn kw_OCCLUSION_METALLIC() -> bool {
    return kw_OCCLUSION_METALLIC_for_layout(XTOON_KEYWORD_LAYOUT_GENERIC);
}

/// `OCCLUSION_METALLIC` keyword on for a selected XSToon keyword layout.
fn kw_OCCLUSION_METALLIC_for_layout(keyword_layout: u32) -> bool {
    if (static_vertexlight_layout(keyword_layout)) {
        return false;
    }
    return xtoon_kw(XTOON_KW_OCCLUSION_METALLIC);
}

/// `RAMPMASK_OUTLINEMASK_THICKNESS` keyword on.
fn kw_RAMPMASK_OUTLINEMASK_THICKNESS() -> bool {
    return kw_RAMPMASK_OUTLINEMASK_THICKNESS_for_layout(XTOON_KEYWORD_LAYOUT_GENERIC);
}

/// `RAMPMASK_OUTLINEMASK_THICKNESS` keyword on for a selected XSToon keyword layout.
fn kw_RAMPMASK_OUTLINEMASK_THICKNESS_for_layout(keyword_layout: u32) -> bool {
    if (static_vertexlight_layout(keyword_layout)) {
        return false;
    }
    return xtoon_kw(XTOON_KW_RAMPMASK_OUTLINEMASK_THICKNESS);
}

/// `Transparent` keyword on.
fn kw_Transparent() -> bool {
    return kw_Transparent_for_layout(XTOON_KEYWORD_LAYOUT_GENERIC);
}

/// `Transparent` keyword on for a selected XSToon keyword layout.
fn kw_Transparent_for_layout(keyword_layout: u32) -> bool {
    if (static_vertexlight_layout(keyword_layout)) {
        return false;
    }
    return xtoon_kw(XTOON_KW_TRANSPARENT);
}

/// `VERTEX_COLOR_ALBEDO` keyword on.
fn kw_VERTEX_COLOR_ALBEDO() -> bool {
    return kw_VERTEX_COLOR_ALBEDO_for_layout(XTOON_KEYWORD_LAYOUT_GENERIC);
}

/// `VERTEX_COLOR_ALBEDO` keyword on for a selected XSToon keyword layout.
fn kw_VERTEX_COLOR_ALBEDO_for_layout(keyword_layout: u32) -> bool {
    if (static_vertexlight_layout(keyword_layout)) {
        return false;
    }
    return xtoon_kw(XTOON_KW_VERTEX_COLOR_ALBEDO);
}

/// `VERTEXLIGHT_ON` keyword on.
fn kw_VERTEXLIGHT_ON() -> bool {
    return kw_VERTEXLIGHT_ON_for_layout(XTOON_KEYWORD_LAYOUT_GENERIC);
}

/// `VERTEXLIGHT_ON` keyword on for a selected XSToon keyword layout.
fn kw_VERTEXLIGHT_ON_for_layout(keyword_layout: u32) -> bool {
    if (static_vertexlight_layout(keyword_layout)) {
        return xtoon_static_kw(XTOON_STATIC_KW_VERTEXLIGHT_ON);
    }
    return xtoon_kw(XTOON_KW_VERTEXLIGHT_ON);
}

/// True when the normal map should be sampled and applied.
fn normal_map_enabled() -> bool {
    return normal_map_enabled_for_layout(XTOON_KEYWORD_LAYOUT_GENERIC);
}

/// True when the normal map should be sampled and applied for a selected keyword layout.
fn normal_map_enabled_for_layout(keyword_layout: u32) -> bool {
    return kw_NORMAL_MAP_for_layout(keyword_layout);
}

/// True when the emission term should be evaluated for this material.
fn emission_map_enabled() -> bool {
    return emission_map_enabled_for_layout(XTOON_KEYWORD_LAYOUT_GENERIC);
}

/// True when the emission term should be evaluated for a selected keyword layout.
fn emission_map_enabled_for_layout(keyword_layout: u32) -> bool {
    return kw_EMISSION_MAP_for_layout(keyword_layout);
}

/// True when the metallic-gloss map should be sampled (expanded from `OCCLUSION_METALLIC`).
fn metallic_map_enabled() -> bool {
    return metallic_map_enabled_for_layout(XTOON_KEYWORD_LAYOUT_GENERIC);
}

/// True when the metallic-gloss map should be sampled for a selected keyword layout.
fn metallic_map_enabled_for_layout(keyword_layout: u32) -> bool {
    return kw_OCCLUSION_METALLIC_for_layout(keyword_layout);
}

/// True when the occlusion map should be sampled (expanded from `OCCLUSION_METALLIC`).
fn occlusion_enabled() -> bool {
    return occlusion_enabled_for_layout(XTOON_KEYWORD_LAYOUT_GENERIC);
}

/// True when the occlusion map should be sampled for a selected keyword layout.
fn occlusion_enabled_for_layout(keyword_layout: u32) -> bool {
    return kw_OCCLUSION_METALLIC_for_layout(keyword_layout);
}

/// True when the ramp selection mask should be sampled (expanded from `RAMPMASK_OUTLINEMASK_THICKNESS`).
fn ramp_mask_enabled() -> bool {
    return ramp_mask_enabled_for_layout(XTOON_KEYWORD_LAYOUT_GENERIC);
}

/// True when the ramp selection mask should be sampled for a selected keyword layout.
fn ramp_mask_enabled_for_layout(keyword_layout: u32) -> bool {
    return kw_RAMPMASK_OUTLINEMASK_THICKNESS_for_layout(keyword_layout);
}

/// True when the outline mask should be sampled (expanded from `RAMPMASK_OUTLINEMASK_THICKNESS`).
fn outline_mask_enabled() -> bool {
    return outline_mask_enabled_for_layout(XTOON_KEYWORD_LAYOUT_GENERIC);
}

/// True when the outline mask should be sampled for a selected keyword layout.
fn outline_mask_enabled_for_layout(keyword_layout: u32) -> bool {
    return kw_RAMPMASK_OUTLINEMASK_THICKNESS_for_layout(keyword_layout);
}

/// True when the thickness map should be sampled (expanded from `RAMPMASK_OUTLINEMASK_THICKNESS`).
fn thickness_enabled() -> bool {
    return thickness_enabled_for_layout(XTOON_KEYWORD_LAYOUT_GENERIC);
}

/// True when the thickness map should be sampled for a selected keyword layout.
fn thickness_enabled_for_layout(keyword_layout: u32) -> bool {
    return kw_RAMPMASK_OUTLINEMASK_THICKNESS_for_layout(keyword_layout);
}

/// True when matcap mode is selected via the `MATCAP` keyword. XSToon 2.0 has no active
/// `_ReflectionMode` enum path, leaving the `MATCAP` keyword as the only opt-in branch.
fn matcap_enabled() -> bool {
    return matcap_enabled_for_layout(XTOON_KEYWORD_LAYOUT_GENERIC);
}

/// True when matcap mode is selected for a selected keyword layout.
fn matcap_enabled_for_layout(keyword_layout: u32) -> bool {
    return kw_MATCAP_for_layout(keyword_layout);
}

/// True when the shader should use the skybox/PBR reflection branch by default whenever `MATCAP`
/// is not set.
fn reflection_uses_pbr() -> bool {
    return reflection_uses_pbr_for_layout(XTOON_KEYWORD_LAYOUT_GENERIC);
}

/// True when the shader should use the skybox/PBR reflection branch for a selected keyword layout.
fn reflection_uses_pbr_for_layout(keyword_layout: u32) -> bool {
    return !matcap_enabled_for_layout(keyword_layout);
}

/// True when vertex-color albedo tinting is enabled via the variant keyword.
fn vertex_color_albedo_enabled() -> bool {
    return vertex_color_albedo_enabled_for_layout(XTOON_KEYWORD_LAYOUT_GENERIC);
}

/// True when vertex-color albedo tinting is enabled for a selected keyword layout.
fn vertex_color_albedo_enabled_for_layout(keyword_layout: u32) -> bool {
    return kw_VERTEX_COLOR_ALBEDO_for_layout(keyword_layout);
}

/// Resolves the runtime alpha mode for shaders whose dispatcher pins
/// `XIEE_ALPHA_MODE = ALPHA_OPAQUE` and defers the decision to the variant bitmask.
/// Mirrors the precedence of the upstream `Cutout AlphaBlend Transparent` multi_compile group:
/// cutout wins over transparent which wins over alpha-blend.
fn resolved_alpha_mode_from_bits(static_alpha_mode: u32) -> u32 {
    return resolved_alpha_mode_from_bits_for_layout(static_alpha_mode, XTOON_KEYWORD_LAYOUT_GENERIC);
}

/// Resolves the runtime alpha mode for a selected XSToon keyword layout.
fn resolved_alpha_mode_from_bits_for_layout(static_alpha_mode: u32, keyword_layout: u32) -> u32 {
    if (static_alpha_mode != xb::ALPHA_OPAQUE) {
        return static_alpha_mode;
    }
    if (kw_Cutout_for_layout(keyword_layout)) {
        return xb::ALPHA_CUTOUT;
    }
    if (kw_Transparent_for_layout(keyword_layout)) {
        return xb::ALPHA_TRANSPARENT;
    }
    if (kw_AlphaBlend_for_layout(keyword_layout)) {
        return xb::ALPHA_FADE;
    }
    return xb::ALPHA_OPAQUE;
}
