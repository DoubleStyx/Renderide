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
/// Tests one keyword bit against the material's `_RenderideVariantBits`.
fn xtoon_kw(mask: u32) -> bool {
    return vb::enabled(xb::mat._RenderideVariantBits, mask);
}

/// True when the shader uses the static XSToon keyword layout.
fn static_vertexlight_layout(keyword_layout: u32) -> bool {
    return keyword_layout == XTOON_KEYWORD_LAYOUT_STATIC_VERTEXLIGHT;
}

/// `AlphaBlend` keyword on for a selected XSToon keyword layout.
fn kw_AlphaBlend_for_layout(keyword_layout: u32) -> bool {
    if (static_vertexlight_layout(keyword_layout)) {
        return false;
    }
    return xtoon_kw(XTOON_KW_ALPHABLEND);
}

/// `Cutout` keyword on for a selected XSToon keyword layout.
fn kw_Cutout_for_layout(keyword_layout: u32) -> bool {
    if (static_vertexlight_layout(keyword_layout)) {
        return false;
    }
    return xtoon_kw(XTOON_KW_CUTOUT);
}

/// `EMISSION_MAP` keyword on for a selected XSToon keyword layout.
fn kw_EMISSION_MAP_for_layout(keyword_layout: u32) -> bool {
    if (static_vertexlight_layout(keyword_layout)) {
        return false;
    }
    return xtoon_kw(XTOON_KW_EMISSION_MAP);
}

/// `MATCAP` keyword on for a selected XSToon keyword layout.
fn kw_MATCAP_for_layout(keyword_layout: u32) -> bool {
    if (static_vertexlight_layout(keyword_layout)) {
        return false;
    }
    return xtoon_kw(XTOON_KW_MATCAP);
}

/// `NORMAL_MAP` keyword on for a selected XSToon keyword layout.
fn kw_NORMAL_MAP_for_layout(keyword_layout: u32) -> bool {
    if (static_vertexlight_layout(keyword_layout)) {
        return false;
    }
    return xtoon_kw(XTOON_KW_NORMAL_MAP);
}

/// `OCCLUSION_METALLIC` keyword on for a selected XSToon keyword layout.
fn kw_OCCLUSION_METALLIC_for_layout(keyword_layout: u32) -> bool {
    if (static_vertexlight_layout(keyword_layout)) {
        return false;
    }
    return xtoon_kw(XTOON_KW_OCCLUSION_METALLIC);
}

/// `RAMPMASK_OUTLINEMASK_THICKNESS` keyword on for a selected XSToon keyword layout.
fn kw_RAMPMASK_OUTLINEMASK_THICKNESS_for_layout(keyword_layout: u32) -> bool {
    if (static_vertexlight_layout(keyword_layout)) {
        return false;
    }
    return xtoon_kw(XTOON_KW_RAMPMASK_OUTLINEMASK_THICKNESS);
}

/// `Transparent` keyword on for a selected XSToon keyword layout.
fn kw_Transparent_for_layout(keyword_layout: u32) -> bool {
    if (static_vertexlight_layout(keyword_layout)) {
        return false;
    }
    return xtoon_kw(XTOON_KW_TRANSPARENT);
}

/// `VERTEX_COLOR_ALBEDO` keyword on for a selected XSToon keyword layout.
fn kw_VERTEX_COLOR_ALBEDO_for_layout(keyword_layout: u32) -> bool {
    if (static_vertexlight_layout(keyword_layout)) {
        return false;
    }
    return xtoon_kw(XTOON_KW_VERTEX_COLOR_ALBEDO);
}

/// True when the normal map should be sampled and applied for a selected keyword layout.
fn normal_map_enabled_for_layout(keyword_layout: u32) -> bool {
    return kw_NORMAL_MAP_for_layout(keyword_layout);
}

/// True when the emission term should be evaluated for a selected keyword layout.
fn emission_map_enabled_for_layout(keyword_layout: u32) -> bool {
    return kw_EMISSION_MAP_for_layout(keyword_layout);
}

/// True when the metallic-gloss map should be sampled for a selected keyword layout.
fn metallic_map_enabled_for_layout(keyword_layout: u32) -> bool {
    return kw_OCCLUSION_METALLIC_for_layout(keyword_layout);
}

/// True when the occlusion map should be sampled for a selected keyword layout.
fn occlusion_enabled_for_layout(keyword_layout: u32) -> bool {
    return kw_OCCLUSION_METALLIC_for_layout(keyword_layout);
}

/// True when the ramp selection mask should be sampled for a selected keyword layout.
fn ramp_mask_enabled_for_layout(keyword_layout: u32) -> bool {
    return kw_RAMPMASK_OUTLINEMASK_THICKNESS_for_layout(keyword_layout);
}

/// True when the outline mask should be sampled for a selected keyword layout.
fn outline_mask_enabled_for_layout(keyword_layout: u32) -> bool {
    return kw_RAMPMASK_OUTLINEMASK_THICKNESS_for_layout(keyword_layout);
}

/// True when the thickness map should be sampled for a selected keyword layout.
fn thickness_enabled_for_layout(keyword_layout: u32) -> bool {
    return kw_RAMPMASK_OUTLINEMASK_THICKNESS_for_layout(keyword_layout);
}

/// True when matcap mode is selected for a selected keyword layout.
fn matcap_enabled_for_layout(keyword_layout: u32) -> bool {
    return kw_MATCAP_for_layout(keyword_layout);
}

/// True when the shader should use the skybox/PBR reflection branch for a selected keyword layout.
fn reflection_uses_pbr_for_layout(keyword_layout: u32) -> bool {
    return !matcap_enabled_for_layout(keyword_layout);
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
