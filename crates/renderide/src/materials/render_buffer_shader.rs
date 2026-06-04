//! Shader routing helpers for PhotonDust render-buffer draws.

/// Billboard/Unlit variant bit for CPU-expanded PhotonDust render-buffer geometry.
///
/// When set, `billboardunlit.wgsl` treats per-vertex point sizes as final particle sizes instead
/// of multiplying them by the material `_PointSize`.
pub(crate) const BILLBOARD_RENDER_BUFFER_ABSOLUTE_SIZE_BIT: u32 = 1u32 << 16;
/// Billboard/Unlit variant bit that enables per-particle color and alpha.
pub(crate) const BILLBOARD_RENDER_BUFFER_VERTEX_COLORS_BIT: u32 = 1u32 << 15;
/// Billboard/Unlit variant bit that enables texture sampling.
pub(crate) const BILLBOARD_RENDER_BUFFER_TEXTURE_BIT: u32 = 1u32 << 10;
/// Billboard/Unlit variant bit that enables base color.
pub(crate) const BILLBOARD_RENDER_BUFFER_COLOR_BIT: u32 = 1u32 << 1;
/// Billboard variant bit that enables simple lighting for non-unlit source materials.
pub(crate) const BILLBOARD_RENDER_BUFFER_SIMPLE_LIT_BIT: u32 = 1u32 << 17;

const BILLBOARD_ALPHA_TEST_BIT: u32 = 1u32;

pub(crate) fn remap_variant_bits_for_billboard(stem: &str, source_bits: u32) -> u32 {
    if is_unlit_family_embedded_stem(stem) {
        BILLBOARD_RENDER_BUFFER_VERTEX_COLORS_BIT
            | remap_unlit_variant_bits_for_billboard(source_bits)
    } else {
        BILLBOARD_RENDER_BUFFER_TEXTURE_BIT
            | BILLBOARD_RENDER_BUFFER_COLOR_BIT
            | BILLBOARD_RENDER_BUFFER_SIMPLE_LIT_BIT
            | map_billboard_vertex_color_variant_bits(stem, source_bits)
            | map_billboard_alpha_clip_variant_bits(stem, source_bits)
    }
}

/// Enables render-buffer sizing and particle color semantics for synthetic billboard draws.
pub(crate) fn ensure_render_buffer_billboard_variant_bits(bits: u32) -> u32 {
    bits | BILLBOARD_RENDER_BUFFER_ABSOLUTE_SIZE_BIT
}

/// Returns whether `stem` names an embedded Unlit-family shader other than Billboard/Unlit.
fn is_unlit_family_embedded_stem(stem: &str) -> bool {
    let lower = stem.to_ascii_lowercase();
    (lower.starts_with("unlit") || lower.contains("_unlit") || lower.contains("unlit_"))
        && !lower.contains("billboard")
}

/// Remaps Froox Unlit keyword bits to Billboard/Unlit keyword bits for material binding.
fn remap_unlit_variant_bits_for_billboard(unlit_bits: u32) -> u32 {
    const PAIRS: &[(u32, u32)] = &[
        (0, 0),
        (1, 1),
        (2, 18),
        (3, 19),
        (4, 2),
        (5, 3),
        (6, 4),
        (7, 8),
        (8, 9),
        (9, 10),
        (11, 13),
        (12, 14),
        (13, 15),
    ];
    let mut out = 0u32;
    for &(from, to) in PAIRS {
        if unlit_bits & (1u32 << from) != 0 {
            out |= 1u32 << to;
        }
    }
    out
}

/// Remaps Froox AlphaClip or Cutoff keyword bits to Billboard/Unlit keyword bits
/// for material binding with non-Unlit materials.
fn map_billboard_alpha_clip_variant_bits(stem: &str, source_bits: u32) -> u32 {
    let lower = stem.to_ascii_lowercase();
    if lower.starts_with("xstoon2.0-cutout") {
        return BILLBOARD_ALPHA_TEST_BIT;
    }
    const ALPHA_CLIP_TWO: &[&str] = &["pbsmetallic", "pbsspecular"];
    if ALPHA_CLIP_TWO
        .iter()
        .any(|prefix| lower.starts_with(prefix))
    {
        return ((source_bits >> 2) & 1) * BILLBOARD_ALPHA_TEST_BIT;
    }
    const ALPHA_CLIP_ONE: &[&str] = &[
        "pbsdisplace",
        "pbsdualsided_",
        "pbsdualsidedspecular_",
        "pbslerp",
        "pbsslice_",
        "pbsslicespecular_",
        "pbsvertexcolor",
        "xstoon",
    ];
    if ALPHA_CLIP_ONE
        .iter()
        .any(|prefix| lower.starts_with(prefix))
    {
        return ((source_bits >> 1) & 1) * BILLBOARD_ALPHA_TEST_BIT;
    }
    const ALPHA_CLIP_ZERO: &[&str] = &["fresnel_", "pbsmultiuv", "reflection"];
    if ALPHA_CLIP_ZERO
        .iter()
        .any(|prefix| lower.starts_with(prefix))
    {
        return (source_bits & 1) * BILLBOARD_ALPHA_TEST_BIT;
    }
    0u32
}

/// Remaps Froox Vertex Colors keyword bits to Billboard/Unlit keyword bits
/// for material binding with non-Unlit materials.
fn map_billboard_vertex_color_variant_bits(stem: &str, source_bits: u32) -> u32 {
    let lower = stem.to_ascii_lowercase();
    let source_bit: Option<u32> = if lower.starts_with("fresnel_") {
        Some(10)
    } else if lower.starts_with("pbsdualsidedtransparent") {
        Some(5)
    } else if lower.starts_with("pbsvertexcolor") || lower.starts_with("pbsdualsided") {
        Some(6)
    } else if lower.starts_with("xstoon") {
        Some(8)
    } else {
        None
    };

    source_bit.map_or(0, |bit| {
        ((source_bits >> bit) & 1) * BILLBOARD_RENDER_BUFFER_VERTEX_COLORS_BIT
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const BILLBOARD_UNLIT_MASK_TEXTURE_CLIP_BIT: u32 = 1u32 << 18;
    const BILLBOARD_UNLIT_MASK_TEXTURE_MUL_BIT: u32 = 1u32 << 19;

    #[test]
    fn detects_unlit_family_stems() {
        assert!(is_unlit_family_embedded_stem("unlit_default"));
        assert!(is_unlit_family_embedded_stem("ui_unlit_default"));
        assert!(!is_unlit_family_embedded_stem("billboardunlit_default"));
    }

    #[test]
    fn remaps_unlit_texture_and_color_bits() {
        let unlit = (1u32 << 1) | (1u32 << 9);
        let billboard = remap_variant_bits_for_billboard("unlit_default", unlit);

        assert_eq!(billboard & (1u32 << 1), 1u32 << 1);
        assert_eq!(billboard & (1u32 << 10), 1u32 << 10);
        assert_eq!(billboard & (1u32 << 9), 0);
    }

    #[test]
    fn remaps_unlit_mask_bits_to_billboard_compatibility_bits() {
        let unlit = (1u32 << 2) | (1u32 << 3);
        let billboard = remap_unlit_variant_bits_for_billboard(unlit);

        assert_eq!(
            billboard & BILLBOARD_UNLIT_MASK_TEXTURE_CLIP_BIT,
            BILLBOARD_UNLIT_MASK_TEXTURE_CLIP_BIT
        );
        assert_eq!(
            billboard & BILLBOARD_UNLIT_MASK_TEXTURE_MUL_BIT,
            BILLBOARD_UNLIT_MASK_TEXTURE_MUL_BIT
        );
        assert_eq!(billboard & (1u32 << 2), 0);
        assert_eq!(billboard & (1u32 << 3), 0);
    }

    #[test]
    fn remaps_pbs_alphaclip_bits() {
        let pbs = (1u32 << 2) | (1u32 << 7);
        let billboard = remap_variant_bits_for_billboard("pbsmetallic_default", pbs);

        assert_eq!(
            billboard,
            BILLBOARD_RENDER_BUFFER_COLOR_BIT
                | BILLBOARD_RENDER_BUFFER_TEXTURE_BIT
                | BILLBOARD_RENDER_BUFFER_SIMPLE_LIT_BIT
                | 1u32
        );
    }

    #[test]
    fn remaps_pbs_no_alphaclip_bits() {
        let pbs = 1u32 << 7;
        let billboard = remap_variant_bits_for_billboard("pbsmetallic_default", pbs);

        assert_eq!(
            billboard,
            BILLBOARD_RENDER_BUFFER_COLOR_BIT
                | BILLBOARD_RENDER_BUFFER_TEXTURE_BIT
                | BILLBOARD_RENDER_BUFFER_SIMPLE_LIT_BIT
        );
    }

    #[test]
    fn remaps_supported_non_unlit_vertex_color_bits_to_billboard_vertex_colors() {
        for (stem, bit) in [
            ("fresnel_default", 10),
            ("pbsdualsidedtransparent_default", 5),
            ("pbsdualsidedspecular_default", 6),
            ("pbsvertexcolortransparent_default", 6),
            ("xstoon2.0_default", 8),
        ] {
            let billboard = remap_variant_bits_for_billboard(stem, 1u32 << bit);

            assert_eq!(
                billboard & BILLBOARD_RENDER_BUFFER_VERTEX_COLORS_BIT,
                BILLBOARD_RENDER_BUFFER_VERTEX_COLORS_BIT,
                "{stem}"
            );
            assert_eq!(
                billboard & BILLBOARD_RENDER_BUFFER_SIMPLE_LIT_BIT,
                BILLBOARD_RENDER_BUFFER_SIMPLE_LIT_BIT,
                "{stem}"
            );
        }
    }

    #[test]
    fn leaves_unsupported_non_unlit_vertex_color_bits_disabled() {
        let billboard = remap_variant_bits_for_billboard("pbsmetallic_default", 1u32 << 7);

        assert_eq!(billboard & BILLBOARD_RENDER_BUFFER_VERTEX_COLORS_BIT, 0);
        assert_eq!(
            billboard & BILLBOARD_RENDER_BUFFER_SIMPLE_LIT_BIT,
            BILLBOARD_RENDER_BUFFER_SIMPLE_LIT_BIT
        );
    }

    #[test]
    fn forces_cutout_for_xiexe_cutout_stems() {
        let billboard = remap_variant_bits_for_billboard("xstoon2.0-cutout_default", 0);

        assert_eq!(
            billboard & BILLBOARD_ALPHA_TEST_BIT,
            BILLBOARD_ALPHA_TEST_BIT
        );
    }

    #[test]
    fn render_buffer_variant_bit_is_reserved() {
        assert_eq!(
            ensure_render_buffer_billboard_variant_bits(0),
            BILLBOARD_RENDER_BUFFER_ABSOLUTE_SIZE_BIT
        );
    }
}
