//! Shader routing helpers for PhotonDust render-buffer draws.

/// Billboard/Unlit variant bit for CPU-expanded PhotonDust render-buffer geometry.
///
/// When set, `billboardunlit.wgsl` treats per-vertex point sizes as final particle sizes instead
/// of multiplying them by the material `_PointSize`.
pub(crate) const BILLBOARD_RENDER_BUFFER_ABSOLUTE_SIZE_BIT: u32 = 1u32 << 16;
/// Billboard/Unlit variant bit that enables per-particle color and alpha.
pub(crate) const BILLBOARD_RENDER_BUFFER_VERTEX_COLORS_BIT: u32 = 1u32 << 15;

/// Returns whether `stem` names an embedded Unlit-family shader other than Billboard/Unlit.
pub(crate) fn is_unlit_family_embedded_stem(stem: &str) -> bool {
    let lower = stem.to_ascii_lowercase();
    (lower.starts_with("unlit") || lower.contains("_unlit") || lower.contains("unlit_"))
        && !lower.contains("billboard")
}

/// Returns whether an embedded draw should remap Unlit keyword bits to Billboard/Unlit bits.
pub(crate) fn should_remap_unlit_variant_bits_for_billboard_draw(
    draw_stem: &str,
    source_shader_stem: Option<&str>,
) -> bool {
    draw_stem.starts_with("billboardunlit")
        && source_shader_stem.is_some_and(is_unlit_family_embedded_stem)
}

/// Enables render-buffer sizing and particle color semantics for synthetic billboard draws.
pub(crate) fn ensure_render_buffer_billboard_variant_bits(bits: u32) -> u32 {
    bits | BILLBOARD_RENDER_BUFFER_ABSOLUTE_SIZE_BIT | BILLBOARD_RENDER_BUFFER_VERTEX_COLORS_BIT
}

/// Remaps Froox Unlit keyword bits to Billboard/Unlit keyword bits for material binding.
pub(crate) fn remap_unlit_variant_bits_for_billboard(unlit_bits: u32) -> u32 {
    const PAIRS: &[(u32, u32)] = &[
        (0, 0),
        (1, 1),
        (2, 17),
        (3, 18),
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

#[cfg(test)]
mod tests {
    use super::*;

    const BILLBOARD_UNLIT_MASK_TEXTURE_CLIP_BIT: u32 = 1u32 << 17;
    const BILLBOARD_UNLIT_MASK_TEXTURE_MUL_BIT: u32 = 1u32 << 18;

    #[test]
    fn detects_unlit_family_stems() {
        assert!(is_unlit_family_embedded_stem("unlit_default"));
        assert!(is_unlit_family_embedded_stem("ui_unlit_default"));
        assert!(!is_unlit_family_embedded_stem("billboardunlit_default"));
    }

    #[test]
    fn remaps_unlit_texture_and_color_bits() {
        let unlit = (1u32 << 1) | (1u32 << 9);
        let billboard = remap_unlit_variant_bits_for_billboard(unlit);

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
    fn render_buffer_variant_bit_is_reserved() {
        assert_eq!(
            ensure_render_buffer_billboard_variant_bits(0),
            BILLBOARD_RENDER_BUFFER_ABSOLUTE_SIZE_BIT | BILLBOARD_RENDER_BUFFER_VERTEX_COLORS_BIT
        );
    }
}
