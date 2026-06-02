//! Shader routing helpers for PhotonDust render-buffer draws.

/// Billboard/Unlit variant bit for CPU-expanded PhotonDust render-buffer geometry.
///
/// When set, `billboardunlit.wgsl` treats per-vertex point sizes as final particle sizes instead
/// of multiplying them by the material `_PointSize`.
pub(crate) const BILLBOARD_RENDER_BUFFER_ABSOLUTE_SIZE_BIT: u32 = 1u32 << 18;
/// Billboard/Unlit variant bit that enables per-particle color and alpha.
pub(crate) const BILLBOARD_RENDER_BUFFER_VERTEX_COLORS_BIT: u32 = 1u32 << 17;
/// Billboard/Unlit variant bit that enables texture sampling.
pub(crate) const BILLBOARD_RENDER_BUFFER_TEXTURE_BIT: u32 = 1u32 << 12;
/// Billboard/Unlit variant bit that enables base color.
pub(crate) const BILLBOARD_RENDER_BUFFER_COLOR_BIT: u32 = 1u32 << 1;
/// Billboard variant bit that enables simple lighting for non-unlit source materials.
pub(crate) const BILLBOARD_RENDER_BUFFER_SIMPLE_LIT_BIT: u32 = 1u32 << 19;

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
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
        (6, 6),
        (7, 10),
        (8, 11),
        (9, 12),
        (11, 15),
        (12, 16),
        (13, 17),
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
        return 1;
    }
    const ALPHA_CLIP_TWO: &[&str] = &["pbsmetallic", "pbsspecular"];
    if ALPHA_CLIP_TWO
        .iter()
        .any(|prefix| lower.starts_with(prefix))
    {
        return (source_bits >> 2) & 1;
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
        return (source_bits >> 1) & 1;
    }
    const ALPHA_CLIP_ZERO: &[&str] = &["fresnel_", "pbsmultiuv", "reflection"];
    if ALPHA_CLIP_ZERO
        .iter()
        .any(|prefix| lower.starts_with(prefix))
    {
        return source_bits & 1;
    }
    0u32
}

/// Remaps Froox Vertex Colors keyword bits to Billboard/Unlit keyword bits
/// for material binding with non-Unlit materials.
fn map_billboard_vertex_color_variant_bits(stem: &str, source_bits: u32) -> u32 {
    let lower = stem.to_ascii_lowercase();
    if lower.starts_with("fresnel_") {
        return ((source_bits >> 10) & 1) << 17;
    }
    if lower.starts_with("pbsdualsidedtransparent") {
        return ((source_bits >> 5) & 1) << 17;
    }
    if lower.starts_with("pbsvertexcolor") || lower.starts_with("pbsdualsided") {
        return ((source_bits >> 6) & 1) << 17;
    }
    if lower.starts_with("xstoon") {
        return ((source_bits >> 8) & 1) << 17;
    }
    0u32
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(billboard & (1u32 << 12), 1u32 << 12);
        assert_eq!(billboard & (1u32 << 11), 0);
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
    fn render_buffer_variant_bit_is_reserved() {
        assert_eq!(
            ensure_render_buffer_billboard_variant_bits(0),
            BILLBOARD_RENDER_BUFFER_ABSOLUTE_SIZE_BIT
        );
    }
}
