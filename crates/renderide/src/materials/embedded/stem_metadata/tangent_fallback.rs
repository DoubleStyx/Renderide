//! Lazy mesh-tangent upload policy for composed embedded WGSL stems.

#[cfg(test)]
use crate::materials::ShaderPermutation;
pub use crate::render_contract::EmbeddedTangentFallbackMode;

#[cfg(test)]
use super::EmbeddedStemQuery;

/// Tangent fallback policy for `base_stem` (independent of permutation).
pub(super) fn tangent_fallback_mode_for_stem(base_stem: &str) -> EmbeddedTangentFallbackMode {
    let stem = base_stem
        .strip_suffix("_default")
        .or_else(|| base_stem.strip_suffix("_multiview"))
        .unwrap_or(base_stem);
    if stem_needs_generated_missing_tangents(stem) {
        EmbeddedTangentFallbackMode::GenerateMissing
    } else {
        EmbeddedTangentFallbackMode::PreserveHostOrDefault
    }
}

fn stem_needs_generated_missing_tangents(stem: &str) -> bool {
    match stem {
        "pbsvoronoicrystal" | "pbsmetallic" | "pbsspecular" | "toonstandard" | "toonwater"
        | "matcap" | "fresnel" | "fresnellerp" | "overlayfresnel" => true,
        "pbsdisplaceshadow" => false,
        _ => {
            stem.starts_with("pbsmultiuv")
                || stem.starts_with("pbslerp")
                || stem.starts_with("pbsrim")
                || stem.starts_with("pbsintersect")
                || stem.starts_with("pbsdualsided")
                || stem.starts_with("pbsvertexcolortransparent")
                || stem.starts_with("pbscolormask")
                || stem.starts_with("pbscolorsplat")
                || stem.starts_with("pbsslice")
                || stem.starts_with("pbsstencil")
                || stem.starts_with("pbsdisplace")
                || stem.starts_with("pbsdistancelerp")
                || stem.starts_with("xstoon2.0")
        }
    }
}

/// Tangent fallback policy for lazy mesh tangent upload.
#[cfg(test)]
pub fn embedded_stem_tangent_fallback_mode(
    base_stem: &str,
    permutation: ShaderPermutation,
) -> EmbeddedTangentFallbackMode {
    EmbeddedStemQuery::for_stem(base_stem, permutation).tangent_fallback_mode()
}

#[cfg(test)]
mod tests {
    use crate::materials::SHADER_PERM_MULTIVIEW_STEREO;
    use crate::materials::ShaderPermutation;

    use super::{EmbeddedTangentFallbackMode, embedded_stem_tangent_fallback_mode};
    use crate::materials::embedded::stem_metadata::vertex_streams::{
        embedded_stem_needs_tangent_stream, embedded_stem_needs_uv0_stream,
    };

    #[test]
    fn tbn_materials_request_generated_missing_tangents() {
        let mono = ShaderPermutation(0);

        assert!(embedded_stem_needs_uv0_stream(
            "pbsvoronoicrystal_default",
            mono
        ));
        assert!(embedded_stem_needs_tangent_stream(
            "pbsvoronoicrystal_default",
            mono
        ));
        assert_eq!(
            embedded_stem_tangent_fallback_mode("pbsvoronoicrystal_default", mono),
            EmbeddedTangentFallbackMode::GenerateMissing
        );
        assert_eq!(
            embedded_stem_tangent_fallback_mode(
                "pbsvoronoicrystal_default",
                SHADER_PERM_MULTIVIEW_STEREO
            ),
            EmbeddedTangentFallbackMode::GenerateMissing
        );

        for stem in [
            "pbsmetallic_default",
            "pbsspecular_default",
            "pbsmultiuv_default",
            "pbsmultiuvspecular_default",
            "pbslerp_default",
            "pbsrimtransparentzwrite_default",
            "pbsintersectspecular_default",
            "pbsdualsidedtransparent_default",
            "pbsvertexcolortransparentspecular_default",
            "pbscolormaskspecular_default",
            "pbscolorsplatspecular_default",
            "pbsslicetransparent_default",
            "pbsstencilspecular_default",
            "pbsdisplacespeculartransparent_default",
            "pbsdistancelerpspeculartransparent_default",
            "toonstandard_default",
            "toonwater_default",
            "matcap_default",
            "fresnel_default",
            "fresnellerp_default",
            "overlayfresnel_default",
            "xstoon2.0_default",
            "xstoon2.0-cutouta2c-outlined_default",
            "xstoon2.0_outlined_default",
        ] {
            assert_eq!(
                embedded_stem_tangent_fallback_mode(stem, mono),
                EmbeddedTangentFallbackMode::GenerateMissing,
                "{stem}"
            );
        }
    }

    #[test]
    fn non_tbn_tangent_payload_materials_do_not_generate_missing_tangents() {
        let mono = ShaderPermutation(0);

        assert!(embedded_stem_needs_tangent_stream(
            "ui_circlesegment_default",
            mono
        ));
        assert!(embedded_stem_needs_tangent_stream(
            "billboardunlit_default",
            mono
        ));
        assert!(embedded_stem_needs_tangent_stream("ui_unlit_default", mono));
        assert_eq!(
            embedded_stem_tangent_fallback_mode("billboardunlit_default", mono),
            EmbeddedTangentFallbackMode::PreserveHostOrDefault
        );
        assert_eq!(
            embedded_stem_tangent_fallback_mode("ui_circlesegment_default", mono),
            EmbeddedTangentFallbackMode::PreserveHostOrDefault
        );
        assert_eq!(
            embedded_stem_tangent_fallback_mode("ui_unlit_default", mono),
            EmbeddedTangentFallbackMode::PreserveHostOrDefault
        );

        for stem in [
            "pbstriplanar_default",
            "pbstriplanarspecular_default",
            "pbstriplanartransparent_default",
            "pbstriplanartransparentspecular_default",
            "pbsdisplaceshadow_default",
            "blur_default",
            "refract_default",
            "reflection_default",
            "unlit_default",
        ] {
            assert_eq!(
                embedded_stem_tangent_fallback_mode(stem, mono),
                EmbeddedTangentFallbackMode::PreserveHostOrDefault,
                "{stem}"
            );
        }
    }
}
