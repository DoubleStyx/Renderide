//! Unity-style shadow-caster participation for embedded raster materials.

use std::sync::Arc;

use crate::materials::RasterPipelineKind;

/// Shadow-caster pipeline policy for a material draw.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ShadowCasterPolicy {
    /// The material has no Unity shadow-caster pass or fallback.
    None,
    /// The material may be rendered through the fast position-only depth caster.
    DepthOnly,
}

impl ShadowCasterPolicy {
    /// Returns whether this policy casts into shadow maps.
    #[inline]
    pub(crate) const fn casts(self) -> bool {
        !matches!(self, Self::None)
    }
}

/// Returns the shadow-caster policy for a resolved raster pipeline kind.
pub(crate) fn shadow_caster_policy_for_pipeline(kind: &RasterPipelineKind) -> ShadowCasterPolicy {
    match kind {
        RasterPipelineKind::Null => ShadowCasterPolicy::DepthOnly,
        RasterPipelineKind::EmbeddedStem(stem) => shadow_caster_policy_for_embedded_stem(stem),
    }
}

fn shadow_caster_policy_for_embedded_stem(stem: &Arc<str>) -> ShadowCasterPolicy {
    let base = base_stem(stem.as_ref());
    if embedded_stem_casts_shadows(base) {
        ShadowCasterPolicy::DepthOnly
    } else {
        ShadowCasterPolicy::None
    }
}

fn base_stem(stem: &str) -> &str {
    stem.strip_suffix("_default")
        .or_else(|| stem.strip_suffix("_multiview"))
        .unwrap_or(stem)
}

fn embedded_stem_casts_shadows(base_stem: &str) -> bool {
    matches!(base_stem, "newunlitshader")
        || base_stem.starts_with("pbs")
        || base_stem.starts_with("xstoon2.0")
        || base_stem.starts_with("furfx-")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn embedded(stem: &'static str) -> RasterPipelineKind {
        RasterPipelineKind::EmbeddedStem(Arc::from(stem))
    }

    #[test]
    fn common_unlit_does_not_cast_without_shadowcaster_or_fallback() {
        assert_eq!(
            shadow_caster_policy_for_pipeline(&embedded("unlit_default")),
            ShadowCasterPolicy::None
        );
        assert_eq!(
            shadow_caster_policy_for_pipeline(&embedded("unlitdistancelerp_default")),
            ShadowCasterPolicy::None
        );
    }

    #[test]
    fn new_unlit_casts_through_diffuse_fallback() {
        assert_eq!(
            shadow_caster_policy_for_pipeline(&embedded("newunlitshader_default")),
            ShadowCasterPolicy::DepthOnly
        );
    }

    #[test]
    fn pbs_xiexe_and_fur_families_cast() {
        for stem in [
            "pbsmetallic_default",
            "pbscolormask_default",
            "pbsdistancelerp_default",
            "xstoon2.0_default",
            "xstoon2.0-cutout_default",
            "furfx-2.0-10layer_default",
        ] {
            assert_eq!(
                shadow_caster_policy_for_pipeline(&embedded(stem)),
                ShadowCasterPolicy::DepthOnly,
                "{stem}"
            );
        }
    }
}
