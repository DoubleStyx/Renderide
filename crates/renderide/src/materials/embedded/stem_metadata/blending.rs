//! Alpha-blending and scene-snapshot flag queries on composed embedded WGSL stems.

#[cfg(test)]
use crate::materials::{SceneColorSnapshotMode, ShaderPermutation};

#[cfg(test)]
use super::EmbeddedStemQuery;

/// `true` when the embedded material stem declares alpha blending in any `//#pass` directive.
#[cfg(test)]
pub fn embedded_stem_uses_alpha_blending(base_stem: &str) -> bool {
    EmbeddedStemQuery::for_stem(base_stem, ShaderPermutation(0)).uses_alpha_blending()
}

/// `true` when the embedded material stem declares a blended pass that writes depth by default.
#[cfg(test)]
pub fn embedded_stem_uses_blended_depth_write(
    base_stem: &str,
    permutation: ShaderPermutation,
) -> bool {
    EmbeddedStemQuery::for_stem(base_stem, permutation).uses_blended_depth_write()
}

/// `true` when the embedded material stem declares blended front/back cull passes.
#[cfg(test)]
pub fn embedded_stem_uses_two_sided_transparency(
    base_stem: &str,
    permutation: ShaderPermutation,
) -> bool {
    EmbeddedStemQuery::for_stem(base_stem, permutation).uses_two_sided_transparency()
}

/// `true` when the composed embedded target declares a scene-depth snapshot binding.
#[cfg(test)]
pub fn embedded_stem_uses_scene_depth_snapshot(
    base_stem: &str,
    permutation: ShaderPermutation,
) -> bool {
    EmbeddedStemQuery::for_stem(base_stem, permutation)
        .snapshot_requirements()
        .uses_scene_depth
}

/// `true` when the composed embedded target declares a scene-color snapshot binding.
#[cfg(test)]
pub fn embedded_stem_uses_scene_color_snapshot(
    base_stem: &str,
    permutation: ShaderPermutation,
) -> bool {
    EmbeddedStemQuery::for_stem(base_stem, permutation)
        .snapshot_requirements()
        .uses_scene_color
}

/// How the embedded material expects the scene-color snapshot to be refreshed.
#[cfg(test)]
pub fn embedded_stem_scene_color_snapshot_mode(
    base_stem: &str,
    permutation: ShaderPermutation,
) -> SceneColorSnapshotMode {
    EmbeddedStemQuery::for_stem(base_stem, permutation).scene_color_snapshot_mode()
}

#[cfg(test)]
mod tests {
    use crate::materials::ShaderPermutation;
    use crate::materials::{
        MaterialDepthCompareDomain, MaterialPassState, MaterialRenderStatePolicy,
        SHADER_PERM_MULTIVIEW_STEREO, SceneColorSnapshotMode,
    };

    use super::{
        embedded_stem_scene_color_snapshot_mode, embedded_stem_uses_alpha_blending,
        embedded_stem_uses_blended_depth_write, embedded_stem_uses_scene_color_snapshot,
        embedded_stem_uses_scene_depth_snapshot, embedded_stem_uses_two_sided_transparency,
    };
    use crate::materials::embedded::stem_metadata::embedded_composed_stem_for_permutation;
    use crate::materials::embedded::stem_metadata::passes::embedded_stem_requires_intersection_pass;

    const FILTER_STEMS: &[&str] = &[
        "blur_default",
        "blur_perobject_default",
        "channelmatrix_default",
        "channelmatrix_perobject_default",
        "gamma_default",
        "gamma_perobject_default",
        "grayscale_default",
        "grayscale_perobject_default",
        "hsv_default",
        "hsv_perobject_default",
        "invert_default",
        "invert_perobject_default",
        "lut_default",
        "lut_perobject_default",
        "pixelate_default",
        "pixelate_perobject_default",
        "posterize_default",
        "posterize_perobject_default",
        "refract_default",
        "refract_perobject_default",
        "threshold_default",
        "threshold_perobject_default",
    ];

    #[test]
    fn metadata_flags_cover_snapshot_and_intersection_material_classes() {
        let mono = ShaderPermutation(0);

        assert!(embedded_stem_uses_scene_color_snapshot(
            "blur_default",
            mono
        ));
        assert!(embedded_stem_uses_scene_color_snapshot(
            "blur_default",
            SHADER_PERM_MULTIVIEW_STEREO
        ));
        assert!(!embedded_stem_requires_intersection_pass(
            "blur_default",
            mono
        ));
        assert!(embedded_stem_uses_scene_depth_snapshot(
            "blur_default",
            mono
        ));

        assert!(embedded_stem_uses_scene_color_snapshot(
            "refract_default",
            mono
        ));
        assert!(!embedded_stem_requires_intersection_pass(
            "refract_default",
            mono
        ));
        assert!(embedded_stem_uses_scene_depth_snapshot(
            "refract_default",
            mono
        ));

        assert!(embedded_stem_requires_intersection_pass(
            "pbsintersect_default",
            mono
        ));
        assert!(!embedded_stem_uses_scene_color_snapshot(
            "pbsintersect_default",
            mono
        ));
        assert!(embedded_stem_uses_scene_depth_snapshot(
            "pbsintersect_default",
            mono
        ));
    }

    #[test]
    fn filter_materials_are_single_forward_filter_scene_color_passes() {
        for &stem in FILTER_STEMS {
            for permutation in [ShaderPermutation(0), SHADER_PERM_MULTIVIEW_STEREO] {
                assert!(
                    embedded_stem_uses_scene_color_snapshot(stem, permutation),
                    "{stem:?} should sample scene color for permutation {permutation:?}",
                );
                assert!(
                    !embedded_stem_requires_intersection_pass(stem, permutation),
                    "{stem:?} should not route through the intersection subpass",
                );

                let composed = embedded_composed_stem_for_permutation(stem, permutation);
                let passes = crate::embedded_shaders::embedded_target_passes(&composed);
                assert_eq!(
                    passes.len(),
                    1,
                    "{composed:?} should declare exactly one raster pass",
                );
                assert_eq!(
                    passes[0].name, "forward_filter",
                    "{composed:?} should draw through the filter pass",
                );
            }
        }
    }

    #[test]
    fn scene_color_snapshot_modes_distinguish_named_and_per_object_grabs() {
        let mono = ShaderPermutation(0);

        for stem in ["blur_default", "pixelate_default", "refract_default"] {
            assert_eq!(
                embedded_stem_scene_color_snapshot_mode(stem, mono),
                SceneColorSnapshotMode::NamedBackgroundGrab,
                "{stem}"
            );
            assert_eq!(
                embedded_stem_scene_color_snapshot_mode(stem, SHADER_PERM_MULTIVIEW_STEREO),
                SceneColorSnapshotMode::NamedBackgroundGrab,
                "{stem}"
            );
        }

        for stem in [
            "blur_perobject_default",
            "pixelate_perobject_default",
            "posterize_default",
            "posterize_perobject_default",
        ] {
            assert_eq!(
                embedded_stem_scene_color_snapshot_mode(stem, mono),
                SceneColorSnapshotMode::PerObjectGrab,
                "{stem}"
            );
        }

        assert_eq!(
            embedded_stem_scene_color_snapshot_mode("pbsmetallic_default", mono),
            SceneColorSnapshotMode::None
        );
    }

    /// Asserts the source-authored filter pass fallback state for one stem.
    fn assert_source_filter_pass_fallbacks(stem: &str, expected_depth_write: bool) {
        for permutation in [ShaderPermutation(0), SHADER_PERM_MULTIVIEW_STEREO] {
            let composed = embedded_composed_stem_for_permutation(stem, permutation);
            let passes = crate::embedded_shaders::embedded_target_passes(&composed);
            assert_eq!(
                passes.len(),
                1,
                "{composed:?} should declare exactly one raster pass",
            );

            let pass = &passes[0];
            assert_eq!(pass.name, "forward_filter", "{composed:?}");
            assert_eq!(
                pass.material_state,
                MaterialPassState::Filter,
                "{composed:?}"
            );
            assert_eq!(pass.write_mask, wgpu::ColorWrites::ALL, "{composed:?}");
            assert_eq!(pass.depth_write, expected_depth_write, "{composed:?}");
            assert_eq!(
                pass.depth_compare,
                crate::gpu::MAIN_FORWARD_DEPTH_COMPARE,
                "{composed:?}",
            );
            assert_eq!(
                pass.depth_compare_domain,
                MaterialDepthCompareDomain::FrooxZTest,
                "{composed:?}",
            );
            assert_eq!(pass.cull_mode, Some(wgpu::Face::Back), "{composed:?}");
            assert_eq!(
                pass.render_state_policy,
                MaterialRenderStatePolicy::ALL_MATERIAL,
                "{composed:?}",
            );
        }
    }

    /// Verifies filter materials in the 31-40 parity batch preserve ShaderLab fallback state.
    #[test]
    fn filter_parity_stems_keep_source_fallback_render_state() {
        for stem in ["gamma_default", "gamma_perobject_default"] {
            assert_source_filter_pass_fallbacks(stem, true);
            assert!(
                embedded_stem_uses_scene_color_snapshot(stem, ShaderPermutation(0)),
                "{stem:?} should sample the scene-color snapshot",
            );
        }

        assert_source_filter_pass_fallbacks("getdepth_default", false);
        assert!(
            embedded_stem_uses_scene_depth_snapshot("getdepth_default", ShaderPermutation(0)),
            "getdepth_default should sample the scene-depth snapshot",
        );
    }

    #[test]
    fn named_background_grabpass_stems_are_distinct_from_per_object_wrappers() {
        let mono = ShaderPermutation(0);

        assert_eq!(
            embedded_stem_scene_color_snapshot_mode("blur_default", mono),
            SceneColorSnapshotMode::NamedBackgroundGrab
        );
        assert_eq!(
            embedded_stem_scene_color_snapshot_mode("blur_perobject_default", mono),
            SceneColorSnapshotMode::PerObjectGrab
        );
        assert_eq!(
            embedded_stem_scene_color_snapshot_mode("channelmatrix_default", mono),
            SceneColorSnapshotMode::PerObjectGrab
        );
        assert_eq!(
            embedded_stem_scene_color_snapshot_mode("pbsintersect_default", mono),
            SceneColorSnapshotMode::None
        );
    }

    #[test]
    fn volume_materials_use_depth_snapshot_without_grab_snapshot() {
        let mono = ShaderPermutation(0);

        for stem in ["fogboxvolume_default", "volumeunlit_default"] {
            assert!(embedded_stem_uses_alpha_blending(stem), "{stem}");
            assert!(
                embedded_stem_uses_scene_depth_snapshot(stem, mono),
                "{stem}"
            );
            assert!(
                !embedded_stem_uses_scene_color_snapshot(stem, mono),
                "{stem}"
            );
            assert!(
                !embedded_stem_requires_intersection_pass(stem, mono),
                "{stem}"
            );
        }
    }

    #[test]
    fn transparent_depth_write_and_two_sided_metadata_are_exposed() {
        let mono = ShaderPermutation(0);

        assert!(embedded_stem_uses_blended_depth_write(
            "furfx-basic-10layer_default",
            mono
        ));
        assert!(!embedded_stem_uses_two_sided_transparency(
            "furfx-basic-10layer_default",
            mono
        ));

        assert!(embedded_stem_uses_two_sided_transparency(
            "pbsdualsidedtransparent_default",
            mono
        ));
        assert!(!embedded_stem_uses_blended_depth_write(
            "pbsdualsidedtransparent_default",
            mono
        ));
    }
}
