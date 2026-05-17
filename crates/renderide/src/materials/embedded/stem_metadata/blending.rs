//! Alpha-blending and scene-snapshot flag queries on composed embedded WGSL stems.

use crate::materials::ShaderPermutation;

use super::EmbeddedStemQuery;

/// `true` when the embedded material stem declares alpha blending in any `//#pass` directive.
pub fn embedded_stem_uses_alpha_blending(base_stem: &str) -> bool {
    EmbeddedStemQuery::for_stem(base_stem, ShaderPermutation(0)).uses_alpha_blending()
}

/// `true` when the embedded material stem declares a blended pass that writes depth by default.
pub fn embedded_stem_uses_blended_depth_write(
    base_stem: &str,
    permutation: ShaderPermutation,
) -> bool {
    EmbeddedStemQuery::for_stem(base_stem, permutation).uses_blended_depth_write()
}

/// `true` when the embedded material stem declares blended front/back cull passes.
pub fn embedded_stem_uses_two_sided_transparency(
    base_stem: &str,
    permutation: ShaderPermutation,
) -> bool {
    EmbeddedStemQuery::for_stem(base_stem, permutation).uses_two_sided_transparency()
}

/// `true` when the composed embedded target declares a scene-depth snapshot binding.
pub fn embedded_stem_uses_scene_depth_snapshot(
    base_stem: &str,
    permutation: ShaderPermutation,
) -> bool {
    EmbeddedStemQuery::for_stem(base_stem, permutation)
        .snapshot_requirements()
        .uses_scene_depth
}

/// `true` when the composed embedded target declares a scene-color snapshot binding.
pub fn embedded_stem_uses_scene_color_snapshot(
    base_stem: &str,
    permutation: ShaderPermutation,
) -> bool {
    EmbeddedStemQuery::for_stem(base_stem, permutation)
        .snapshot_requirements()
        .uses_scene_color
}

#[cfg(test)]
mod tests {
    use crate::materials::SHADER_PERM_MULTIVIEW_STEREO;
    use crate::materials::ShaderPermutation;

    use super::{
        embedded_stem_uses_alpha_blending, embedded_stem_uses_blended_depth_write,
        embedded_stem_uses_scene_color_snapshot, embedded_stem_uses_scene_depth_snapshot,
        embedded_stem_uses_two_sided_transparency,
    };
    use crate::materials::embedded::stem_metadata::{
        embedded_composed_stem_for_permutation, embedded_stem_requires_intersection_pass,
    };

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
