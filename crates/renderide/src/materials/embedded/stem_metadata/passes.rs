//! Pass-count and depth-prepass queries on composed embedded WGSL stems.

use crate::materials::{MaterialPassDesc, ShaderPermutation};

use super::EmbeddedStemQuery;

/// Number of raster passes that will be submitted for one embedded draw batch.
pub fn embedded_stem_pipeline_pass_count(base_stem: &str, permutation: ShaderPermutation) -> usize {
    EmbeddedStemQuery::for_stem(base_stem, permutation).pipeline_pass_count()
}

/// `true` when the composed embedded target declares intersection-depth material behavior.
#[cfg(test)]
pub fn embedded_stem_requires_intersection_pass(
    base_stem: &str,
    permutation: ShaderPermutation,
) -> bool {
    EmbeddedStemQuery::for_stem(base_stem, permutation)
        .snapshot_requirements()
        .requires_intersection_pass
}

/// Returns the material pass that the generic world-mesh depth prepass may safely mirror.
pub fn embedded_stem_depth_prepass_pass(
    base_stem: &str,
    permutation: ShaderPermutation,
) -> Option<MaterialPassDesc> {
    EmbeddedStemQuery::for_stem(base_stem, permutation).depth_prepass_pass()
}

#[cfg(test)]
mod tests {
    use crate::embedded_shaders;
    use crate::materials::MaterialPassState;

    #[test]
    fn first_shader_batch_fixed_state_stems_keep_expected_passes() {
        let circle = embedded_shaders::embedded_target_passes("circle_default");
        assert_eq!(circle.len(), 1);
        assert_eq!(circle[0].name, "transparent_rgb");
        assert_eq!(circle[0].material_state, MaterialPassState::Static);
        assert_eq!(circle[0].write_mask, wgpu::ColorWrites::COLOR);
        assert!(!circle[0].depth_write);
        assert_eq!(circle[0].cull_mode, None);
        assert!(circle[0].blend.is_some());

        let depth_projection = embedded_shaders::embedded_target_passes("depthprojection_default");
        assert_eq!(depth_projection.len(), 1);
        assert_eq!(depth_projection[0].name, "forward_two_sided");
        assert_eq!(
            depth_projection[0].material_state,
            MaterialPassState::Forward
        );
        assert_eq!(depth_projection[0].cull_mode, None);
    }
}
