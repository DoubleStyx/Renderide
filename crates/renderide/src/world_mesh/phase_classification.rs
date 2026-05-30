//! Shared material-to-world-mesh-phase classification.

use crate::materials::UNITY_RENDER_QUEUE_ALPHA_TEST;
use crate::world_mesh::MaterialDrawBatchKey;

use super::instances::WorldMeshPhase;

/// Phase classification for one material batch key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WorldMeshBatchPhase {
    /// Primary render phase that records this material batch.
    pub(crate) phase: WorldMeshPhase,
    /// Whether regular forward draws for this batch record after the skybox.
    pub(crate) post_skybox: bool,
    /// Whether this batch needs a scene-color snapshot immediately before drawing.
    pub(crate) grab_pass: bool,
    /// Whether draws in this phase must remain in strict order-sensitive submission.
    pub(crate) strict_order: bool,
}

/// Classifies one material batch key into the world-mesh phase that records it.
pub(crate) fn classify_world_mesh_batch(key: &MaterialDrawBatchKey) -> WorldMeshBatchPhase {
    let intersect = key.embedded_requires_intersection_pass;
    let grab_pass = key.embedded_uses_scene_color_snapshot;
    let post_skybox = !intersect && !grab_pass && key.records_after_skybox();
    let strict_order = grab_pass || (!intersect && key.requires_strict_order());
    let phase = phase_for_window(key, intersect, grab_pass, post_skybox);
    debug_assert!(
        !(intersect && grab_pass),
        "intersection and grab-pass subpasses are mutually exclusive"
    );

    WorldMeshBatchPhase {
        phase,
        post_skybox,
        grab_pass,
        strict_order,
    }
}

/// Selects the primary phase for one same-batch-key window.
fn phase_for_window(
    key: &MaterialDrawBatchKey,
    intersect: bool,
    grab_pass: bool,
    post_skybox: bool,
) -> WorldMeshPhase {
    if intersect {
        WorldMeshPhase::Intersection
    } else if grab_pass {
        WorldMeshPhase::TransparentGrab
    } else if post_skybox {
        WorldMeshPhase::Transparent
    } else if key.render_queue >= UNITY_RENDER_QUEUE_ALPHA_TEST {
        WorldMeshPhase::ForwardAlphaTest
    } else {
        WorldMeshPhase::ForwardOpaque
    }
}

#[cfg(test)]
mod tests {
    use crate::materials::{
        MaterialBlendMode, UNITY_RENDER_QUEUE_ALPHA_TEST, UNITY_RENDER_QUEUE_TRANSPARENT,
        UNITY_TRANSPARENT_RENDER_QUEUE_MIN,
    };
    use crate::world_mesh::WorldMeshPhase;
    use crate::world_mesh::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};

    use super::classify_world_mesh_batch;

    /// Builds a fixture batch key from a dummy draw item.
    fn key(alpha_blended: bool) -> crate::world_mesh::MaterialDrawBatchKey {
        dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 0,
            slot_index: 0,
            collect_order: 0,
            alpha_blended,
        })
        .batch_key
    }

    #[test]
    fn classifies_regular_opaque_and_alpha_test_phases() {
        let opaque = key(false);
        assert_eq!(
            classify_world_mesh_batch(&opaque).phase,
            WorldMeshPhase::ForwardOpaque
        );
        assert!(!classify_world_mesh_batch(&opaque).post_skybox);
        assert!(!classify_world_mesh_batch(&opaque).strict_order);

        let mut alpha_test = key(false);
        alpha_test.render_queue = UNITY_RENDER_QUEUE_ALPHA_TEST;
        assert_eq!(
            classify_world_mesh_batch(&alpha_test).phase,
            WorldMeshPhase::ForwardAlphaTest
        );
        assert!(!classify_world_mesh_batch(&alpha_test).post_skybox);
        assert!(!classify_world_mesh_batch(&alpha_test).strict_order);
    }

    #[test]
    fn classifies_geometry_last_queue_as_alpha_test_without_strict_order() {
        let mut late_opaque = key(false);
        late_opaque.render_queue = UNITY_TRANSPARENT_RENDER_QUEUE_MIN - 1;
        late_opaque.blend_mode = MaterialBlendMode::Opaque;

        let classification = classify_world_mesh_batch(&late_opaque);

        assert_eq!(classification.phase, WorldMeshPhase::ForwardAlphaTest);
        assert!(!classification.post_skybox);
        assert!(!classification.strict_order);
    }

    #[test]
    fn classifies_transparent_queue_as_strict_ordered_transparent() {
        let mut transparent = key(false);
        transparent.render_queue = UNITY_TRANSPARENT_RENDER_QUEUE_MIN;
        transparent.blend_mode = MaterialBlendMode::Opaque;

        let classification = classify_world_mesh_batch(&transparent);

        assert_eq!(classification.phase, WorldMeshPhase::Transparent);
        assert!(classification.post_skybox);
        assert!(classification.strict_order);
    }

    #[test]
    fn classifies_alpha_blend_at_transparent_boundary_as_strict_ordered() {
        let mut alpha = key(true);
        alpha.render_queue = UNITY_TRANSPARENT_RENDER_QUEUE_MIN;
        alpha.blend_mode = MaterialBlendMode::StemDefault;

        let classification = classify_world_mesh_batch(&alpha);

        assert_eq!(classification.phase, WorldMeshPhase::Transparent);
        assert!(classification.post_skybox);
        assert!(classification.strict_order);
    }

    #[test]
    fn classifies_special_tail_and_snapshot_phases() {
        let mut transparent = key(false);
        transparent.render_queue = UNITY_RENDER_QUEUE_TRANSPARENT;
        assert_eq!(
            classify_world_mesh_batch(&transparent).phase,
            WorldMeshPhase::Transparent
        );

        let mut grab = key(false);
        grab.embedded_uses_scene_color_snapshot = true;
        assert_eq!(
            classify_world_mesh_batch(&grab).phase,
            WorldMeshPhase::TransparentGrab
        );

        let mut intersect = key(false);
        intersect.embedded_requires_intersection_pass = true;
        assert_eq!(
            classify_world_mesh_batch(&intersect).phase,
            WorldMeshPhase::Intersection
        );
    }
}
