//! Shared material-to-world-mesh-phase classification.

use crate::materials::{UNITY_RENDER_QUEUE_ALPHA_TEST, render_queue_is_transparent};
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
}

/// Classifies one material batch key into the world-mesh phase that records it.
pub(crate) fn classify_world_mesh_batch(key: &MaterialDrawBatchKey) -> WorldMeshBatchPhase {
    let intersect = key.embedded_requires_intersection_pass;
    let grab_pass = key.embedded_uses_scene_color_snapshot;
    let post_skybox = !intersect && !grab_pass && regular_window_records_after_skybox(key);
    let phase = phase_for_window(key, intersect, grab_pass, post_skybox);
    debug_assert!(
        !(intersect && grab_pass),
        "intersection and grab-pass subpasses are mutually exclusive"
    );

    WorldMeshBatchPhase {
        phase,
        post_skybox,
        grab_pass,
    }
}

/// Returns whether a regular forward draw must render after the skybox/background draw.
fn regular_window_records_after_skybox(key: &MaterialDrawBatchKey) -> bool {
    key.alpha_blended
        || render_queue_is_transparent(key.render_queue)
        || key.render_state.depth_write == Some(false)
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
    use crate::materials::{UNITY_RENDER_QUEUE_ALPHA_TEST, UNITY_RENDER_QUEUE_TRANSPARENT};
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

        let mut alpha_test = key(false);
        alpha_test.render_queue = UNITY_RENDER_QUEUE_ALPHA_TEST;
        assert_eq!(
            classify_world_mesh_batch(&alpha_test).phase,
            WorldMeshPhase::ForwardAlphaTest
        );
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
