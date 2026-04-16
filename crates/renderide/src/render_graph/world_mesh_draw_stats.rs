//! Batch and draw counters for the debug HUD (aligned with sorted [`WorldMeshDrawItem`] order).

use super::world_mesh_draw_prep::{
    build_instance_batches, MaterialDrawBatchKey, WorldMeshDrawItem,
};

/// Draw and batch counts for the debug HUD (aligned with sorted [`WorldMeshDrawItem`] order).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorldMeshDrawStats {
    /// Distinct `(batch_key, overlay)` groups after sorting.
    pub batch_total: usize,
    /// Batches for non-overlay draws only.
    pub batch_main: usize,
    /// Batches for overlay draws only.
    pub batch_overlay: usize,
    /// Total indexed draws submitted.
    pub draws_total: usize,
    /// Draws in the main (non-overlay) layer.
    pub draws_main: usize,
    /// Draws in the overlay layer.
    pub draws_overlay: usize,
    /// Non-skinned mesh draws.
    pub rigid_draws: usize,
    /// Skinned mesh draws.
    pub skinned_draws: usize,
    /// Slots that went through frustum culling before the final draw list (if culling was enabled).
    pub draws_pre_cull: usize,
    /// Draws removed by frustum culling.
    pub draws_culled: usize,
    /// Draws removed by Hi-Z occlusion when enabled.
    pub draws_hi_z_culled: usize,
    /// GPU instance batches after merge (one indexed draw each); at most `draws_total`.
    pub instance_batch_total: usize,
}

/// Computes batch boundaries from material/property-block/skin/overlay changes after sorting.
///
/// `supports_base_instance` should match the forward pass (see [`crate::render_graph::passes::WorldMeshForwardPass`])
/// so [`WorldMeshDrawStats::instance_batch_total`] reflects the same merge policy.
pub fn world_mesh_draw_stats_from_sorted(
    draws: &[WorldMeshDrawItem],
    cull: Option<(usize, usize, usize)>,
    supports_base_instance: bool,
) -> WorldMeshDrawStats {
    let draws_total = draws.len();
    let draws_main = draws.iter().filter(|d| !d.is_overlay).count();
    let draws_overlay = draws_total - draws_main;
    let rigid_draws = draws.iter().filter(|d| !d.skinned).count();
    let skinned_draws = draws_total - rigid_draws;

    let mut batch_total = 0usize;
    let mut batch_main = 0usize;
    let mut batch_overlay = 0usize;
    let mut prev: Option<(MaterialDrawBatchKey, bool)> = None;
    for d in draws {
        let cur = (d.batch_key.clone(), d.is_overlay);
        let same_as_prev = prev
            .as_ref()
            .is_some_and(|(k, o)| k == &d.batch_key && *o == d.is_overlay);
        if !same_as_prev {
            batch_total += 1;
            if d.is_overlay {
                batch_overlay += 1;
            } else {
                batch_main += 1;
            }
            prev = Some(cur);
        }
    }

    let (draws_pre_cull, draws_culled, draws_hi_z_culled) = cull.unwrap_or((0, 0, 0));

    let draw_indices: Vec<usize> = (0..draws.len()).collect();
    let instance_batch_total =
        build_instance_batches(draws, &draw_indices, supports_base_instance).len();

    WorldMeshDrawStats {
        batch_total,
        batch_main,
        batch_overlay,
        draws_total,
        draws_main,
        draws_overlay,
        rigid_draws,
        skinned_draws,
        draws_pre_cull,
        draws_culled,
        draws_hi_z_culled,
        instance_batch_total,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render_graph::test_fixtures::{dummy_world_mesh_draw_item, DummyDrawItemSpec};

    #[test]
    fn world_mesh_draw_stats_empty() {
        let s = world_mesh_draw_stats_from_sorted(&[], None, true);
        assert_eq!(s.batch_total, 0);
        assert_eq!(s.draws_total, 0);
        assert_eq!(s.instance_batch_total, 0);
    }

    #[test]
    fn world_mesh_draw_stats_single_batch() {
        let a = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 0,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: false,
        });
        let b = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 0,
            slot_index: 1,
            collect_order: 1,
            alpha_blended: false,
        });
        let draws = vec![a, b];
        let s = world_mesh_draw_stats_from_sorted(&draws, None, true);
        assert_eq!(s.batch_total, 1);
        assert_eq!(s.draws_total, 2);
        assert_eq!(s.rigid_draws, 2);
        assert_eq!(s.instance_batch_total, 1);
    }
}
