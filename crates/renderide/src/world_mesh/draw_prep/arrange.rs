//! Phase-binned draw arrangement before world-mesh instance planning.

use std::cmp::Ordering;

use hashbrown::HashMap;

use crate::world_mesh::MaterialDrawBatchKey;
use crate::world_mesh::WorldMeshPhase;
use crate::world_mesh::phase_classification::classify_world_mesh_batch;

use super::item::{WorldMeshDrawArrangementStats, WorldMeshDrawItem};
use super::sort::sort_order_sensitive_draws;

/// Key for one nontransparent bin.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct NonTransparentBinKey {
    /// Main-layer draws sort before overlay draws.
    is_overlay: bool,
    /// Primary render phase for the bin.
    phase: WorldMeshPhase,
    /// Effective Unity render queue.
    render_queue: i32,
    /// Cached hash of [`Self::batch_key`] for cheap bin ordering.
    batch_key_hash: u64,
    /// Material and pipeline state shared by all draws in the bin.
    batch_key: MaterialDrawBatchKey,
    /// Resident mesh asset id.
    mesh_asset_id: i32,
    /// First index in the submesh range.
    first_index: u32,
    /// Number of indices in the submesh range.
    index_count: u32,
}

impl NonTransparentBinKey {
    /// Builds the bin key for one draw and its pre-classified render phase.
    fn from_draw(item: &WorldMeshDrawItem, phase: WorldMeshPhase) -> Self {
        Self {
            is_overlay: item.is_overlay,
            phase,
            render_queue: item.batch_key.render_queue,
            batch_key_hash: item.batch_key_hash,
            batch_key: item.batch_key.clone(),
            mesh_asset_id: item.mesh_asset_id,
            first_index: item.first_index,
            index_count: item.index_count,
        }
    }
}

/// Arranges collected draws with bins for nontransparent phases and strict sorting for the
/// transparent tail.
pub(super) fn arrange_draws_by_phase_bins(
    items: &mut Vec<WorldMeshDrawItem>,
    allow_parallel_sort: bool,
) -> WorldMeshDrawArrangementStats {
    profiling::scope!("mesh::arrange_draws_by_phase_bins");
    if items.is_empty() {
        return WorldMeshDrawArrangementStats::default();
    }

    let input = std::mem::take(items);
    let mut bins: HashMap<NonTransparentBinKey, Vec<WorldMeshDrawItem>> =
        HashMap::with_capacity(input.len().min(1_024));
    let mut strict_ordered = Vec::new();

    for item in input {
        let phase = classify_world_mesh_batch(&item.batch_key).phase;
        if phase_requires_strict_order(phase) {
            strict_ordered.push(item);
        } else {
            bins.entry(NonTransparentBinKey::from_draw(&item, phase))
                .or_default()
                .push(item);
        }
    }

    let mut binned: Vec<_> = bins.into_iter().collect();
    let stats = WorldMeshDrawArrangementStats {
        nontransparent_bins: binned.len(),
        nontransparent_binned_draws: binned.iter().map(|(_, draws)| draws.len()).sum(),
        strict_sorted_draws: strict_ordered.len(),
    };

    {
        profiling::scope!("mesh::arrange_draws_by_phase_bins::sort_bins");
        binned.sort_unstable_by(|(a, _), (b, _)| cmp_nontransparent_bin_keys(a, b));
    }
    {
        profiling::scope!("mesh::arrange_draws_by_phase_bins::sort_strict_ordered");
        sort_order_sensitive_draws(&mut strict_ordered, allow_parallel_sort);
    }
    {
        profiling::scope!("mesh::arrange_draws_by_phase_bins::flatten");
        items.reserve(stats.nontransparent_binned_draws + stats.strict_sorted_draws);
        for (_, mut bin_items) in binned {
            items.append(&mut bin_items);
        }
        items.append(&mut strict_ordered);
    }

    stats
}

/// Returns whether draws in `phase` must retain strict transparent/grab ordering.
fn phase_requires_strict_order(phase: WorldMeshPhase) -> bool {
    matches!(
        phase,
        WorldMeshPhase::Transparent | WorldMeshPhase::TransparentGrab
    )
}

/// Stable rank used to flatten nontransparent phases in pass order.
fn phase_flatten_rank(phase: WorldMeshPhase) -> u8 {
    match phase {
        WorldMeshPhase::ForwardOpaque => 0,
        WorldMeshPhase::ForwardAlphaTest => 1,
        WorldMeshPhase::Intersection => 2,
        WorldMeshPhase::Transparent => 3,
        WorldMeshPhase::TransparentGrab => 4,
        WorldMeshPhase::DepthOnly => 5,
        WorldMeshPhase::ViewNormals => 6,
    }
}

/// Orders nontransparent bins so same material packet keys stay contiguous while preserving
/// high-level pass order.
fn cmp_nontransparent_bin_keys(a: &NonTransparentBinKey, b: &NonTransparentBinKey) -> Ordering {
    a.is_overlay
        .cmp(&b.is_overlay)
        .then_with(|| phase_flatten_rank(a.phase).cmp(&phase_flatten_rank(b.phase)))
        .then(a.render_queue.cmp(&b.render_queue))
        .then(a.batch_key_hash.cmp(&b.batch_key_hash))
        .then_with(|| a.batch_key.cmp(&b.batch_key))
        .then(a.mesh_asset_id.cmp(&b.mesh_asset_id))
        .then(a.first_index.cmp(&b.first_index))
        .then(a.index_count.cmp(&b.index_count))
}

#[cfg(test)]
mod tests {
    use crate::materials::{
        UNITY_RENDER_QUEUE_ALPHA_TEST, UNITY_RENDER_QUEUE_TRANSPARENT, render_queue_is_transparent,
    };
    use crate::world_mesh::draw_prep::pack_sort_prefix;
    use crate::world_mesh::materials::compute_batch_key_hash;
    use crate::world_mesh::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};

    use crate::world_mesh::WorldMeshDrawItem;

    use super::arrange_draws_by_phase_bins;

    /// Builds an opaque dummy draw item.
    fn opaque(mesh: i32, material: i32, collect_order: usize) -> WorldMeshDrawItem {
        dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: material,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: mesh,
            node_id: collect_order as i32,
            slot_index: 0,
            collect_order,
            alpha_blended: false,
        })
    }

    /// Refreshes precomputed batch and sort keys after mutating material state.
    fn refresh_keys(item: &mut WorldMeshDrawItem) {
        item.batch_key_hash = compute_batch_key_hash(&item.batch_key);
        item.sort_prefix = pack_sort_prefix(
            item.is_overlay,
            item.batch_key.render_queue,
            item._opaque_depth_bucket,
            item.batch_key_hash,
        );
    }

    /// Sets a draw's render queue and refreshes precomputed keys.
    fn set_render_queue(item: &mut WorldMeshDrawItem, render_queue: i32) {
        item.batch_key.render_queue = render_queue;
        refresh_keys(item);
    }

    /// Sets the sort distance used by transparent strict ordering.
    fn set_camera_distance(item: &mut WorldMeshDrawItem, distance_sq: f32) {
        item.camera_distance_sq = distance_sq;
    }

    #[test]
    fn opaque_bins_keep_same_material_contiguous_without_full_item_sort() {
        let mut repeated_mesh = opaque(10, 1, 0);
        repeated_mesh.node_id = 100;
        let mut draws = vec![
            repeated_mesh,
            opaque(20, 2, 1),
            opaque(11, 1, 2),
            opaque(10, 1, 3),
        ];

        let stats = arrange_draws_by_phase_bins(&mut draws, false);

        assert_eq!(stats.nontransparent_binned_draws, 4);
        assert_eq!(stats.strict_sorted_draws, 0);
        let material_runs: Vec<_> = draws
            .iter()
            .map(|draw| draw.batch_key.material_asset_id)
            .fold(Vec::<i32>::new(), |mut runs, material| {
                if runs.last().copied() != Some(material) {
                    runs.push(material);
                }
                runs
            });
        assert_eq!(material_runs.len(), 2);
        let material_one: Vec<_> = draws
            .iter()
            .filter(|draw| draw.batch_key.material_asset_id == 1)
            .map(|draw| draw.mesh_asset_id)
            .collect();
        assert_eq!(material_one, vec![10, 10, 11]);
    }

    #[test]
    fn alpha_test_and_intersection_bins_flatten_before_transparent_tail() {
        let mut alpha_test = opaque(1, 1, 0);
        set_render_queue(&mut alpha_test, UNITY_RENDER_QUEUE_ALPHA_TEST);
        let mut intersect = opaque(1, 2, 1);
        intersect.batch_key.embedded_requires_intersection_pass = true;
        refresh_keys(&mut intersect);
        let mut transparent = opaque(1, 3, 2);
        set_render_queue(&mut transparent, UNITY_RENDER_QUEUE_TRANSPARENT);

        let mut draws = vec![transparent, intersect, alpha_test];
        let stats = arrange_draws_by_phase_bins(&mut draws, false);

        assert_eq!(stats.nontransparent_binned_draws, 2);
        assert_eq!(stats.strict_sorted_draws, 1);
        assert_eq!(
            draws[0].batch_key.render_queue,
            UNITY_RENDER_QUEUE_ALPHA_TEST
        );
        assert!(draws[1].batch_key.embedded_requires_intersection_pass);
        assert!(render_queue_is_transparent(draws[2].batch_key.render_queue));
    }

    #[test]
    fn transparent_tail_keeps_back_to_front_order() {
        let mut near = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 1,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: true,
        });
        set_camera_distance(&mut near, 1.0);
        let mut far = near.clone();
        far.node_id = 2;
        far.collect_order = 1;
        set_camera_distance(&mut far, 64.0);

        let mut draws = vec![near, far];
        arrange_draws_by_phase_bins(&mut draws, false);

        assert_eq!(draws[0].node_id, 2);
        assert_eq!(draws[1].node_id, 1);
    }

    #[test]
    fn grab_and_regular_transparent_share_one_strict_tail_order() {
        let mut grab = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 1,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: true,
        });
        grab.batch_key.embedded_uses_scene_color_snapshot = true;
        refresh_keys(&mut grab);
        set_camera_distance(&mut grab, 100.0);
        let mut regular = grab.clone();
        regular.node_id = 2;
        regular.collect_order = 1;
        regular.batch_key.embedded_uses_scene_color_snapshot = false;
        refresh_keys(&mut regular);
        set_camera_distance(&mut regular, 4.0);

        let mut draws = vec![regular, grab];
        arrange_draws_by_phase_bins(&mut draws, false);

        assert!(draws[0].batch_key.embedded_uses_scene_color_snapshot);
        assert!(!draws[1].batch_key.embedded_uses_scene_color_snapshot);
    }
}
