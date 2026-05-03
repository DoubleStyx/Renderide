//! Batch keys and draw list ordering for world mesh forward.

use std::cmp::Ordering;

use rayon::slice::ParallelSliceMut;

use crate::materials::render_queue_is_transparent;

use super::item::WorldMeshDrawItem;

/// Bit width of the render-queue field inside [`WorldMeshDrawItem::sort_prefix`].
const SORT_PREFIX_RENDER_QUEUE_BITS: u32 = 18;
/// Maximum render-queue value representable inside [`WorldMeshDrawItem::sort_prefix`].
const SORT_PREFIX_RENDER_QUEUE_MAX: i32 = (1 << SORT_PREFIX_RENDER_QUEUE_BITS) - 1;

/// Bit shift for the overlay flag (highest bit, sorts last by default).
const SORT_PREFIX_OVERLAY_SHIFT: u32 = 63;
/// Bit shift for the 18-bit render queue (just below overlay).
const SORT_PREFIX_RENDER_QUEUE_SHIFT: u32 = 45;
/// Bit shift for the transparent-queue flag.
const SORT_PREFIX_TRANSPARENT_SHIFT: u32 = 44;
/// Bit shift for the 8-bit opaque depth bucket.
const SORT_PREFIX_DEPTH_BUCKET_SHIFT: u32 = 36;
/// Bit shift for the 32-bit upper half of the batch-key hash.
const SORT_PREFIX_BATCH_HASH_SHIFT: u32 = 4;

/// Maps camera-distance squared into a coarse logarithmic front-to-back bucket.
///
/// Called once per draw at candidate evaluation and the result stored on
/// [`WorldMeshDrawItem::opaque_depth_bucket`]; the comparator then reads the field directly
/// instead of recomputing `sqrt` + `log2` on every pairwise compare.
pub(super) fn opaque_depth_bucket(distance_sq: f32) -> u16 {
    if !distance_sq.is_finite() || distance_sq <= 0.0 {
        return 0;
    }
    let distance = distance_sq.sqrt().max(1e-4);
    ((distance.log2() + 16.0).floor().clamp(0.0, 255.0)) as u16
}

/// Packs the dominant ordering prefix of a draw into a single [`u64`] so the hot sort path can
/// use [`u64::cmp`] instead of a multi-field comparator chain.
///
/// Transparent-queue draws zero the depth-bucket and hash bits so every transparent draw inside
/// the same `(overlay, render_queue)` bucket compares equal; [`sort_draws`] resorts each such
/// run afterwards using the structural comparator on `(sorting_order, camera_distance_sq,
/// collect_order)`.
#[inline]
pub fn pack_sort_prefix(
    is_overlay: bool,
    render_queue: i32,
    opaque_depth_bucket: u16,
    batch_key_hash: u64,
) -> u64 {
    let overlay_bit = u64::from(is_overlay);
    let render_queue_clamped = render_queue.clamp(0, SORT_PREFIX_RENDER_QUEUE_MAX) as u64;
    let is_transparent = render_queue_is_transparent(render_queue);
    let transparent_bit = u64::from(is_transparent);
    let (depth_bits, hash_bits) = if is_transparent {
        (0u64, 0u64)
    } else {
        (
            u64::from(opaque_depth_bucket.min((1u16 << 8) - 1)),
            batch_key_hash >> 32,
        )
    };

    (overlay_bit << SORT_PREFIX_OVERLAY_SHIFT)
        | (render_queue_clamped << SORT_PREFIX_RENDER_QUEUE_SHIFT)
        | (transparent_bit << SORT_PREFIX_TRANSPARENT_SHIFT)
        | (depth_bits << SORT_PREFIX_DEPTH_BUCKET_SHIFT)
        | (hash_bits << SORT_PREFIX_BATCH_HASH_SHIFT)
}

/// Tiebreaker for transparent draws sharing the same `(overlay, render_queue)` bucket: stable
/// `sorting_order`, then back-to-front `camera_distance_sq` (using `total_cmp` to handle NaN
/// safely), then `collect_order`. Used both by [`resort_intra_prefix_runs`] in the runtime path
/// and by [`cmp_world_mesh_draw_items`] when a test compares two transparent draws directly.
#[inline]
fn cmp_transparent_intra_run(a: &WorldMeshDrawItem, b: &WorldMeshDrawItem) -> Ordering {
    a.sorting_order
        .cmp(&b.sorting_order)
        .then_with(|| b.camera_distance_sq.total_cmp(&a.camera_distance_sq))
        .then(a.collect_order.cmp(&b.collect_order))
}

/// Tiebreaker for opaque draws sharing the same packed prefix.
///
/// Two opaque draws share a packed prefix when their `(overlay, render_queue, depth_bucket,
/// batch_key_hash_hi32)` agree. Within that bucket the original comparator preserved a
/// deterministic order via the full `batch_key_hash`, then a structural `batch_key` compare on
/// hash collisions, then `sorting_order` descending, then `(mesh_asset_id, node_id, slot_index)`.
/// This function reproduces that order for the post-radix fix-up in
/// [`resort_intra_prefix_runs`].
#[inline]
fn cmp_opaque_intra_prefix(a: &WorldMeshDrawItem, b: &WorldMeshDrawItem) -> Ordering {
    a.batch_key_hash
        .cmp(&b.batch_key_hash)
        .then_with(|| a.batch_key.cmp(&b.batch_key))
        .then(b.sorting_order.cmp(&a.sorting_order))
        .then(a.mesh_asset_id.cmp(&b.mesh_asset_id))
        .then(a.node_id.cmp(&b.node_id))
        .then(a.slot_index.cmp(&b.slot_index))
}

/// Full structural comparator equivalent to the pre-packing `cmp_world_mesh_draw_items`.
///
/// Test-only: the runtime sort path consumes [`WorldMeshDrawItem::sort_prefix`] via
/// `sort_unstable_by_key` and only uses the structural comparator on transparent intra-run
/// fix-up (see [`resort_transparent_runs`]).
#[cfg(test)]
fn cmp_world_mesh_draw_items(a: &WorldMeshDrawItem, b: &WorldMeshDrawItem) -> Ordering {
    a.sort_prefix.cmp(&b.sort_prefix).then_with(|| {
        let a_transparent = render_queue_is_transparent(a.batch_key.render_queue);
        let b_transparent = render_queue_is_transparent(b.batch_key.render_queue);
        match (a_transparent, b_transparent) {
            (false, false) => a
                .batch_key_hash
                .cmp(&b.batch_key_hash)
                .then_with(|| a.batch_key.cmp(&b.batch_key))
                .then(b.sorting_order.cmp(&a.sorting_order))
                .then(a.mesh_asset_id.cmp(&b.mesh_asset_id))
                .then(a.node_id.cmp(&b.node_id))
                .then(a.slot_index.cmp(&b.slot_index)),
            (true, true) => cmp_transparent_intra_run(a, b),
            _ => Ordering::Equal,
        }
    })
}

/// Pre-depth-bucket ordering retained for regression tests that need to isolate batch-key order.
#[cfg(test)]
fn cmp_world_mesh_draw_items_without_depth_bucket(
    a: &WorldMeshDrawItem,
    b: &WorldMeshDrawItem,
) -> Ordering {
    a.is_overlay
        .cmp(&b.is_overlay)
        .then(a.batch_key.render_queue.cmp(&b.batch_key.render_queue))
        .then(
            render_queue_is_transparent(a.batch_key.render_queue)
                .cmp(&render_queue_is_transparent(b.batch_key.render_queue)),
        )
        .then_with(|| {
            match (
                render_queue_is_transparent(a.batch_key.render_queue),
                render_queue_is_transparent(b.batch_key.render_queue),
            ) {
                (false, false) => a
                    .batch_key
                    .cmp(&b.batch_key)
                    .then(b.sorting_order.cmp(&a.sorting_order))
                    .then(a.mesh_asset_id.cmp(&b.mesh_asset_id))
                    .then(a.node_id.cmp(&b.node_id))
                    .then(a.slot_index.cmp(&b.slot_index)),
                (true, true) => a
                    .sorting_order
                    .cmp(&b.sorting_order)
                    .then_with(|| b.camera_distance_sq.total_cmp(&a.camera_distance_sq))
                    .then(a.collect_order.cmp(&b.collect_order)),
                _ => Ordering::Equal,
            }
        })
}

/// Walks the slice (already sorted by [`WorldMeshDrawItem::sort_prefix`]) and resorts each
/// contiguous run of equal-prefix items with the structural intra-prefix comparator.
///
/// Two cases produce a multi-element run:
///
/// * Opaque draws sharing `(overlay, render_queue, depth_bucket, batch_key_hash_hi32)`. Within
///   such a run the structural opaque comparator preserves the deterministic
///   `batch_key_hash` -> `batch_key` -> `sorting_order` (descending) -> `mesh / node / slot`
///   ordering. Common when many draws share a batch key.
/// * Transparent draws inside the same `(overlay, render_queue)` bucket. [`pack_sort_prefix`]
///   zeros the depth-bucket and hash bits for transparent items so they all collide on the
///   primary key; the transparent comparator then sorts by `sorting_order`, back-to-front
///   `camera_distance_sq`, then `collect_order`.
fn resort_intra_prefix_runs(items: &mut [WorldMeshDrawItem]) {
    profiling::scope!("mesh::sort_intra_prefix_runs");
    let mut start = 0;
    while start < items.len() {
        let prefix = items[start].sort_prefix;
        let mut end = start + 1;
        while end < items.len() && items[end].sort_prefix == prefix {
            end += 1;
        }
        if end - start > 1 {
            let is_transparent = render_queue_is_transparent(items[start].batch_key.render_queue);
            if is_transparent {
                items[start..end].sort_unstable_by(cmp_transparent_intra_run);
            } else {
                items[start..end].sort_unstable_by(cmp_opaque_intra_prefix);
            }
        }
        start = end;
    }
}

/// Sorts opaque draws for batching and alpha UI/text draws in stable canvas order.
///
/// Primary pass: parallel `sort_unstable_by_key` over [`WorldMeshDrawItem::sort_prefix`] —
/// replaces the prior multi-field `cmp_world_mesh_draw_items` chain with a single `u64::cmp`
/// per pairwise compare, which is the dominant cost reduction. Secondary pass:
/// [`resort_intra_prefix_runs`] resolves opaque and transparent ties using the structural
/// comparators.
pub fn sort_draws(items: &mut [WorldMeshDrawItem]) {
    profiling::scope!("mesh::sort_draws");
    items.par_sort_unstable_by_key(|item| item.sort_prefix);
    resort_intra_prefix_runs(items);
}

/// Same ordering as [`sort_draws`] without rayon (for nested parallel batches).
pub(super) fn sort_draws_serial(items: &mut [WorldMeshDrawItem]) {
    profiling::scope!("mesh::sort_draws_serial");
    items.sort_unstable_by_key(|item| item.sort_prefix);
    resort_intra_prefix_runs(items);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materials::{
        UNITY_RENDER_QUEUE_ALPHA_TEST, UNITY_RENDER_QUEUE_OVERLAY, UNITY_RENDER_QUEUE_TRANSPARENT,
    };
    use crate::render_graph::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};
    use crate::world_mesh::materials::compute_batch_key_hash;

    /// Sets `camera_distance_sq` and refreshes the precomputed `opaque_depth_bucket` and
    /// `sort_prefix` so test fixtures match what `evaluate_draw_candidate` would produce in
    /// production.
    fn set_camera_distance(item: &mut WorldMeshDrawItem, distance_sq: f32) {
        item.camera_distance_sq = distance_sq;
        item.opaque_depth_bucket = opaque_depth_bucket(distance_sq);
        item.sort_prefix = pack_sort_prefix(
            item.is_overlay,
            item.batch_key.render_queue,
            item.opaque_depth_bucket,
            item.batch_key_hash,
        );
    }

    fn set_render_queue(item: &mut WorldMeshDrawItem, render_queue: i32) {
        item.batch_key.render_queue = render_queue;
        item.batch_key_hash = compute_batch_key_hash(&item.batch_key);
        item.sort_prefix = pack_sort_prefix(
            item.is_overlay,
            item.batch_key.render_queue,
            item.opaque_depth_bucket,
            item.batch_key_hash,
        );
    }

    #[test]
    fn opaque_sort_prefers_nearer_depth_bucket_before_batch_key() {
        let mut near = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 2,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 2,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: false,
        });
        set_camera_distance(&mut near, 1.0);
        let mut far = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 1,
            slot_index: 0,
            collect_order: 1,
            alpha_blended: false,
        });
        set_camera_distance(&mut far, 4096.0);

        assert_eq!(
            cmp_world_mesh_draw_items(&near, &far),
            Ordering::Less,
            "near opaque draws should sort before lower material ids when depth buckets differ"
        );
        assert_eq!(
            cmp_world_mesh_draw_items_without_depth_bucket(&near, &far),
            Ordering::Greater,
            "the regression setup must differ from pure batch-key ordering"
        );
    }

    #[test]
    fn transparent_sort_remains_back_to_front() {
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
        let mut far = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 2,
            slot_index: 0,
            collect_order: 1,
            alpha_blended: true,
        });
        set_camera_distance(&mut far, 4096.0);

        assert_eq!(cmp_world_mesh_draw_items(&far, &near), Ordering::Less);
    }

    #[test]
    fn render_queue_orders_before_transparent_distance() {
        let mut near_early_queue = dummy_world_mesh_draw_item(DummyDrawItemSpec {
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
        set_camera_distance(&mut near_early_queue, 1.0);
        set_render_queue(&mut near_early_queue, UNITY_RENDER_QUEUE_TRANSPARENT);

        let mut far_late_queue = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 2,
            slot_index: 0,
            collect_order: 1,
            alpha_blended: true,
        });
        set_camera_distance(&mut far_late_queue, 4096.0);
        set_render_queue(&mut far_late_queue, UNITY_RENDER_QUEUE_TRANSPARENT + 5);

        assert_eq!(
            cmp_world_mesh_draw_items(&near_early_queue, &far_late_queue),
            Ordering::Less,
            "lower transparent render queues must draw before farther later queues"
        );
    }

    #[test]
    fn render_queue_orders_alpha_test_transparent_and_overlay_ranges() {
        let mut transparent = dummy_world_mesh_draw_item(DummyDrawItemSpec {
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
        set_render_queue(&mut transparent, UNITY_RENDER_QUEUE_TRANSPARENT);

        let mut alpha_test = transparent.clone();
        set_render_queue(&mut alpha_test, UNITY_RENDER_QUEUE_ALPHA_TEST);

        let mut late_transparent = transparent.clone();
        set_render_queue(&mut late_transparent, UNITY_RENDER_QUEUE_TRANSPARENT + 5);

        let mut overlay = transparent.clone();
        set_render_queue(&mut overlay, UNITY_RENDER_QUEUE_OVERLAY);

        let mut items = vec![overlay, late_transparent, transparent, alpha_test];
        sort_draws_serial(&mut items);

        let queues: Vec<_> = items
            .iter()
            .map(|item| item.batch_key.render_queue)
            .collect();
        assert_eq!(
            queues,
            vec![
                UNITY_RENDER_QUEUE_ALPHA_TEST,
                UNITY_RENDER_QUEUE_TRANSPARENT,
                UNITY_RENDER_QUEUE_TRANSPARENT + 5,
                UNITY_RENDER_QUEUE_OVERLAY,
            ]
        );
    }

    #[test]
    fn pack_sort_prefix_orders_overlay_after_main() {
        let main = pack_sort_prefix(false, UNITY_RENDER_QUEUE_TRANSPARENT, 0, 0);
        let overlay = pack_sort_prefix(true, 0, 0, 0);
        assert!(main < overlay);
    }

    #[test]
    fn pack_sort_prefix_orders_lower_render_queue_first() {
        let lo = pack_sort_prefix(false, 0, 0, 0);
        let hi = pack_sort_prefix(false, UNITY_RENDER_QUEUE_TRANSPARENT, 0, 0);
        assert!(lo < hi);
    }

    #[test]
    fn pack_sort_prefix_zeros_depth_and_hash_for_transparent() {
        let with_depth_and_hash = pack_sort_prefix(
            false,
            UNITY_RENDER_QUEUE_TRANSPARENT,
            200,
            0xDEAD_BEEF_DEAD_BEEF,
        );
        let bare = pack_sort_prefix(false, UNITY_RENDER_QUEUE_TRANSPARENT, 0, 0);
        assert_eq!(
            with_depth_and_hash, bare,
            "transparent draws must share a key within their (overlay, render_queue) bucket"
        );
    }

    #[test]
    fn pack_sort_prefix_keeps_depth_and_hash_for_opaque() {
        let near = pack_sort_prefix(false, 0, 10, 0);
        let far = pack_sort_prefix(false, 0, 200, 0);
        assert!(near < far);
        let same_depth_lo_hash = pack_sort_prefix(false, 0, 10, 0);
        let same_depth_hi_hash = pack_sort_prefix(false, 0, 10, u64::MAX);
        assert!(same_depth_lo_hash < same_depth_hi_hash);
    }
}
