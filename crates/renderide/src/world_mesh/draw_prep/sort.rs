//! Batch keys and draw list ordering for world mesh forward.

use std::cmp::Ordering;

use rayon::slice::ParallelSliceMut;

use crate::materials::render_queue_is_transparent;

use super::item::WorldMeshDrawItem;

/// Equal-prefix run length above which the secondary structural resort uses Rayon.
///
/// The primary prefix sort already used the worker pool. This gate is for large transparent
/// buckets and opaque hash-prefix buckets where the tie-breaker comparator can still dominate.
const INTRA_PREFIX_RUN_PARALLEL_MIN: usize = 1_024;

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
/// [`WorldMeshDrawItem::_opaque_depth_bucket`]; the comparator then reads the field directly
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
/// run afterwards using the class-aware structural comparator.
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

/// Tiebreaker for transparent draws sharing the same `(overlay, render_queue)` bucket.
///
/// Order-dependent classes keep `sorting_order`, back-to-front distance, then collection order.
/// Commutative additive/multiply classes may sort by batch key inside the same sorting-order
/// bucket because their color composition is order independent.
#[inline]
pub(super) fn cmp_transparent_intra_run(a: &WorldMeshDrawItem, b: &WorldMeshDrawItem) -> Ordering {
    a.sorting_order
        .cmp(&b.sorting_order)
        .then_with(|| cmp_transparent_class_tie(a, b))
        .then(a.collect_order.cmp(&b.collect_order))
}

/// Comparator for draws whose visible result depends on submission order.
///
/// The prefix matches the old transparent-sort buckets: main-layer work before overlay, then
/// ascending Unity render queue. Within that bucket the existing transparent class comparator
/// preserves back-to-front ordering for ordered alpha and batch-friendly ordering for commutative
/// blends.
#[inline]
pub(super) fn cmp_order_sensitive_draws(a: &WorldMeshDrawItem, b: &WorldMeshDrawItem) -> Ordering {
    a.is_overlay
        .cmp(&b.is_overlay)
        .then(a.batch_key.render_queue.cmp(&b.batch_key.render_queue))
        .then_with(|| cmp_transparent_intra_run(a, b))
        .then(a.sort_prefix.cmp(&b.sort_prefix))
}

/// Orders transparent draws after `sorting_order` has already matched.
#[inline]
fn cmp_transparent_class_tie(a: &WorldMeshDrawItem, b: &WorldMeshDrawItem) -> Ordering {
    if a.batch_key.transparent_class.allows_relaxed_batching()
        && b.batch_key.transparent_class.allows_relaxed_batching()
    {
        return a
            .batch_key_hash
            .cmp(&b.batch_key_hash)
            .then_with(|| a.batch_key.cmp(&b.batch_key))
            .then_with(|| b.camera_distance_sq.total_cmp(&a.camera_distance_sq));
    }

    b.camera_distance_sq.total_cmp(&a.camera_distance_sq)
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
#[cfg(test)]
fn cmp_opaque_intra_prefix(a: &WorldMeshDrawItem, b: &WorldMeshDrawItem) -> Ordering {
    a.batch_key_hash
        .cmp(&b.batch_key_hash)
        .then_with(|| a.batch_key.cmp(&b.batch_key))
        .then(b.sorting_order.cmp(&a.sorting_order))
        .then(a.mesh_asset_id.cmp(&b.mesh_asset_id))
        .then(a.node_id.cmp(&b.node_id))
        .then(a.slot_index.cmp(&b.slot_index))
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
///   primary key; the transparent comparator then sorts by `sorting_order`, an appropriate class
///   tie-breaker, then `collect_order`.
#[cfg(test)]
fn resort_intra_prefix_runs(items: &mut [WorldMeshDrawItem], allow_parallel: bool) {
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
                sort_intra_prefix_run(
                    &mut items[start..end],
                    cmp_transparent_intra_run,
                    allow_parallel,
                );
            } else {
                sort_intra_prefix_run(
                    &mut items[start..end],
                    cmp_opaque_intra_prefix,
                    allow_parallel,
                );
            }
        }
        start = end;
    }
}

fn sort_intra_prefix_run(
    run: &mut [WorldMeshDrawItem],
    cmp: fn(&WorldMeshDrawItem, &WorldMeshDrawItem) -> Ordering,
    allow_parallel: bool,
) {
    if allow_parallel && run.len() >= INTRA_PREFIX_RUN_PARALLEL_MIN {
        profiling::scope!("mesh::sort_intra_prefix_run_parallel");
        run.par_sort_unstable_by(cmp);
    } else {
        run.sort_unstable_by(cmp);
    }
}

/// Sorts order-sensitive transparent/grab draws while leaving nontransparent bins out of the
/// full item sort.
pub(super) fn sort_order_sensitive_draws(items: &mut [WorldMeshDrawItem], allow_parallel: bool) {
    profiling::scope!("mesh::sort_order_sensitive_draws");
    sort_intra_prefix_run(items, cmp_order_sensitive_draws, allow_parallel);
}

/// Sorts opaque draws for batching and transparent draws by compatibility class.
///
/// Primary pass: parallel `sort_unstable_by_key` over [`WorldMeshDrawItem::sort_prefix`] —
/// replaces the prior multi-field `cmp_world_mesh_draw_items` chain with a single `u64::cmp`
/// per pairwise compare, which is the dominant cost reduction. Secondary pass:
/// [`resort_intra_prefix_runs`] resolves opaque and transparent ties using the structural
/// comparators.
#[cfg(test)]
pub fn sort_draws(items: &mut [WorldMeshDrawItem]) {
    profiling::scope!("mesh::sort_draws");
    items.par_sort_unstable_by_key(|item| item.sort_prefix);
    resort_intra_prefix_runs(items, true);
}

/// Same ordering as [`sort_draws`] without rayon (for nested parallel batches).
#[cfg(test)]
pub(super) fn sort_draws_serial(items: &mut [WorldMeshDrawItem]) {
    profiling::scope!("mesh::sort_draws_serial");
    items.sort_unstable_by_key(|item| item.sort_prefix);
    resort_intra_prefix_runs(items, false);
}

#[cfg(test)]
mod tests;
