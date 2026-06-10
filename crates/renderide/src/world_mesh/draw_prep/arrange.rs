//! Phase-binned draw arrangement before world-mesh instance planning.

use std::cmp::Ordering;

use rayon::prelude::*;

use crate::world_mesh::WorldMeshPhase;
use crate::world_mesh::phase_classification::classify_world_mesh_batch;

use super::item::{WorldMeshDrawArrangementStats, WorldMeshDrawItem};
use super::sort::cmp_order_sensitive_draws;

/// Draw count at which compact arrangement row sorting uses Rayon workers.
const ARRANGE_PARALLEL_MIN_DRAWS: usize = 512;

/// Bin count at which bin-key sorting uses Rayon workers.
const ARRANGE_PARALLEL_MIN_BINS: usize = 512;

/// Key for one nontransparent bin.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NonTransparentBinKey {
    /// Main-layer draws sort before overlay draws.
    is_overlay: bool,
    /// Primary render phase for the bin.
    phase: WorldMeshPhase,
    /// Stable pass-order rank for [`Self::phase`].
    phase_rank: u8,
    /// Effective Unity render queue.
    render_queue: i32,
    /// Material-stack ordering key for slots that reuse the final submesh.
    stack: Option<NonTransparentStackBinKey>,
    /// Compact per-arrangement material and pipeline batch identifier.
    batch_id: u32,
    /// Resident mesh asset id.
    mesh_asset_id: i32,
    /// First index in the submesh range.
    first_index: u32,
    /// Number of indices in the submesh range.
    index_count: u32,
}

impl NonTransparentBinKey {
    /// Builds the bin key for one draw and its pre-classified render phase.
    fn from_draw(item: &WorldMeshDrawItem, phase: WorldMeshPhase, batch_id: u32) -> Self {
        Self {
            is_overlay: item.is_overlay,
            phase,
            phase_rank: phase_flatten_rank(phase),
            render_queue: item.batch_key.render_queue,
            stack: NonTransparentStackBinKey::from_draw(item),
            batch_id,
            mesh_asset_id: item.mesh_asset_id,
            first_index: item.first_index,
            index_count: item.index_count,
        }
    }
}

/// Ordering key for a nontransparent material-stack bin.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NonTransparentStackBinKey {
    /// Host render space id.
    space_id: crate::scene::RenderSpaceId,
    /// Whether this key points at a skinned renderer table.
    skinned: bool,
    /// Dense renderer index inside the selected renderer table.
    renderable_index: usize,
    /// Renderer-local stable identity.
    instance_id: u64,
    /// Resident mesh asset id.
    mesh_asset_id: i32,
    /// First index in the stacked submesh range.
    first_index: u32,
    /// Number of indices in the stacked submesh range.
    index_count: u32,
    /// First material slot participating in this stack.
    first_stacked_slot_index: usize,
    /// Material slot represented by this bin.
    slot_index: usize,
}

impl NonTransparentStackBinKey {
    /// Builds a stack bin key for one draw item when it participates in material stacking.
    fn from_draw(item: &WorldMeshDrawItem) -> Option<Self> {
        let stack = item.material_stack_order?;
        Some(Self {
            space_id: item.space_id,
            skinned: item.skinned,
            renderable_index: item.renderable_index,
            instance_id: item.instance_id.0,
            mesh_asset_id: item.mesh_asset_id,
            first_index: item.first_index,
            index_count: item.index_count,
            first_stacked_slot_index: stack.first_stacked_slot_index,
            slot_index: item.slot_index,
        })
    }
}

/// Per-arrangement compact IDs for material and pipeline batch keys.
#[derive(Debug)]
struct BatchIdTable {
    /// Dense per-draw batch ids indexed by flattened draw index.
    draw_ids: Vec<u32>,
}

impl BatchIdTable {
    /// Builds compact batch IDs from the flattened draw list.
    fn build_from_items(items: &[WorldMeshDrawItem], allow_parallel: bool) -> Self {
        profiling::scope!("mesh::arrange_draws_by_phase_bins::batch_ids");
        let mut sorted_indices = (0..items.len()).collect::<Vec<_>>();
        sort_batch_indices(&mut sorted_indices, items, allow_parallel);

        let mut draw_ids = vec![u32::MAX; items.len()];
        let mut previous = None::<usize>;
        let mut batch_id = 0u32;
        for &index in &sorted_indices {
            if let Some(previous_index) = previous
                && !same_batch_identity(&items[previous_index], &items[index])
            {
                batch_id = batch_id.saturating_add(1);
            }
            if let Some(slot) = draw_ids.get_mut(index) {
                *slot = batch_id;
            }
            previous = Some(index);
        }

        Self { draw_ids }
    }

    /// Returns the compact batch ID for a flattened draw index.
    #[inline]
    fn id_for_index(&self, index: usize) -> u32 {
        self.draw_ids.get(index).copied().unwrap_or(u32::MAX)
    }
}

/// Compact nontransparent draw row sorted instead of the full draw item.
#[derive(Clone, Copy, Debug)]
struct NonTransparentDrawRow {
    /// Bin key for this row.
    key: NonTransparentBinKey,
    /// Flattened source draw index.
    source_index: usize,
}

/// Strict order-sensitive draw row sorted by the existing transparent comparator.
#[derive(Clone, Copy, Debug)]
struct StrictDrawRow {
    /// Flattened source draw index.
    source_index: usize,
    /// Compact batch id for comparing against post-skybox nontransparent bins.
    batch_id: u32,
}

/// Contiguous same-key range in the sorted nontransparent row list.
#[derive(Clone, Copy, Debug)]
struct TailBinRow {
    /// Shared key for the row range.
    key: NonTransparentBinKey,
    /// Start row index, inclusive.
    start: usize,
    /// End row index, exclusive.
    end: usize,
}

/// Compact arrangement order plus diagnostics counters.
#[derive(Clone, Debug)]
pub(super) struct WorldMeshDrawArrangementOrder {
    /// Flattened draw indices in final submission order.
    pub(super) indices: Vec<usize>,
    /// Arrangement counters for diagnostics.
    pub(super) stats: WorldMeshDrawArrangementStats,
}

impl WorldMeshDrawArrangementOrder {
    /// Builds an empty order.
    fn empty() -> Self {
        Self {
            indices: Vec::new(),
            stats: WorldMeshDrawArrangementStats::default(),
        }
    }
}

/// Sorts flattened draw indices by material batch identity.
fn sort_batch_indices(indices: &mut [usize], items: &[WorldMeshDrawItem], allow_parallel: bool) {
    if allow_parallel && indices.len() >= ARRANGE_PARALLEL_MIN_DRAWS {
        profiling::scope!("mesh::arrange_draws_by_phase_bins::batch_ids_sort_parallel");
        indices.par_sort_unstable_by(|&a, &b| cmp_batch_identity(&items[a], &items[b]));
    } else {
        profiling::scope!("mesh::arrange_draws_by_phase_bins::batch_ids_sort_serial");
        indices.sort_unstable_by(|&a, &b| cmp_batch_identity(&items[a], &items[b]));
    }
}

/// Compares the full material batch identity, using the precomputed hash as the common fast path.
fn cmp_batch_identity(a: &WorldMeshDrawItem, b: &WorldMeshDrawItem) -> Ordering {
    a.batch_key_hash
        .cmp(&b.batch_key_hash)
        .then_with(|| a.batch_key.cmp(&b.batch_key))
}

/// Returns whether two draws share the same material batch identity.
fn same_batch_identity(a: &WorldMeshDrawItem, b: &WorldMeshDrawItem) -> bool {
    a.batch_key_hash == b.batch_key_hash && a.batch_key == b.batch_key
}

/// Flattens deterministic draw chunks and optionally assigns dense collection order.
pub(super) fn flatten_draw_chunks(
    chunks: Vec<Vec<WorldMeshDrawItem>>,
    assign_collect_order: bool,
) -> Vec<WorldMeshDrawItem> {
    profiling::scope!("mesh::arrange_draws_by_phase_bins::flatten_input");
    let draw_count = chunks.iter().map(Vec::len).sum::<usize>();
    let mut items = Vec::with_capacity(draw_count);
    let mut collect_order = 0usize;
    for mut chunk in chunks {
        if assign_collect_order {
            for item in &mut chunk {
                item.collect_order = collect_order;
                collect_order = collect_order.saturating_add(1);
            }
        }
        items.append(&mut chunk);
    }
    items
}

/// Builds the compact final draw order for a flattened draw list.
pub(super) fn arrange_draw_items_order(
    items: &[WorldMeshDrawItem],
    allow_parallel_sort: bool,
) -> WorldMeshDrawArrangementOrder {
    profiling::scope!("mesh::arrange_draws_by_phase_bins::order");
    if items.is_empty() {
        return WorldMeshDrawArrangementOrder::empty();
    }

    let batch_ids = BatchIdTable::build_from_items(items, allow_parallel_sort);
    let (mut nontransparent_rows, mut strict_rows) = build_arrangement_rows(items, &batch_ids);

    sort_nontransparent_rows(&mut nontransparent_rows, allow_parallel_sort);
    sort_strict_rows(&mut strict_rows, items, allow_parallel_sort);

    let mut indices = Vec::with_capacity(items.len());
    let (nontransparent_bins, tail_bins) =
        append_pre_skybox_bins_and_collect_tail(&nontransparent_rows, &mut indices);
    append_post_skybox_tail_indices(
        &mut indices,
        &tail_bins,
        &nontransparent_rows,
        &strict_rows,
        items,
    );

    WorldMeshDrawArrangementOrder {
        indices,
        stats: WorldMeshDrawArrangementStats {
            nontransparent_bins,
            nontransparent_binned_draws: nontransparent_rows.len(),
            strict_sorted_draws: strict_rows.len(),
        },
    }
}

/// Moves flattened draw items into the supplied final order.
pub(super) fn materialize_arranged_draw_order(
    items: Vec<WorldMeshDrawItem>,
    order: &WorldMeshDrawArrangementOrder,
) -> (Vec<WorldMeshDrawItem>, WorldMeshDrawArrangementStats) {
    let stats = order.stats;
    if !validate_arranged_draw_order(&order.indices, items.len()) {
        debug_assert!(
            order.indices.is_empty(),
            "arrangement order must cover every draw exactly once"
        );
        return (items, stats);
    }
    (
        apply_validated_arranged_draw_order(items, &order.indices),
        stats,
    )
}

/// Returns whether `order` can be safely applied to `draw_count` flattened draws.
pub(super) fn validate_arranged_draw_order(order: &[usize], draw_count: usize) -> bool {
    if order.len() != draw_count {
        return false;
    }
    let mut seen = vec![false; draw_count];
    for &index in order {
        let Some(slot) = seen.get_mut(index) else {
            return false;
        };
        if *slot {
            return false;
        }
        *slot = true;
    }
    true
}

/// Applies a prevalidated draw order by moving each item exactly once.
pub(super) fn apply_validated_arranged_draw_order(
    items: Vec<WorldMeshDrawItem>,
    order: &[usize],
) -> Vec<WorldMeshDrawItem> {
    profiling::scope!("mesh::arrange_draws_by_phase_bins::materialize");
    let mut slots = items.into_iter().map(Some).collect::<Vec<_>>();
    let mut arranged = Vec::with_capacity(order.len());
    for &index in order {
        if let Some(item) = slots.get_mut(index).and_then(Option::take) {
            arranged.push(item);
        }
    }
    arranged
}

/// Builds compact row vectors from the flattened draw list.
fn build_arrangement_rows(
    items: &[WorldMeshDrawItem],
    batch_ids: &BatchIdTable,
) -> (Vec<NonTransparentDrawRow>, Vec<StrictDrawRow>) {
    profiling::scope!("mesh::arrange_draws_by_phase_bins::build_rows");
    let mut nontransparent_rows = Vec::with_capacity(items.len());
    let mut strict_rows = Vec::new();
    for (source_index, item) in items.iter().enumerate() {
        let classification = classify_world_mesh_batch(&item.batch_key);
        let batch_id = batch_ids.id_for_index(source_index);
        if classification.strict_order {
            strict_rows.push(StrictDrawRow {
                source_index,
                batch_id,
            });
        } else {
            nontransparent_rows.push(NonTransparentDrawRow {
                key: NonTransparentBinKey::from_draw(item, classification.phase, batch_id),
                source_index,
            });
        }
    }
    (nontransparent_rows, strict_rows)
}

/// Sorts nontransparent rows by compact bin key while preserving in-bin collection order.
fn sort_nontransparent_rows(rows: &mut [NonTransparentDrawRow], allow_parallel: bool) {
    profiling::scope!("mesh::arrange_draws_by_phase_bins::sort_nontransparent_rows");
    if allow_parallel && rows.len() >= ARRANGE_PARALLEL_MIN_BINS {
        rows.par_sort_unstable_by(cmp_nontransparent_rows);
    } else {
        rows.sort_unstable_by(cmp_nontransparent_rows);
    }
}

/// Compares compact nontransparent rows.
fn cmp_nontransparent_rows(a: &NonTransparentDrawRow, b: &NonTransparentDrawRow) -> Ordering {
    cmp_nontransparent_bin_keys(&a.key, &b.key).then(a.source_index.cmp(&b.source_index))
}

/// Sorts strict order-sensitive rows through the existing draw comparator.
fn sort_strict_rows(rows: &mut [StrictDrawRow], items: &[WorldMeshDrawItem], allow_parallel: bool) {
    profiling::scope!("mesh::arrange_draws_by_phase_bins::sort_strict_rows");
    if allow_parallel && rows.len() >= ARRANGE_PARALLEL_MIN_DRAWS {
        rows.par_sort_unstable_by(|a, b| {
            cmp_order_sensitive_draws(&items[a.source_index], &items[b.source_index])
        });
    } else {
        rows.sort_unstable_by(|a, b| {
            cmp_order_sensitive_draws(&items[a.source_index], &items[b.source_index])
        });
    }
}

/// Appends pre-skybox nontransparent bins and returns post-skybox bins for ordered merging.
fn append_pre_skybox_bins_and_collect_tail(
    rows: &[NonTransparentDrawRow],
    indices: &mut Vec<usize>,
) -> (usize, Vec<TailBinRow>) {
    profiling::scope!("mesh::arrange_draws_by_phase_bins::flatten_nontransparent_bins");
    let mut tail_bins = Vec::new();
    let mut bin_count = 0usize;
    let mut start = 0usize;
    while start < rows.len() {
        let key = rows[start].key;
        let mut end = start + 1;
        while end < rows.len() && rows[end].key == key {
            end += 1;
        }
        bin_count = bin_count.saturating_add(1);
        if key.phase_rank < post_skybox_rank() {
            append_nontransparent_row_range(indices, rows, start, end);
        } else {
            tail_bins.push(TailBinRow { key, start, end });
        }
        start = end;
    }
    (bin_count, tail_bins)
}

/// Appends a contiguous row range into final draw order.
fn append_nontransparent_row_range(
    indices: &mut Vec<usize>,
    rows: &[NonTransparentDrawRow],
    start: usize,
    end: usize,
) {
    indices.extend(rows[start..end].iter().map(|row| row.source_index));
}

/// Arranges collected draws with bins for nontransparent phases and strict sorting for the
/// transparent tail.
#[cfg(test)]
pub(super) fn arrange_draws_by_phase_bins(
    items: &mut Vec<WorldMeshDrawItem>,
    allow_parallel_sort: bool,
) -> WorldMeshDrawArrangementStats {
    profiling::scope!("mesh::arrange_draws_by_phase_bins");
    if items.is_empty() {
        return WorldMeshDrawArrangementStats::default();
    }

    let input = std::mem::take(items);
    let (arranged, stats) =
        arrange_draw_chunks_by_phase_bins_impl(vec![input], allow_parallel_sort, false);
    *items = arranged;
    stats
}

/// Arranges collected draw chunks with bins for nontransparent phases and strict sorting for the
/// transparent tail.
pub(super) fn arrange_draw_chunks_by_phase_bins(
    chunks: Vec<Vec<WorldMeshDrawItem>>,
    allow_parallel_sort: bool,
) -> (Vec<WorldMeshDrawItem>, WorldMeshDrawArrangementStats) {
    arrange_draw_chunks_by_phase_bins_impl(chunks, allow_parallel_sort, true)
}

/// Shared chunked draw arrangement implementation.
fn arrange_draw_chunks_by_phase_bins_impl(
    chunks: Vec<Vec<WorldMeshDrawItem>>,
    allow_parallel_sort: bool,
    assign_collect_order: bool,
) -> (Vec<WorldMeshDrawItem>, WorldMeshDrawArrangementStats) {
    profiling::scope!("mesh::arrange_draws_by_phase_bins");
    let items = flatten_draw_chunks(chunks, assign_collect_order);
    if items.is_empty() {
        return (Vec::new(), WorldMeshDrawArrangementStats::default());
    }
    let order = arrange_draw_items_order(&items, allow_parallel_sort);
    materialize_arranged_draw_order(items, &order)
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
        .then(a.phase_rank.cmp(&b.phase_rank))
        .then(a.render_queue.cmp(&b.render_queue))
        .then(a.stack.is_some().cmp(&b.stack.is_some()))
        .then_with(|| cmp_nontransparent_stack_keys(a.stack.as_ref(), b.stack.as_ref()))
        .then(a.batch_id.cmp(&b.batch_id))
        .then(a.mesh_asset_id.cmp(&b.mesh_asset_id))
        .then(a.first_index.cmp(&b.first_index))
        .then(a.index_count.cmp(&b.index_count))
}

/// Orders material-stack bins by source renderer, reused submesh, and ascending material slot.
fn cmp_nontransparent_stack_keys(
    a: Option<&NonTransparentStackBinKey>,
    b: Option<&NonTransparentStackBinKey>,
) -> Ordering {
    let (Some(a), Some(b)) = (a, b) else {
        return Ordering::Equal;
    };
    a.space_id
        .cmp(&b.space_id)
        .then(a.skinned.cmp(&b.skinned))
        .then(a.renderable_index.cmp(&b.renderable_index))
        .then(a.instance_id.cmp(&b.instance_id))
        .then(a.mesh_asset_id.cmp(&b.mesh_asset_id))
        .then(a.first_index.cmp(&b.first_index))
        .then(a.index_count.cmp(&b.index_count))
        .then(a.first_stacked_slot_index.cmp(&b.first_stacked_slot_index))
        .then(a.slot_index.cmp(&b.slot_index))
}

/// Stable rank where post-skybox work starts.
#[inline]
fn post_skybox_rank() -> u8 {
    phase_flatten_rank(WorldMeshPhase::Transparent)
}

/// Appends post-skybox bins and strict-order rows in their shared queue order.
fn append_post_skybox_tail_indices(
    indices: &mut Vec<usize>,
    tail_bins: &[TailBinRow],
    nontransparent_rows: &[NonTransparentDrawRow],
    strict_rows: &[StrictDrawRow],
    items: &[WorldMeshDrawItem],
) {
    profiling::scope!("mesh::arrange_draws_by_phase_bins::flatten_tail");
    let mut bin_index = 0usize;
    let mut strict_index = 0usize;
    loop {
        let append_bin = match (tail_bins.get(bin_index), strict_rows.get(strict_index)) {
            (Some(bin), Some(strict_row)) => {
                cmp_nontransparent_bin_to_strict_draw(&bin.key, strict_row, items)
                    != Ordering::Greater
            }
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => break,
        };

        if append_bin {
            let Some(bin) = tail_bins.get(bin_index) else {
                break;
            };
            append_nontransparent_row_range(indices, nontransparent_rows, bin.start, bin.end);
            bin_index += 1;
        } else {
            let Some(row) = strict_rows.get(strict_index) else {
                break;
            };
            indices.push(row.source_index);
            strict_index += 1;
        }
    }
}

/// Compares one nontransparent post-skybox bin against an order-sensitive draw.
fn cmp_nontransparent_bin_to_strict_draw(
    bin: &NonTransparentBinKey,
    row: &StrictDrawRow,
    items: &[WorldMeshDrawItem],
) -> Ordering {
    let item = &items[row.source_index];
    bin.is_overlay
        .cmp(&item.is_overlay)
        .then(bin.render_queue.cmp(&item.batch_key.render_queue))
        .then(false.cmp(&item.batch_key.uses_transparent_sorting()))
        .then(bin.batch_id.cmp(&row.batch_id))
        .then(bin.mesh_asset_id.cmp(&item.mesh_asset_id))
        .then(bin.first_index.cmp(&item.first_index))
        .then(bin.index_count.cmp(&item.index_count))
        .then(Ordering::Less)
}

#[cfg(test)]
mod tests {
    use crate::materials::{
        UNITY_RENDER_QUEUE_ALPHA_TEST, UNITY_RENDER_QUEUE_TRANSPARENT,
        UNITY_TRANSPARENT_RENDER_QUEUE_MIN,
    };
    use crate::scene::MeshRendererInstanceId;
    use crate::world_mesh::draw_prep::item::MaterialStackOrder;
    use crate::world_mesh::draw_prep::pack_sort_prefix;
    use crate::world_mesh::materials::compute_batch_key_hash;
    use crate::world_mesh::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};

    use crate::world_mesh::WorldMeshDrawItem;

    use super::{
        ARRANGE_PARALLEL_MIN_DRAWS, arrange_draw_chunks_by_phase_bins, arrange_draws_by_phase_bins,
    };

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
            item.batch_key.uses_transparent_sorting(),
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

    /// Marks a draw as one layer of the same two-submesh, three-material stack.
    fn mark_stacked_layer(item: &mut WorldMeshDrawItem, slot_index: usize) {
        item.node_id = 50;
        item.renderable_index = 7;
        item.instance_id = MeshRendererInstanceId(7);
        item.slot_index = slot_index;
        item.material_stack_order = MaterialStackOrder::from_slot_counts(slot_index, 3, 2);
        item.first_index = 3;
        item.index_count = 6;
    }

    /// Captures the fields that define arranged draw order for these tests.
    fn arranged_signature(items: &[WorldMeshDrawItem]) -> Vec<(usize, i32, i32, bool, bool)> {
        items
            .iter()
            .map(|item| {
                (
                    item.collect_order,
                    item.mesh_asset_id,
                    item.batch_key.material_asset_id,
                    item.batch_key.uses_transparent_sorting(),
                    item.batch_key.embedded_requires_intersection_pass,
                )
            })
            .collect()
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
    fn nontransparent_stacked_layers_preserve_slot_order_across_material_bins() {
        let mut first_layer = opaque(10, 100, 0);
        mark_stacked_layer(&mut first_layer, 1);
        let mut second_layer = opaque(10, 200, 1);
        mark_stacked_layer(&mut second_layer, 2);

        let mut draws = vec![second_layer, first_layer];
        let stats = arrange_draws_by_phase_bins(&mut draws, false);

        assert_eq!(stats.nontransparent_binned_draws, 2);
        assert_eq!(
            draws.iter().map(|item| item.slot_index).collect::<Vec<_>>(),
            vec![1, 2]
        );
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
        assert!(draws[2].batch_key.uses_transparent_sorting());
    }

    #[test]
    fn geometry_last_queue_bins_before_transparent_tail_without_transparent_sorting() {
        let mut alpha = dummy_world_mesh_draw_item(DummyDrawItemSpec {
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
        set_render_queue(&mut alpha, UNITY_TRANSPARENT_RENDER_QUEUE_MIN);
        set_camera_distance(&mut alpha, 16.0);

        let mut geometry_last = opaque(1, 2, 1);
        geometry_last.batch_key.blend_mode = crate::materials::MaterialBlendMode::Opaque;
        set_render_queue(&mut geometry_last, UNITY_TRANSPARENT_RENDER_QUEUE_MIN - 1);

        let mut transparent = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 3,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 3,
            slot_index: 0,
            collect_order: 2,
            alpha_blended: true,
        });
        set_render_queue(&mut transparent, UNITY_RENDER_QUEUE_TRANSPARENT);
        set_camera_distance(&mut transparent, 4.0);

        let mut draws = vec![transparent, geometry_last, alpha];
        let stats = arrange_draws_by_phase_bins(&mut draws, false);

        assert_eq!(stats.nontransparent_binned_draws, 1);
        assert_eq!(stats.strict_sorted_draws, 2);
        assert_eq!(
            draws
                .iter()
                .map(|item| item.batch_key.render_queue)
                .collect::<Vec<_>>(),
            vec![
                UNITY_TRANSPARENT_RENDER_QUEUE_MIN - 1,
                UNITY_TRANSPARENT_RENDER_QUEUE_MIN,
                UNITY_RENDER_QUEUE_TRANSPARENT,
            ]
        );
        assert!(!draws[0].batch_key.uses_transparent_sorting());
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
    fn transparent_intersection_draws_share_transparent_tail_order() {
        let mut intersect_near = dummy_world_mesh_draw_item(DummyDrawItemSpec {
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
        intersect_near.batch_key.embedded_requires_intersection_pass = true;
        intersect_near.batch_key.embedded_uses_scene_depth_snapshot = true;
        refresh_keys(&mut intersect_near);
        set_camera_distance(&mut intersect_near, 4.0);

        let mut transparent_far = dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 2,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 2,
            slot_index: 0,
            collect_order: 1,
            alpha_blended: true,
        });
        set_camera_distance(&mut transparent_far, 64.0);

        let mut draws = vec![intersect_near, transparent_far];
        let stats = arrange_draws_by_phase_bins(&mut draws, false);

        assert_eq!(stats.nontransparent_binned_draws, 0);
        assert_eq!(stats.strict_sorted_draws, 2);
        assert!(!draws[0].batch_key.embedded_requires_intersection_pass);
        assert!(draws[1].batch_key.embedded_requires_intersection_pass);
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

    #[test]
    fn parallel_partition_matches_serial_arrangement() {
        let mut serial = (0..ARRANGE_PARALLEL_MIN_DRAWS + 64)
            .map(|idx| {
                let mut item = opaque((idx % 23) as i32, (idx % 31) as i32, idx);
                if idx % 11 == 0 {
                    set_render_queue(&mut item, UNITY_RENDER_QUEUE_TRANSPARENT);
                    set_camera_distance(&mut item, (idx % 97) as f32 + 1.0);
                } else if idx % 7 == 0 {
                    set_render_queue(&mut item, UNITY_RENDER_QUEUE_ALPHA_TEST);
                }
                if idx % 17 == 0 {
                    item.batch_key.embedded_requires_intersection_pass = true;
                    refresh_keys(&mut item);
                }
                item
            })
            .collect::<Vec<_>>();
        let mut parallel = serial.clone();

        let serial_stats = arrange_draws_by_phase_bins(&mut serial, false);
        let parallel_stats = arrange_draws_by_phase_bins(&mut parallel, true);

        assert_eq!(parallel_stats, serial_stats);
        assert_eq!(arranged_signature(&parallel), arranged_signature(&serial));
    }

    #[test]
    fn chunked_arrangement_assigns_collect_order_across_chunks() {
        let chunks = vec![
            vec![opaque(10, 1, 99), opaque(10, 1, 98)],
            vec![opaque(10, 1, 97), opaque(10, 1, 96)],
        ];

        let (draws, stats) = arrange_draw_chunks_by_phase_bins(chunks, false);

        assert_eq!(stats.nontransparent_bins, 1);
        assert_eq!(stats.nontransparent_binned_draws, 4);
        assert_eq!(
            draws
                .iter()
                .map(|item| item.collect_order)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
    }

    #[test]
    fn chunked_parallel_arrangement_matches_chunked_serial_arrangement() {
        let source = (0..ARRANGE_PARALLEL_MIN_DRAWS + 96)
            .map(|idx| {
                let mut item = opaque((idx % 19) as i32, (idx % 29) as i32, idx);
                if idx % 13 == 0 {
                    set_render_queue(&mut item, UNITY_RENDER_QUEUE_TRANSPARENT);
                    set_camera_distance(&mut item, (idx % 89) as f32 + 1.0);
                } else if idx % 5 == 0 {
                    set_render_queue(&mut item, UNITY_RENDER_QUEUE_ALPHA_TEST);
                }
                if idx % 23 == 0 {
                    item.batch_key.embedded_uses_scene_color_snapshot = true;
                    refresh_keys(&mut item);
                }
                item
            })
            .collect::<Vec<_>>();
        let chunks = source
            .chunks(37)
            .map(|chunk| chunk.to_vec())
            .collect::<Vec<_>>();

        let (serial, serial_stats) = arrange_draw_chunks_by_phase_bins(chunks.clone(), false);
        let (parallel, parallel_stats) = arrange_draw_chunks_by_phase_bins(chunks, true);

        assert_eq!(parallel_stats, serial_stats);
        assert_eq!(arranged_signature(&parallel), arranged_signature(&serial));
    }
}
