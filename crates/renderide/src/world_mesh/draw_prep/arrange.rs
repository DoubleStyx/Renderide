//! Phase-binned draw arrangement before world-mesh instance planning.

use std::cmp::Ordering;

use hashbrown::HashMap;
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
    fn from_draw(
        item: &WorldMeshDrawItem,
        phase: WorldMeshPhase,
        batch_id: u32,
        surface_stacks: &NonTransparentSurfaceStackTable,
    ) -> Self {
        Self {
            is_overlay: item.is_overlay,
            phase,
            phase_rank: phase_flatten_rank(phase),
            render_queue: item.batch_key.render_queue,
            stack: NonTransparentStackBinKey::from_draw(item, phase, surface_stacks),
            batch_id,
            mesh_asset_id: item.mesh_asset_id,
            first_index: item.first_index,
            index_count: item.index_count,
        }
    }
}

/// Draw-surface identity used to find equal-depth nontransparent renderer stacks.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct NonTransparentSurfaceStackKey {
    /// Main-layer draws sort before overlay draws.
    is_overlay: bool,
    /// Primary render phase for the surface.
    phase: WorldMeshPhase,
    /// Effective Unity render queue.
    render_queue: i32,
    /// Host sorting order carried by the mesh renderer.
    sorting_order: i32,
    /// Host render space id.
    space_id: crate::scene::RenderSpaceId,
    /// Whether this key points at a skinned renderer table.
    skinned: bool,
    /// Scene transform node id.
    node_id: i32,
    /// Resident mesh asset id.
    mesh_asset_id: i32,
    /// First index in the submesh range.
    first_index: u32,
    /// Number of indices in the submesh range.
    index_count: u32,
}

impl NonTransparentSurfaceStackKey {
    /// Builds a same-surface key for one nontransparent draw item.
    fn from_draw(item: &WorldMeshDrawItem, phase: WorldMeshPhase) -> Self {
        Self {
            is_overlay: item.is_overlay,
            phase,
            render_queue: item.batch_key.render_queue,
            sorting_order: item.sorting_order,
            space_id: item.space_id,
            skinned: item.skinned,
            node_id: item.node_id,
            mesh_asset_id: item.mesh_asset_id,
            first_index: item.first_index,
            index_count: item.index_count,
        }
    }
}

/// Per-arrangement duplicate-surface lookup for nontransparent renderer stacks.
#[derive(Debug, Default)]
struct NonTransparentSurfaceStackTable {
    /// Number of nontransparent draws that share each surface key.
    counts: HashMap<NonTransparentSurfaceStackKey, usize>,
}

impl NonTransparentSurfaceStackTable {
    /// Builds duplicate-surface counts from flattened draw items.
    fn build_from_items(items: &[WorldMeshDrawItem]) -> Self {
        profiling::scope!("mesh::arrange_draws_by_phase_bins::surface_stacks");
        let mut counts = HashMap::with_capacity(items.len().min(1_024));
        collect_surface_stack_counts_into(items, &mut counts);
        Self { counts }
    }

    /// Returns whether this draw surface has multiple nontransparent layers.
    #[inline]
    fn is_stacked_surface(&self, key: &NonTransparentSurfaceStackKey) -> bool {
        self.counts.get(key).copied().unwrap_or(0) > 1
    }
}

/// Ordering key for a nontransparent same-surface bin.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NonTransparentStackBinKey {
    /// Surface whose equal-depth layers must keep Unity-style renderer order.
    surface: NonTransparentSurfaceStackKey,
    /// Dense renderer index inside the selected renderer table.
    renderable_index: usize,
    /// Renderer-local stable identity.
    instance_id: u64,
    /// First material slot participating in a single-renderer material stack.
    first_stacked_slot_index: usize,
    /// Material slot represented by this bin.
    slot_index: usize,
}

impl NonTransparentStackBinKey {
    /// Builds a stack bin key for one draw item when its equal-depth order is visible.
    fn from_draw(
        item: &WorldMeshDrawItem,
        phase: WorldMeshPhase,
        surface_stacks: &NonTransparentSurfaceStackTable,
    ) -> Option<Self> {
        let surface = NonTransparentSurfaceStackKey::from_draw(item, phase);
        let material_stack_slot = item
            .material_stack_order
            .map(|stack| stack.first_stacked_slot_index);
        if material_stack_slot.is_none() && !surface_stacks.is_stacked_surface(&surface) {
            return None;
        }
        Some(Self {
            surface,
            renderable_index: item.renderable_index,
            instance_id: item.instance_id.0,
            first_stacked_slot_index: material_stack_slot.unwrap_or(item.slot_index),
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
    let surface_stacks = NonTransparentSurfaceStackTable::build_from_items(items);
    let (mut nontransparent_rows, mut strict_rows) =
        build_arrangement_rows(items, &batch_ids, &surface_stacks);

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
    surface_stacks: &NonTransparentSurfaceStackTable,
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
                key: NonTransparentBinKey::from_draw(
                    item,
                    classification.phase,
                    batch_id,
                    surface_stacks,
                ),
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

/// Adds one draw chunk's nontransparent surface-stack counts into `counts`.
fn collect_surface_stack_counts_into(
    items: &[WorldMeshDrawItem],
    counts: &mut HashMap<NonTransparentSurfaceStackKey, usize>,
) {
    for item in items {
        let classification = classify_world_mesh_batch(&item.batch_key);
        if classification.strict_order {
            continue;
        }
        let key = NonTransparentSurfaceStackKey::from_draw(item, classification.phase);
        counts
            .entry(key)
            .and_modify(|count| *count = count.saturating_add(1))
            .or_insert(1);
    }
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
    cmp_nontransparent_surface_stack_keys(&a.surface, &b.surface)
        .then(a.renderable_index.cmp(&b.renderable_index))
        .then(a.instance_id.cmp(&b.instance_id))
        .then(a.first_stacked_slot_index.cmp(&b.first_stacked_slot_index))
        .then(a.slot_index.cmp(&b.slot_index))
}

/// Orders equal-depth surface-stack identities in the same hierarchy as nontransparent bins.
fn cmp_nontransparent_surface_stack_keys(
    a: &NonTransparentSurfaceStackKey,
    b: &NonTransparentSurfaceStackKey,
) -> Ordering {
    a.is_overlay
        .cmp(&b.is_overlay)
        .then_with(|| phase_flatten_rank(a.phase).cmp(&phase_flatten_rank(b.phase)))
        .then(a.render_queue.cmp(&b.render_queue))
        .then(a.sorting_order.cmp(&b.sorting_order))
        .then(a.space_id.cmp(&b.space_id))
        .then(a.skinned.cmp(&b.skinned))
        .then(a.node_id.cmp(&b.node_id))
        .then(a.mesh_asset_id.cmp(&b.mesh_asset_id))
        .then(a.first_index.cmp(&b.first_index))
        .then(a.index_count.cmp(&b.index_count))
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
mod tests;
