//! Phase-binned draw arrangement before world-mesh instance planning.

use std::cmp::Ordering;

use hashbrown::HashMap;
use rayon::prelude::*;

use crate::cpu_parallelism::{
    ParallelAdmission, current_reference_worker_count, record_parallel_admission,
};
use crate::world_mesh::MaterialDrawBatchKey;
use crate::world_mesh::WorldMeshPhase;
use crate::world_mesh::phase_classification::classify_world_mesh_batch;

use super::item::{WorldMeshDrawArrangementStats, WorldMeshDrawItem};
use super::sort::sort_order_sensitive_draws;

/// Draws assigned to one phase-partition worker chunk.
const ARRANGE_PARALLEL_CHUNK_DRAWS: usize = 128;

/// Draw count at which phase partitioning uses Rayon workers.
///
/// Partitioning builds worker-local maps and then merges them, so this remains more conservative
/// than simple per-renderer fan-out while still covering medium draw lists.
const ARRANGE_PARALLEL_MIN_DRAWS: usize = ARRANGE_PARALLEL_CHUNK_DRAWS * 2;

/// Draw chunks assigned to one arrangement worker.
const ARRANGE_PARALLEL_CHUNK_TASKS: usize = 1;

/// Bin count at which bin-key sorting uses Rayon workers.
const ARRANGE_PARALLEL_MIN_BINS: usize = 512;

/// Key for one nontransparent bin.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct NonTransparentBinKey {
    /// Main-layer draws sort before overlay draws.
    is_overlay: bool,
    /// Primary render phase for the bin.
    phase: WorldMeshPhase,
    /// Effective Unity render queue.
    render_queue: i32,
    /// Surface-stack ordering key for draws whose equal-depth order is visible.
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
        batch_ids: &BatchIdTable,
        surface_stacks: &NonTransparentSurfaceStackTable,
    ) -> Self {
        Self {
            is_overlay: item.is_overlay,
            phase,
            render_queue: item.batch_key.render_queue,
            stack: NonTransparentStackBinKey::from_draw(item, phase, surface_stacks),
            batch_id: batch_ids.id_for_draw(item),
            mesh_asset_id: item.mesh_asset_id,
            first_index: item.first_index,
            index_count: item.index_count,
        }
    }
}

/// Draw-surface identity used to find equal-depth nontransparent renderer stacks.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
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
    /// Builds duplicate-surface counts from deterministic draw chunks.
    fn build_from_chunks(
        chunks: &[Vec<WorldMeshDrawItem>],
        allow_parallel: bool,
        draw_count: usize,
    ) -> Self {
        profiling::scope!("mesh::arrange_draws_by_phase_bins::surface_stacks");
        let admission = arrange_chunk_admission(
            draw_count,
            chunks.len(),
            current_reference_worker_count(),
            allow_parallel,
        );
        record_parallel_admission(
            "world_mesh_arrange_surface_stacks",
            draw_count,
            chunks.len(),
            admission,
        );
        let counts = if admission.is_parallel() {
            profiling::scope!("mesh::arrange_draws_by_phase_bins::surface_stacks_parallel");
            chunks
                .par_iter()
                .with_min_len(ARRANGE_PARALLEL_CHUNK_TASKS)
                .map(|chunk| collect_surface_stack_counts(chunk))
                .reduce(HashMap::new, |mut target, source| {
                    merge_surface_stack_counts(&mut target, source);
                    target
                })
        } else {
            profiling::scope!("mesh::arrange_draws_by_phase_bins::surface_stacks_serial");
            let mut counts = HashMap::with_capacity(draw_count.min(1_024));
            for chunk in chunks {
                collect_surface_stack_counts_into(chunk, &mut counts);
            }
            counts
        };
        Self { counts }
    }

    /// Returns whether this draw surface has multiple nontransparent layers.
    #[inline]
    fn is_stacked_surface(&self, key: &NonTransparentSurfaceStackKey) -> bool {
        self.counts.get(key).copied().unwrap_or(0) > 1
    }
}

/// Ordering key for a nontransparent same-surface bin.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct NonTransparentStackBinKey {
    /// Surface whose equal-depth layers must keep Unity-style renderer order.
    surface: NonTransparentSurfaceStackKey,
    /// First material slot participating in a single-renderer material stack.
    first_stacked_slot_index: usize,
    /// Renderer-local stable identity, assigned when the renderer entry was created.
    instance_id: u64,
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
            first_stacked_slot_index: material_stack_slot.unwrap_or(item.slot_index),
            instance_id: item.instance_id.0,
            slot_index: item.slot_index,
        })
    }
}

/// Per-arrangement compact IDs for material and pipeline batch keys.
#[derive(Debug, Default)]
struct BatchIdTable {
    /// Stable ID lookup by resolved material batch key.
    ids: HashMap<MaterialDrawBatchKey, u32>,
    /// Dense per-draw batch ids indexed by [`WorldMeshDrawItem::collect_order`].
    draw_ids: Option<Vec<u32>>,
}

impl BatchIdTable {
    /// Builds compact batch IDs from draw chunks.
    fn build_from_chunks(
        chunks: &[Vec<WorldMeshDrawItem>],
        allow_parallel: bool,
        build_dense_draw_ids: bool,
    ) -> Self {
        profiling::scope!("mesh::arrange_draws_by_phase_bins::batch_ids");
        let draw_count = chunks.iter().map(Vec::len).sum::<usize>();
        let admission = arrange_chunk_admission(
            draw_count,
            chunks.len(),
            current_reference_worker_count(),
            allow_parallel,
        );
        record_parallel_admission(
            "world_mesh_arrange_batch_ids",
            draw_count,
            chunks.len(),
            admission,
        );
        let unique = if admission.is_parallel() {
            profiling::scope!("mesh::arrange_draws_by_phase_bins::batch_ids_parallel");
            chunks
                .par_iter()
                .with_min_len(ARRANGE_PARALLEL_CHUNK_TASKS)
                .map(|chunk| collect_unique_batch_ids(chunk))
                .reduce(HashMap::new, |mut target, source| {
                    merge_unique_batch_ids(&mut target, source);
                    target
                })
        } else {
            profiling::scope!("mesh::arrange_draws_by_phase_bins::batch_ids_serial");
            let mut unique = HashMap::with_capacity(draw_count.min(1_024));
            for chunk in chunks {
                for item in chunk {
                    unique
                        .entry(item.batch_key.clone())
                        .or_insert(item.batch_key_hash);
                }
            }
            unique
        };
        let mut table = Self::from_unique(unique);
        if build_dense_draw_ids {
            table.populate_dense_draw_ids(chunks, draw_count);
        }
        table
    }

    /// Builds compact batch IDs from an already-deduplicated key map.
    fn from_unique(unique: HashMap<MaterialDrawBatchKey, u64>) -> Self {
        let mut ordered = unique.into_iter().collect::<Vec<_>>();
        ordered.sort_unstable_by(|(a_key, a_hash), (b_key, b_hash)| {
            a_hash.cmp(b_hash).then_with(|| a_key.cmp(b_key))
        });
        let mut ids = HashMap::with_capacity(ordered.len());
        for (index, (key, _)) in ordered.into_iter().enumerate() {
            ids.insert(key, index.min(u32::MAX as usize) as u32);
        }
        Self {
            ids,
            draw_ids: None,
        }
    }

    /// Precomputes draw-local batch ids after collection order has been assigned densely.
    fn populate_dense_draw_ids(&mut self, chunks: &[Vec<WorldMeshDrawItem>], draw_count: usize) {
        profiling::scope!("mesh::arrange_draws_by_phase_bins::dense_batch_ids");
        let mut draw_ids = vec![u32::MAX; draw_count];
        for chunk in chunks {
            for item in chunk {
                let Some(slot) = draw_ids.get_mut(item.collect_order) else {
                    self.draw_ids = None;
                    return;
                };
                *slot = self.ids.get(&item.batch_key).copied().unwrap_or(u32::MAX);
            }
        }
        self.draw_ids = Some(draw_ids);
    }

    /// Returns the compact batch ID for a draw item.
    #[inline]
    fn id_for_draw(&self, item: &WorldMeshDrawItem) -> u32 {
        if let Some(draw_ids) = &self.draw_ids
            && let Some(&id) = draw_ids.get(item.collect_order)
        {
            return id;
        }
        self.ids.get(&item.batch_key).copied().unwrap_or(u32::MAX)
    }
}

/// Worker-local partition result for one draw chunk.
#[derive(Debug, Default)]
struct PartitionedDrawChunk {
    /// Nontransparent bins produced by this chunk.
    bins: HashMap<NonTransparentBinKey, Vec<WorldMeshDrawItem>>,
    /// Strict order-sensitive draws produced by this chunk.
    strict_ordered: Vec<WorldMeshDrawItem>,
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
    mut chunks: Vec<Vec<WorldMeshDrawItem>>,
    allow_parallel_sort: bool,
    assign_collect_order: bool,
) -> (Vec<WorldMeshDrawItem>, WorldMeshDrawArrangementStats) {
    profiling::scope!("mesh::arrange_draws_by_phase_bins");
    let draw_count = chunks.iter().map(Vec::len).sum::<usize>();
    if draw_count == 0 {
        return (Vec::new(), WorldMeshDrawArrangementStats::default());
    }
    if assign_collect_order {
        assign_chunk_collect_order(&mut chunks);
    }

    let batch_ids =
        BatchIdTable::build_from_chunks(&chunks, allow_parallel_sort, assign_collect_order);
    let surface_stacks = NonTransparentSurfaceStackTable::build_from_chunks(
        &chunks,
        allow_parallel_sort,
        draw_count,
    );
    let (bins, mut strict_ordered) = partition_draw_chunks(
        chunks,
        &batch_ids,
        &surface_stacks,
        allow_parallel_sort,
        draw_count,
    );

    let mut binned: Vec<_> = bins.into_iter().collect();
    let stats = WorldMeshDrawArrangementStats {
        nontransparent_bins: binned.len(),
        nontransparent_binned_draws: binned.iter().map(|(_, draws)| draws.len()).sum(),
        strict_sorted_draws: strict_ordered.len(),
    };

    {
        profiling::scope!("mesh::arrange_draws_by_phase_bins::sort_bins");
        if allow_parallel_sort && binned.len() >= ARRANGE_PARALLEL_MIN_BINS {
            binned.par_sort_unstable_by(|(a, _), (b, _)| cmp_nontransparent_bin_keys(a, b));
        } else {
            binned.sort_unstable_by(|(a, _), (b, _)| cmp_nontransparent_bin_keys(a, b));
        }
    }
    {
        profiling::scope!("mesh::arrange_draws_by_phase_bins::sort_strict_ordered");
        sort_order_sensitive_draws(&mut strict_ordered, allow_parallel_sort);
    }
    let mut arranged =
        Vec::with_capacity(stats.nontransparent_binned_draws + stats.strict_sorted_draws);
    {
        profiling::scope!("mesh::arrange_draws_by_phase_bins::flatten");
        let tail_start =
            binned.partition_point(|(key, _)| phase_flatten_rank(key.phase) < post_skybox_rank());
        let tail_bins = binned.split_off(tail_start);
        for (_, mut bin_items) in binned {
            arranged.append(&mut bin_items);
        }
        append_post_skybox_tail(&mut arranged, tail_bins, strict_ordered, &batch_ids);
    }

    (arranged, stats)
}

/// Assigns global collection order across deterministic draw chunks.
fn assign_chunk_collect_order(chunks: &mut [Vec<WorldMeshDrawItem>]) {
    profiling::scope!("mesh::arrange_draws_by_phase_bins::assign_collect_order");
    let mut collect_order = 0usize;
    for chunk in chunks {
        for item in chunk {
            item.collect_order = collect_order;
            collect_order += 1;
        }
    }
}

/// Returns the admission decision for chunked draw arrangement work.
fn arrange_chunk_admission(
    draw_count: usize,
    chunk_count: usize,
    worker_count: usize,
    allow_parallel: bool,
) -> ParallelAdmission {
    if allow_parallel
        && worker_count > 1
        && draw_count >= ARRANGE_PARALLEL_MIN_DRAWS
        && chunk_count >= ARRANGE_PARALLEL_CHUNK_TASKS * 2
    {
        ParallelAdmission::Parallel {
            chunk_size: ARRANGE_PARALLEL_CHUNK_TASKS,
        }
    } else {
        ParallelAdmission::Serial
    }
}

/// Collects unique material batch IDs from one draw chunk.
fn collect_unique_batch_ids(chunk: &[WorldMeshDrawItem]) -> HashMap<MaterialDrawBatchKey, u64> {
    let mut unique = HashMap::with_capacity(chunk.len().min(1_024));
    for item in chunk {
        unique
            .entry(item.batch_key.clone())
            .or_insert(item.batch_key_hash);
    }
    unique
}

/// Collects nontransparent surface-stack counts from one draw chunk.
fn collect_surface_stack_counts(
    chunk: &[WorldMeshDrawItem],
) -> HashMap<NonTransparentSurfaceStackKey, usize> {
    let mut counts = HashMap::with_capacity(chunk.len().min(1_024));
    collect_surface_stack_counts_into(chunk, &mut counts);
    counts
}

/// Adds one draw chunk's nontransparent surface-stack counts into `counts`.
fn collect_surface_stack_counts_into(
    chunk: &[WorldMeshDrawItem],
    counts: &mut HashMap<NonTransparentSurfaceStackKey, usize>,
) {
    for item in chunk {
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

/// Merges surface-stack counts from a worker-local map into a target map.
fn merge_surface_stack_counts(
    target: &mut HashMap<NonTransparentSurfaceStackKey, usize>,
    source: HashMap<NonTransparentSurfaceStackKey, usize>,
) {
    for (key, count) in source {
        target
            .entry(key)
            .and_modify(|target_count| {
                *target_count = target_count.saturating_add(count);
            })
            .or_insert(count);
    }
}

/// Merges a source batch-ID map into a target map.
fn merge_unique_batch_ids(
    target: &mut HashMap<MaterialDrawBatchKey, u64>,
    source: HashMap<MaterialDrawBatchKey, u64>,
) {
    for (key, hash) in source {
        target.entry(key).or_insert(hash);
    }
}

/// Partitions draw chunks into phase bins.
fn partition_draw_chunks(
    chunks: Vec<Vec<WorldMeshDrawItem>>,
    batch_ids: &BatchIdTable,
    surface_stacks: &NonTransparentSurfaceStackTable,
    allow_parallel: bool,
    draw_count: usize,
) -> (
    HashMap<NonTransparentBinKey, Vec<WorldMeshDrawItem>>,
    Vec<WorldMeshDrawItem>,
) {
    let admission = arrange_chunk_admission(
        draw_count,
        chunks.len(),
        current_reference_worker_count(),
        allow_parallel,
    );
    record_parallel_admission(
        "world_mesh_arrange_partition",
        draw_count,
        chunks.len(),
        admission,
    );
    let partitioned = if admission.is_parallel() {
        profiling::scope!("mesh::arrange_draws_by_phase_bins::parallel_partition");
        chunks
            .into_par_iter()
            .with_min_len(ARRANGE_PARALLEL_CHUNK_TASKS)
            .map(|chunk| partition_draw_chunk(chunk, batch_ids, surface_stacks))
            .collect::<Vec<_>>()
    } else {
        profiling::scope!("mesh::arrange_draws_by_phase_bins::serial_partition");
        chunks
            .into_iter()
            .map(|chunk| partition_draw_chunk(chunk, batch_ids, surface_stacks))
            .collect::<Vec<_>>()
    };
    merge_partitioned_chunks(partitioned)
}

/// Partitions one draw chunk into phase bins on the caller thread.
fn partition_draw_chunk(
    input: Vec<WorldMeshDrawItem>,
    batch_ids: &BatchIdTable,
    surface_stacks: &NonTransparentSurfaceStackTable,
) -> PartitionedDrawChunk {
    let mut bins: HashMap<NonTransparentBinKey, Vec<WorldMeshDrawItem>> =
        HashMap::with_capacity(input.len().min(1_024));
    let mut strict_ordered = Vec::new();
    for item in input {
        partition_draw_item(
            item,
            batch_ids,
            surface_stacks,
            &mut bins,
            &mut strict_ordered,
        );
    }
    PartitionedDrawChunk {
        bins,
        strict_ordered,
    }
}

/// Merges worker-local partition results in deterministic chunk order.
fn merge_partitioned_chunks(
    chunks: Vec<PartitionedDrawChunk>,
) -> (
    HashMap<NonTransparentBinKey, Vec<WorldMeshDrawItem>>,
    Vec<WorldMeshDrawItem>,
) {
    let mut bins = HashMap::new();
    let mut strict_ordered = Vec::new();
    for mut chunk in chunks {
        merge_bins(&mut bins, chunk.bins);
        strict_ordered.append(&mut chunk.strict_ordered);
    }
    (bins, strict_ordered)
}

/// Routes one draw into either a phase bin or the strict-order tail.
fn partition_draw_item(
    item: WorldMeshDrawItem,
    batch_ids: &BatchIdTable,
    surface_stacks: &NonTransparentSurfaceStackTable,
    bins: &mut HashMap<NonTransparentBinKey, Vec<WorldMeshDrawItem>>,
    strict_ordered: &mut Vec<WorldMeshDrawItem>,
) {
    let classification = classify_world_mesh_batch(&item.batch_key);
    if classification.strict_order {
        strict_ordered.push(item);
    } else {
        bins.entry(NonTransparentBinKey::from_draw(
            &item,
            classification.phase,
            batch_ids,
            surface_stacks,
        ))
        .or_default()
        .push(item);
    }
}

/// Merges worker-local nontransparent bins into the caller-owned destination.
fn merge_bins(
    target: &mut HashMap<NonTransparentBinKey, Vec<WorldMeshDrawItem>>,
    source: HashMap<NonTransparentBinKey, Vec<WorldMeshDrawItem>>,
) {
    for (key, mut items) in source {
        target.entry(key).or_default().append(&mut items);
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
        .then_with(|| phase_flatten_rank(a.phase).cmp(&phase_flatten_rank(b.phase)))
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

/// Appends post-skybox bins and strict-order draws in their shared queue order.
fn append_post_skybox_tail(
    items: &mut Vec<WorldMeshDrawItem>,
    tail_bins: Vec<(NonTransparentBinKey, Vec<WorldMeshDrawItem>)>,
    strict_ordered: Vec<WorldMeshDrawItem>,
    batch_ids: &BatchIdTable,
) {
    let mut bins = tail_bins.into_iter().peekable();
    let mut strict = strict_ordered.into_iter().peekable();
    loop {
        let append_bin = match (bins.peek(), strict.peek()) {
            (Some((bin_key, _)), Some(strict_item)) => {
                cmp_nontransparent_bin_to_strict_draw(bin_key, strict_item, batch_ids)
                    != Ordering::Greater
            }
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => break,
        };

        if append_bin {
            let Some((_, mut bin_items)) = bins.next() else {
                break;
            };
            items.append(&mut bin_items);
        } else {
            let Some(item) = strict.next() else {
                break;
            };
            items.push(item);
        }
    }
}

/// Compares one nontransparent post-skybox bin against an order-sensitive draw.
fn cmp_nontransparent_bin_to_strict_draw(
    bin: &NonTransparentBinKey,
    item: &WorldMeshDrawItem,
    batch_ids: &BatchIdTable,
) -> Ordering {
    bin.is_overlay
        .cmp(&item.is_overlay)
        .then(bin.render_queue.cmp(&item.batch_key.render_queue))
        .then(false.cmp(&item.batch_key.uses_transparent_sorting()))
        .then(bin.batch_id.cmp(&batch_ids.id_for_draw(item)))
        .then(bin.mesh_asset_id.cmp(&item.mesh_asset_id))
        .then(bin.first_index.cmp(&item.first_index))
        .then(bin.index_count.cmp(&item.index_count))
        .then(Ordering::Less)
}

#[cfg(test)]
mod tests;
