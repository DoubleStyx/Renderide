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
        batch_ids: &BatchIdTable,
    ) -> Self {
        Self {
            is_overlay: item.is_overlay,
            phase,
            render_queue: item.batch_key.render_queue,
            stack: NonTransparentStackBinKey::from_draw(item),
            batch_id: batch_ids.id_for_draw(item),
            mesh_asset_id: item.mesh_asset_id,
            first_index: item.first_index,
            index_count: item.index_count,
        }
    }
}

/// Ordering key for a nontransparent material-stack bin.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
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
    let (bins, mut strict_ordered) =
        partition_draw_chunks(chunks, &batch_ids, allow_parallel_sort, draw_count);

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
            .map(|chunk| partition_draw_chunk(chunk, batch_ids))
            .collect::<Vec<_>>()
    } else {
        profiling::scope!("mesh::arrange_draws_by_phase_bins::serial_partition");
        chunks
            .into_iter()
            .map(|chunk| partition_draw_chunk(chunk, batch_ids))
            .collect::<Vec<_>>()
    };
    merge_partitioned_chunks(partitioned)
}

/// Partitions one draw chunk into phase bins on the caller thread.
fn partition_draw_chunk(
    input: Vec<WorldMeshDrawItem>,
    batch_ids: &BatchIdTable,
) -> PartitionedDrawChunk {
    let mut bins: HashMap<NonTransparentBinKey, Vec<WorldMeshDrawItem>> =
        HashMap::with_capacity(input.len().min(1_024));
    let mut strict_ordered = Vec::new();
    for item in input {
        partition_draw_item(item, batch_ids, &mut bins, &mut strict_ordered);
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
