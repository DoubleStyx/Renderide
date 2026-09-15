//! Per-render-space CPU spatial index for prepared renderer runs.

use glam::{Mat4, Vec3, Vec3A};
use hashbrown::{HashMap, HashSet};

#[cfg(test)]
use crate::particles::ParticleDrawParams;
use crate::scene::{RenderSpaceId, SceneSpaceRead};
#[cfg(test)]
use crate::shared::ShadowCastMode;
use crate::world_mesh::culling::{WorldMeshCullInput};

use super::super::bitset::DenseBitSet;
use super::super::item::WorldMeshVisibilityStats;
use super::{FramePreparedDraw, FramePreparedRun};

const BVH_LEAF_SIZE: usize = 8;
pub(super) const SPATIAL_LINEAR_RUN_LIMIT: usize = 64;
/// Indexed runs a space needs before its BVH walk is split across two workers.
///
/// The traversal is read-only, and candidate order does not matter because the gather sorts and
/// dedups by run index afterwards. Small spaces stay serial so rayon dispatch cannot cost more
/// than the walk. -xlinka
const SPATIAL_PARALLEL_QUERY_MIN_RUNS: usize = 2048;
const SPATIAL_DENSE_GATHER_MIN_CANDIDATE_DIVISOR: usize = 2;

/// Spatial query output consumed by per-view prepared draw collection.
pub(in crate::world_mesh::draw_prep) struct PreparedSpatialRunCandidates {
    /// Renderer runs that survived spatial filtering, in original prepared-run order.
    pub(in crate::world_mesh::draw_prep) runs: Vec<FramePreparedRun>,
    /// Slot-level cull counters for runs rejected by the spatial index itself.
    pub(in crate::world_mesh::draw_prep) cull_stats: (usize, usize, usize),
    /// Visibility broadphase counters for this query.
    pub(in crate::world_mesh::draw_prep) visibility: WorldMeshVisibilityStats,
}

/// Per-render-space BVH and linear fallback buckets for prepared renderer runs.
#[derive(Clone, Default)]
pub(super) struct PreparedSpatialIndex {
    spaces: HashMap<RenderSpaceId, PreparedSpatialSpace>,
}

impl PreparedSpatialIndex {
    /// Rebuilds all per-space spatial data from the current prepared run snapshot.
    pub(super) fn rebuild(&mut self, draws: &[FramePreparedDraw], runs: &[FramePreparedRun]) {
        profiling::scope!("mesh::prepared_renderables::spatial_rebuild");
        self.spaces.clear();
        let mut builders: HashMap<RenderSpaceId, PreparedSpatialSpaceBuilder> = HashMap::new();
        // Runs are laid out contiguously per space, so carry the current builder instead of probing
        // the map for all ~37k runs on every rebuild. -xlinka
        let mut current: Option<(RenderSpaceId, PreparedSpatialSpaceBuilder)> = None;
        for (run_index, run) in runs.iter().copied().enumerate() {
            let Some(first) = draws.get(run.start as usize) else {
                continue;
            };
            let slot_count = (run.end - run.start) as usize;
            if current.as_ref().is_none_or(|(id, _)| *id != first.space_id) {
                if let Some((id, done)) = current.take() {
                    merge_spatial_builder(&mut builders, id, done);
                }
                let existing = builders.remove(&first.space_id).unwrap_or_default();
                current = Some((first.space_id, existing));
            }
            let builder = current
                .as_mut()
                .map(|(_, builder)| builder)
                .expect("current builder was just set");
            if let Some((aabb_min, aabb_max)) = indexable_run_bounds(first) {
                builder.indexed.push(IndexedPreparedRun {
                    run_index,
                    aabb_min,
                    aabb_max,
                    center: (aabb_min + aabb_max) * 0.5,
                    slot_count,
                });
            } else {
                builder.linear.push(LinearPreparedRun {
                    run_index,
                    bounds: None,
                    slot_count,
                });
            }
        }
        if let Some((id, done)) = current.take() {
            merge_spatial_builder(&mut builders, id, done);
        }
        for (space_id, builder) in builders {
            self.spaces.insert(space_id, builder.finish());
        }
    }

    /// Collects prepared renderer runs for the requested render spaces after optional frustum culling.
    pub(super) fn query_runs(
        &self,
        runs: &[FramePreparedRun],
        space_ids: &[RenderSpaceId],
        scene: &(impl SceneSpaceRead + ?Sized),
        culling: Option<&WorldMeshCullInput<'_>>,
    ) -> PreparedSpatialRunCandidates {
        profiling::scope!("mesh::prepared_renderables::spatial_query");
        let mut candidate_indices = Vec::new();
        let mut raw_candidate_marks = 0usize;
        let mut cull_stats = (0usize, 0usize, 0usize);
        let mut visibility = WorldMeshVisibilityStats::default();
        {
            profiling::scope!("mesh::prepared_renderables::spatial_query_mark_candidates");
            for &space_id in space_ids {
                let Some(space) = self.spaces.get(&space_id) else {
                    continue;
                };
                space.query(
                    space_id,
                    scene,
                    culling,
                    &mut PreparedSpatialQueryOutput {
                        candidates: &mut candidate_indices,
                        raw_candidate_marks: &mut raw_candidate_marks,
                        cull_stats: &mut cull_stats,
                        visibility: &mut visibility,
                    },
                );
            }
        }

        let out = gather_spatial_candidate_runs(runs, &mut candidate_indices);
        let candidate_runs = out.len();
        PreparedSpatialRunCandidates {
            runs: out,
            cull_stats,
            visibility: WorldMeshVisibilityStats {
                candidate_runs,
                raw_candidate_marks,
                duplicate_candidate_marks: raw_candidate_marks.saturating_sub(candidate_runs),
                ..visibility
            },
        }
    }

    /// Refits existing per-space spatial bounds without changing run membership or tree topology.
    #[cfg(test)]
    pub(super) fn refit_spaces<I>(
        &mut self,
        draws: &[FramePreparedDraw],
        runs: &[FramePreparedRun],
        space_ids: I,
    ) -> usize
    where
        I: IntoIterator<Item = RenderSpaceId>,
    {
        self.refit_spaces_for_runs(draws, runs, space_ids, None)
    }

    /// Refits `space_ids`, restricted to `changed_runs` when the caller knows which runs moved.
    ///
    /// `None` keeps the conservative full sweep. `Some` walks only the changed entries and their
    /// BVH ancestors, which is the difference between O(scene) and O(changed) for a patch.
    pub(super) fn refit_spaces_for_runs<I>(
        &mut self,
        draws: &[FramePreparedDraw],
        runs: &[FramePreparedRun],
        space_ids: I,
        changed_runs: Option<&HashSet<usize>>,
    ) -> usize
    where
        I: IntoIterator<Item = RenderSpaceId>,
    {
        let mut refit_count = 0usize;
        for space_id in space_ids {
            let Some(space) = self.spaces.get_mut(&space_id) else {
                continue;
            };
            profiling::scope!("mesh::prepared_renderables::spatial_refit");
            match changed_runs {
                Some(changed) if !changed.is_empty() => {
                    profiling::scope!("mesh::prepared_renderables::spatial_refit_incremental");
                    space.refit_runs(draws, runs, changed);
                }
                Some(_) => {}
                None => space.refit(draws, runs),
            }
            refit_count = refit_count.saturating_add(1);
        }
        refit_count
    }

    /// Returns whether `space_id` uses a BVH instead of only linear buckets.
    #[cfg(test)]
    pub(super) fn space_uses_bvh_for_tests(&self, space_id: RenderSpaceId) -> bool {
        self.spaces
            .get(&space_id)
            .is_some_and(PreparedSpatialSpace::uses_bvh)
    }
}

/// Folds a completed per-space builder back into the map, merging if the space reappeared.
fn merge_spatial_builder(
    builders: &mut HashMap<RenderSpaceId, PreparedSpatialSpaceBuilder>,
    space_id: RenderSpaceId,
    mut done: PreparedSpatialSpaceBuilder,
) {
    match builders.entry(space_id) {
        hashbrown::hash_map::Entry::Occupied(mut slot) => {
            let existing = slot.get_mut();
            existing.indexed.append(&mut done.indexed);
            existing.linear.append(&mut done.linear);
        }
        hashbrown::hash_map::Entry::Vacant(slot) => {
            slot.insert(done);
        }
    }
}

#[derive(Default)]
struct PreparedSpatialSpaceBuilder {
    indexed: Vec<IndexedPreparedRun>,
    linear: Vec<LinearPreparedRun>,
}

impl PreparedSpatialSpaceBuilder {
    fn finish(mut self) -> PreparedSpatialSpace {
        let mut space = PreparedSpatialSpace {
            linear: self.linear,
            indexed: Vec::new(),
            order: Vec::new(),
            nodes: Vec::new(),
            root: None,
            slot_by_run: HashMap::new(),
            node_of_slot: Vec::new(),
            indexed_run_count: self.indexed.len(),
            fallback_run_count: 0,
        };
        space.fallback_run_count = space.linear.len();
        if self.indexed.len() <= SPATIAL_LINEAR_RUN_LIMIT {
            space
                .linear
                .extend(self.indexed.drain(..).map(|entry| LinearPreparedRun {
                    run_index: entry.run_index,
                    bounds: Some((entry.aabb_min, entry.aabb_max)),
                    slot_count: entry.slot_count,
                }));
            return space;
        }

        space.order = (0..self.indexed.len()).collect();
        space.indexed = self.indexed;
        space.slot_by_run = space
            .indexed
            .iter()
            .enumerate()
            .map(|(slot, entry)| (entry.run_index, slot))
            .collect();
        let mut order = std::mem::take(&mut space.order);
        let end = order.len();
        space.root = Some(space.build_node(&mut order, 0, end));
        space.order = order;
        space.node_of_slot = vec![usize::MAX; space.indexed.len()];
        for node_index in 0..space.nodes.len() {
            let node = space.nodes[node_index];
            if node.count == 0 {
                continue;
            }
            for &slot in &space.order[node.start..node.start + node.count] {
                if let Some(owner) = space.node_of_slot.get_mut(slot) {
                    *owner = node_index;
                }
            }
        }
        space
    }
}

/// One render space's BVH plus conservative linear fallback runs.
#[derive(Clone, Default)]
struct PreparedSpatialSpace {
    linear: Vec<LinearPreparedRun>,
    indexed: Vec<IndexedPreparedRun>,
    order: Vec<usize>,
    nodes: Vec<PreparedBvhNode>,
    root: Option<usize>,
    indexed_run_count: usize,
    fallback_run_count: usize,
    /// Indexed-entry slot for a run index, and the leaf node owning each slot.
    ///
    /// Without these the "incremental" refit still scanned every indexed entry to find the changed
    /// ones and every node to find their leaves, so it stayed O(scene) and measured 271us/call in
    /// Darkcity11. -xlinka
    slot_by_run: HashMap<usize, usize>,
    node_of_slot: Vec<usize>,
}

impl PreparedSpatialSpace {
    fn refit(&mut self, draws: &[FramePreparedDraw], runs: &[FramePreparedRun]) {
        for entry in &mut self.linear {
            entry.bounds = refit_linear_bounds(draws, runs, entry);
        }
        for entry in &mut self.indexed {
            let (aabb_min, aabb_max) = refit_indexed_bounds(draws, runs, entry.run_index);
            entry.aabb_min = aabb_min;
            entry.aabb_max = aabb_max;
            entry.center = (aabb_min + aabb_max) * 0.5;
        }
        self.refit_nodes();
    }

    /// Refits only the entries owning `changed_runs`, then propagates bounds up their ancestors.
    ///
    /// The full [`Self::refit`] rescans every linear and indexed entry and then sweeps every BVH
    /// node. Both the invariant base and the context overlay call it, so a one-renderer patch paid
    /// an O(scene) sweep twice per frame; Darkcity5 measured 2053us per `patch_mesh_renderers` call
    /// against 0.86 dirty renderers. -xlinka
    fn refit_runs(
        &mut self,
        draws: &[FramePreparedDraw],
        runs: &[FramePreparedRun],
        changed_runs: &HashSet<usize>,
    ) {
        for entry in &mut self.linear {
            if changed_runs.contains(&entry.run_index) {
                entry.bounds = refit_linear_bounds(draws, runs, entry);
            }
        }
        // Jump straight to the changed entries and their owning leaves. Scanning all entries and
        // all nodes to find them is what made this O(scene) despite being the incremental path.
        let mut dirty_nodes = HashSet::new();
        for &run_index in changed_runs {
            let Some(&slot) = self.slot_by_run.get(&run_index) else {
                continue;
            };
            let Some(entry) = self.indexed.get_mut(slot) else {
                continue;
            };
            let (aabb_min, aabb_max) = refit_indexed_bounds(draws, runs, entry.run_index);
            entry.aabb_min = aabb_min;
            entry.aabb_max = aabb_max;
            entry.center = (aabb_min + aabb_max) * 0.5;
            if let Some(&node_index) = self.node_of_slot.get(slot)
                && node_index != usize::MAX
            {
                dirty_nodes.insert(node_index);
            }
        }
        if dirty_nodes.is_empty() {
            return;
        }
        let mut cursor = dirty_nodes.into_iter().collect::<Vec<_>>();
        let mut visited = HashSet::new();
        while let Some(node_index) = cursor.pop() {
            if !visited.insert(node_index) {
                continue;
            }
            let node = self.nodes[node_index];
            let (aabb_min, aabb_max, slot_count, run_count) = if node.count > 0 {
                bounds_for_order(
                    &self.indexed,
                    &self.order[node.start..node.start + node.count],
                )
            } else {
                let left = self.nodes[node.left];
                let right = self.nodes[node.right];
                (
                    left.aabb_min.min(right.aabb_min),
                    left.aabb_max.max(right.aabb_max),
                    left.slot_count.saturating_add(right.slot_count),
                    left.run_count.saturating_add(right.run_count),
                )
            };
            self.nodes[node_index].aabb_min = aabb_min;
            self.nodes[node_index].aabb_max = aabb_max;
            self.nodes[node_index].slot_count = slot_count;
            self.nodes[node_index].run_count = run_count;
            if node.parent != usize::MAX {
                visited.remove(&node.parent);
                cursor.push(node.parent);
            }
        }
    }

    fn refit_nodes(&mut self) {
        for node_index in (0..self.nodes.len()).rev() {
            let node = self.nodes[node_index];
            let (aabb_min, aabb_max, slot_count, run_count) = if node.count > 0 {
                bounds_for_order(
                    &self.indexed,
                    &self.order[node.start..node.start + node.count],
                )
            } else {
                let left = self.nodes[node.left];
                let right = self.nodes[node.right];
                (
                    left.aabb_min.min(right.aabb_min),
                    left.aabb_max.max(right.aabb_max),
                    left.slot_count.saturating_add(right.slot_count),
                    left.run_count.saturating_add(right.run_count),
                )
            };
            self.nodes[node_index].aabb_min = aabb_min;
            self.nodes[node_index].aabb_max = aabb_max;
            self.nodes[node_index].slot_count = slot_count;
            self.nodes[node_index].run_count = run_count;
        }
    }

    fn query(
        &self,
        space_id: RenderSpaceId,
        scene: &(impl SceneSpaceRead + ?Sized),
        culling: Option<&WorldMeshCullInput<'_>>,
        out: &mut PreparedSpatialQueryOutput<'_>,
    ) {
        out.visibility.indexed_runs += self.indexed_run_count;
        out.visibility.fallback_runs += self.fallback_run_count;
        out.visibility.linear_fallback_runs += self.linear.len();
        let view = culling.map_or(Mat4::IDENTITY, |culling| {
            culling
                .host_camera
                .explicit_world_to_view()
                .unwrap_or_else(|| {
                    scene.space(space_id).map_or(Mat4::IDENTITY, |space| {
                        crate::camera::view_matrix_for_world_mesh_render_space(scene, space)
                    })
                })
        });
        let cull_context = PreparedSpatialCullContext { culling, view };
        {
            profiling::scope!("mesh::prepared_renderables::spatial_query_linear");
            for entry in &self.linear {
                query_linear_run(cull_context.culling, cull_context.view, entry, out);
            }
        }
        if let Some(root) = self.root {
            profiling::scope!("mesh::prepared_renderables::spatial_query_bvh");
            let node = self.nodes[root];
            if node.count == 0 && self.indexed.len() >= SPATIAL_PARALLEL_QUERY_MIN_RUNS {
                profiling::scope!("mesh::prepared_renderables::spatial_query_bvh_parallel");
                let (left, right) = rayon::join(
                    || {
                        let mut part = PreparedSpatialQueryPart::default();
                        self.query_node(node.left, &cull_context, &mut part.as_output());
                        part
                    },
                    || {
                        let mut part = PreparedSpatialQueryPart::default();
                        self.query_node(node.right, &cull_context, &mut part.as_output());
                        part
                    },
                );
                left.drain_into(out);
                right.drain_into(out);
                return;
            }
            self.query_node(root, &cull_context, out);
        }
    }

    fn query_node(
        &self,
        node_index: usize,
        cull_context: &PreparedSpatialCullContext<'_, '_>,
        out: &mut PreparedSpatialQueryOutput<'_>,
    ) {
        let node = self.nodes[node_index];
        if let Some(culling) = cull_context.culling
            && !spatial_aabb_visible_with_view(
                culling,
                cull_context.view,
                node.aabb_min,
                node.aabb_max,
            )
        {
            record_spatial_frustum_reject(
                out.cull_stats,
                out.visibility,
                node.run_count,
                node.slot_count,
            );
            return;
        }
        if node.count > 0 {
            for &entry_index in &self.order[node.start..node.start + node.count] {
                let entry = self.indexed[entry_index];
                query_indexed_run(cull_context.culling, cull_context.view, entry, out);
            }
        } else {
            self.query_node(node.left, cull_context, out);
            self.query_node(node.right, cull_context, out);
        }
    }

    fn build_node(&mut self, order: &mut [usize], start: usize, end: usize) -> usize {
        let (aabb_min, aabb_max, slot_count, run_count) =
            bounds_for_order(&self.indexed, &order[start..end]);
        let index = self.nodes.len();
        self.nodes.push(PreparedBvhNode {
            aabb_min,
            aabb_max,
            slot_count,
            run_count,
            start,
            count: 0,
            left: 0,
            right: 0,
            parent: usize::MAX,
        });
        let count = end - start;
        if count <= BVH_LEAF_SIZE {
            self.nodes[index].count = count;
            return index;
        }

        let axis = largest_axis(aabb_max - aabb_min);
        order[start..end].sort_unstable_by(|&a, &b| {
            axis_value(self.indexed[a].center, axis)
                .total_cmp(&axis_value(self.indexed[b].center, axis))
                .then_with(|| self.indexed[a].run_index.cmp(&self.indexed[b].run_index))
        });
        let mid = start + count / 2;
        let left = self.build_node(order, start, mid);
        let right = self.build_node(order, mid, end);
        self.nodes[index].left = left;
        self.nodes[index].right = right;
        self.nodes[left].parent = index;
        self.nodes[right].parent = index;
        index
    }

    #[cfg(test)]
    fn uses_bvh(&self) -> bool {
        self.root.is_some()
    }
}

fn refit_linear_bounds(
    draws: &[FramePreparedDraw],
    runs: &[FramePreparedRun],
    entry: &LinearPreparedRun,
) -> Option<(Vec3A, Vec3A)> {
    let run = runs.get(entry.run_index)?;
    let first = draws.get(run.start as usize)?;
    indexable_run_bounds(first)
}

fn refit_indexed_bounds(
    draws: &[FramePreparedDraw],
    runs: &[FramePreparedRun],
    run_index: usize,
) -> (Vec3A, Vec3A) {
    runs.get(run_index)
        .and_then(|run| draws.get(run.start as usize))
        .and_then(indexable_run_bounds)
        .unwrap_or_else(conservative_visible_bounds)
}

fn conservative_visible_bounds() -> (Vec3A, Vec3A) {
    let extent = f32::MAX * 0.25;
    (Vec3A::splat(-extent), Vec3A::splat(extent))
}

/// Shared cull state for one prepared-space spatial query.
struct PreparedSpatialCullContext<'cull_ref, 'cull_data> {
    /// Optional CPU frustum and Hi-Z culling input.
    culling: Option<&'cull_ref WorldMeshCullInput<'cull_data>>,
    /// World-to-view matrix for `space_id`, resolved once per query.
    ///
    /// `world_aabb_visible_for_cull` re-resolved the space and rebuilt this matrix on EVERY node
    /// and EVERY run it tested. Darkcity8 measured 1938us per query against ~37k runs, which is
    /// tens of thousands of hash lookups and matrix builds for a value constant across the whole
    /// traversal. -xlinka
    view: Mat4,
}

/// Mutable query output shared across spatial traversal helpers.
struct PreparedSpatialQueryOutput<'a> {
    /// Unique candidate run marks.
    candidates: &'a mut Vec<usize>,
    /// Raw candidate marks before duplicate suppression.
    raw_candidate_marks: &'a mut usize,
    /// Slot-level cull counters.
    cull_stats: &'a mut (usize, usize, usize),
    /// Visibility broadphase counters.
    visibility: &'a mut WorldMeshVisibilityStats,
}

/// Owned accumulator for one parallel subtree walk.
#[derive(Default)]
struct PreparedSpatialQueryPart {
    candidates: Vec<usize>,
    raw_candidate_marks: usize,
    cull_stats: (usize, usize, usize),
    visibility: WorldMeshVisibilityStats,
}

impl PreparedSpatialQueryPart {
    fn as_output(&mut self) -> PreparedSpatialQueryOutput<'_> {
        PreparedSpatialQueryOutput {
            candidates: &mut self.candidates,
            raw_candidate_marks: &mut self.raw_candidate_marks,
            cull_stats: &mut self.cull_stats,
            visibility: &mut self.visibility,
        }
    }

    fn drain_into(mut self, out: &mut PreparedSpatialQueryOutput<'_>) {
        out.candidates.append(&mut self.candidates);
        *out.raw_candidate_marks = out
            .raw_candidate_marks
            .saturating_add(self.raw_candidate_marks);
        out.cull_stats.0 = out.cull_stats.0.saturating_add(self.cull_stats.0);
        out.cull_stats.1 = out.cull_stats.1.saturating_add(self.cull_stats.1);
        out.cull_stats.2 = out.cull_stats.2.saturating_add(self.cull_stats.2);
        out.visibility.merge(self.visibility);
    }
}

impl PreparedSpatialQueryOutput<'_> {
    /// Marks one visible run candidate.
    fn mark_candidate_run(&mut self, run_index: usize) {
        *self.raw_candidate_marks = self.raw_candidate_marks.saturating_add(1);
        self.candidates.push(run_index);
    }
}

fn gather_spatial_candidate_runs(
    runs: &[FramePreparedRun],
    candidate_indices: &mut Vec<usize>,
) -> Vec<FramePreparedRun> {
    profiling::scope!("mesh::prepared_renderables::spatial_query_gather");
    if should_gather_spatial_candidates_dense(candidate_indices.len(), runs.len()) {
        gather_spatial_candidate_runs_dense(runs, candidate_indices)
    } else {
        gather_spatial_candidate_runs_sparse(runs, candidate_indices)
    }
}

#[inline]
fn should_gather_spatial_candidates_dense(candidate_marks: usize, run_count: usize) -> bool {
    candidate_marks.saturating_mul(SPATIAL_DENSE_GATHER_MIN_CANDIDATE_DIVISOR) >= run_count
}

fn gather_spatial_candidate_runs_sparse(
    runs: &[FramePreparedRun],
    candidate_indices: &mut Vec<usize>,
) -> Vec<FramePreparedRun> {
    profiling::scope!("mesh::prepared_renderables::spatial_query_gather_sparse");
    candidate_indices.sort_unstable();
    candidate_indices.dedup();
    let mut out = Vec::with_capacity(candidate_indices.len());
    for &run_index in candidate_indices.iter() {
        if let Some(run) = runs.get(run_index).copied() {
            out.push(run);
        }
    }
    out
}

fn gather_spatial_candidate_runs_dense(
    runs: &[FramePreparedRun],
    candidate_indices: &[usize],
) -> Vec<FramePreparedRun> {
    profiling::scope!("mesh::prepared_renderables::spatial_query_gather_dense");
    let mut candidate_bits = DenseBitSet::default();
    candidate_bits.clear_and_resize(runs.len());
    for &run_index in candidate_indices {
        candidate_bits.insert(run_index);
    }
    let mut out = Vec::with_capacity(candidate_indices.len().min(runs.len()));
    for (run_index, run) in runs.iter().copied().enumerate() {
        if candidate_bits.contains(run_index) {
            out.push(run);
        }
    }
    out
}

#[derive(Clone, Copy)]
struct LinearPreparedRun {
    run_index: usize,
    bounds: Option<(Vec3A, Vec3A)>,
    slot_count: usize,
}

#[derive(Clone, Copy)]
struct IndexedPreparedRun {
    run_index: usize,
    aabb_min: Vec3A,
    aabb_max: Vec3A,
    center: Vec3A,
    slot_count: usize,
}

#[derive(Clone, Copy)]
struct PreparedBvhNode {
    aabb_min: Vec3A,
    aabb_max: Vec3A,
    slot_count: usize,
    run_count: usize,
    start: usize,
    count: usize,
    left: usize,
    right: usize,
    /// Parent node index, `usize::MAX` at the root. Lets a changed leaf walk up instead of
    /// forcing a full bottom-up sweep of every node.
    parent: usize,
}

fn query_linear_run(
    culling: Option<&WorldMeshCullInput<'_>>,
    view: Mat4,
    entry: &LinearPreparedRun,
    out: &mut PreparedSpatialQueryOutput<'_>,
) {
    if let (Some(culling), Some((aabb_min, aabb_max))) = (culling, entry.bounds)
        && !spatial_aabb_visible_with_view(culling, view, aabb_min, aabb_max)
    {
        record_spatial_frustum_reject(out.cull_stats, out.visibility, 1, entry.slot_count);
        return;
    }
    out.mark_candidate_run(entry.run_index);
}

fn query_indexed_run(
    culling: Option<&WorldMeshCullInput<'_>>,
    view: Mat4,
    entry: IndexedPreparedRun,
    out: &mut PreparedSpatialQueryOutput<'_>,
) {
    if let Some(culling) = culling
        && !spatial_aabb_visible_with_view(culling, view, entry.aabb_min, entry.aabb_max)
    {
        record_spatial_frustum_reject(out.cull_stats, out.visibility, 1, entry.slot_count);
        return;
    }
    out.mark_candidate_run(entry.run_index);
}

fn record_spatial_frustum_reject(
    stats: &mut (usize, usize, usize),
    visibility: &mut WorldMeshVisibilityStats,
    run_count: usize,
    slot_count: usize,
) {
    stats.0 = stats.0.saturating_add(slot_count);
    stats.1 = stats.1.saturating_add(slot_count);
    visibility.broadphase_culled_runs = visibility.broadphase_culled_runs.saturating_add(run_count);
    visibility.broadphase_culled_draws = visibility
        .broadphase_culled_draws
        .saturating_add(slot_count);
}

/// Frustum test against an already-resolved world-to-view matrix.
fn spatial_aabb_visible_with_view(
    culling: &WorldMeshCullInput<'_>,
    view: Mat4,
    aabb_min: Vec3A,
    aabb_max: Vec3A,
) -> bool {
    crate::world_mesh::culling::world_aabb_visible_for_cull_with_view(
        culling,
        view,
        Vec3::from(aabb_min),
        Vec3::from(aabb_max),
    )
}

pub(in crate::world_mesh::draw_prep) fn indexable_run_bounds(
    first: &FramePreparedDraw,
) -> Option<(Vec3A, Vec3A)> {
    if first.is_overlay {
        return None;
    }
    let (aabb_min, aabb_max) = first.cull_geometry?.world_aabb?;
    if !aabb_valid(aabb_min, aabb_max) {
        return None;
    }
    Some((Vec3A::from(aabb_min), Vec3A::from(aabb_max)))
}

fn aabb_valid(aabb_min: Vec3, aabb_max: Vec3) -> bool {
    aabb_min.is_finite() && aabb_max.is_finite() && (aabb_max - aabb_min).cmpgt(Vec3::ZERO).all()
}

fn bounds_for_order(
    entries: &[IndexedPreparedRun],
    order: &[usize],
) -> (Vec3A, Vec3A, usize, usize) {
    let mut aabb_min = Vec3A::splat(f32::INFINITY);
    let mut aabb_max = Vec3A::splat(f32::NEG_INFINITY);
    let mut slot_count = 0usize;
    for &entry_index in order {
        let entry = entries[entry_index];
        aabb_min = aabb_min.min(entry.aabb_min);
        aabb_max = aabb_max.max(entry.aabb_max);
        slot_count = slot_count.saturating_add(entry.slot_count);
    }
    (aabb_min, aabb_max, slot_count, order.len())
}

fn largest_axis(v: Vec3A) -> usize {
    if v.x >= v.y && v.x >= v.z {
        0
    } else if v.y >= v.z {
        1
    } else {
        2
    }
}

fn axis_value(v: Vec3A, axis: usize) -> f32 {
    match axis {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use glam::Mat4;

    use crate::camera::HostCameraFrame;
    use crate::scene::{MeshRendererInstanceId, RenderSpaceId, SceneCoordinator};
    use crate::shared::RenderTransform;
    use crate::world_mesh::culling::{
        MeshCullGeometry, WorldMeshCullInput, WorldMeshCullProjParams,
    };

    fn prepared_draw_with_bounds(
        space_id: RenderSpaceId,
        renderable_index: usize,
        min: Vec3,
        max: Vec3,
    ) -> FramePreparedDraw {
        FramePreparedDraw {
            space_id,
            renderable_index,
            instance_id: MeshRendererInstanceId(renderable_index as u64 + 1),
            renderer_ordinal: renderable_index,
            node_id: renderable_index as i32,
            mesh_asset_id: 10,
            is_overlay: false,
            is_hidden: false,
            sorting_order: 0,
            shadow_cast_mode: ShadowCastMode::On,
            skinned: false,
            world_space_deformed: false,
            blendshape_deformed: false,
            tangent_blendshape_deform_active: false,
            slot_index: 0,
            material_stack_order: None,
            first_index: 0,
            index_count: 3,
            material_asset_id: 1,
            property_block_id: None,
            cull_geometry: Some(MeshCullGeometry {
                world_aabb: Some((min, max)),
                rigid_world_matrix: Some(Mat4::IDENTITY),
                front_face_world_matrix: Some(Mat4::IDENTITY),
            }),
            rigid_world_matrix_override: None,
            particle_draw: ParticleDrawParams::default(),
        }
    }

    fn spatial_scene_and_cull(
        space_id: RenderSpaceId,
    ) -> (SceneCoordinator, HostCameraFrame, WorldMeshCullProjParams) {
        let mut scene = SceneCoordinator::new();
        scene.test_seed_space_identity_worlds(space_id, vec![RenderTransform::default()], vec![-1]);
        (
            scene,
            HostCameraFrame::default(),
            WorldMeshCullProjParams {
                world_proj: Mat4::IDENTITY,
                overlay_proj: Mat4::IDENTITY,
                vr_stereo: None,
            },
        )
    }

    #[test]
    fn spatial_refit_updates_bounds_without_rebuilding_run_membership() {
        let space_id = RenderSpaceId(7);
        let (scene, host_camera, proj) = spatial_scene_and_cull(space_id);
        let culling = WorldMeshCullInput {
            proj,
            host_camera: &host_camera,
            hi_z: None,
            hi_z_temporal: None,
        };
        let mut draws = (0..80)
            .map(|idx| {
                prepared_draw_with_bounds(
                    space_id,
                    idx,
                    Vec3::new(2.0, -0.5, -0.5),
                    Vec3::new(3.0, 0.5, 0.5),
                )
            })
            .collect::<Vec<_>>();
        let runs = (0..draws.len())
            .map(|idx| FramePreparedRun {
                start: idx as u32,
                end: idx as u32 + 1,
            })
            .collect::<Vec<_>>();
        let mut spatial = PreparedSpatialIndex::default();
        spatial.rebuild(&draws, &runs);
        let before = spatial.query_runs(&runs, &[space_id], &scene, Some(&culling));

        draws[0].cull_geometry = Some(MeshCullGeometry {
            world_aabb: Some((Vec3::new(-0.25, -0.25, -0.25), Vec3::new(0.25, 0.25, 0.25))),
            rigid_world_matrix: Some(Mat4::IDENTITY),
            front_face_world_matrix: Some(Mat4::IDENTITY),
        });
        let refit_count = spatial.refit_spaces(&draws, &runs, [space_id]);
        let after = spatial.query_runs(&runs, &[space_id], &scene, Some(&culling));

        assert!(spatial.space_uses_bvh_for_tests(space_id));
        assert_eq!(before.runs.len(), 0);
        assert_eq!(refit_count, 1);
        assert_eq!(after.runs.len(), 1);
        assert_eq!(after.runs[0], FramePreparedRun { start: 0, end: 1 });
    }

    #[test]
    fn an_incremental_refit_matches_the_full_sweep_exactly() {
        // The incremental path touches only the changed entries and their BVH ancestors. If it ever
        // diverges from the full sweep the result is wrong culling, which shows up as missing or
        // stale geometry rather than a crash, so pin the two against each other.
        let space_id = RenderSpaceId(11);
        let (scene, host_camera, proj) = spatial_scene_and_cull(space_id);
        let culling = WorldMeshCullInput {
            proj,
            host_camera: &host_camera,
            hi_z: None,
            hi_z_temporal: None,
        };
        let mut draws = (0..80)
            .map(|idx| {
                prepared_draw_with_bounds(
                    space_id,
                    idx,
                    Vec3::new(2.0, -0.5, -0.5),
                    Vec3::new(3.0, 0.5, 0.5),
                )
            })
            .collect::<Vec<_>>();
        let runs = (0..draws.len())
            .map(|idx| FramePreparedRun {
                start: idx as u32,
                end: idx as u32 + 1,
            })
            .collect::<Vec<_>>();

        let mut full = PreparedSpatialIndex::default();
        full.rebuild(&draws, &runs);
        let mut incremental = PreparedSpatialIndex::default();
        incremental.rebuild(&draws, &runs);

        // Move two runs into view; only those two are named to the incremental path.
        for idx in [0usize, 37] {
            draws[idx].cull_geometry = Some(MeshCullGeometry {
                world_aabb: Some((Vec3::new(-0.25, -0.25, -0.25), Vec3::new(0.25, 0.25, 0.25))),
                rigid_world_matrix: Some(Mat4::IDENTITY),
                front_face_world_matrix: Some(Mat4::IDENTITY),
            });
        }
        let changed = [0usize, 37].into_iter().collect::<HashSet<_>>();

        full.refit_spaces(&draws, &runs, [space_id]);
        incremental.refit_spaces_for_runs(&draws, &runs, [space_id], Some(&changed));

        let expected = full.query_runs(&runs, &[space_id], &scene, Some(&culling));
        let actual = incremental.query_runs(&runs, &[space_id], &scene, Some(&culling));
        assert_eq!(expected.runs, actual.runs);
        assert_eq!(actual.runs.len(), 2);
    }

    #[test]
    fn a_parallel_bvh_walk_returns_the_same_runs_as_a_serial_one() {
        // The split walk is read-only and candidate order does not matter because the gather sorts
        // and dedups, but a merge that drops or double-counts runs would silently change what gets
        // drawn. Build a space above the parallel threshold and compare against the serial result.
        let space_id = RenderSpaceId(77);
        let (scene, host_camera, proj) = spatial_scene_and_cull(space_id);
        let culling = WorldMeshCullInput {
            proj,
            host_camera: &host_camera,
            hi_z: None,
            hi_z_temporal: None,
        };
        let count = SPATIAL_PARALLEL_QUERY_MIN_RUNS + 512;
        let draws = (0..count)
            .map(|idx| {
                // Half in view, half far outside it, so both accept and reject paths are merged.
                let base = if idx % 2 == 0 { -0.25 } else { 500.0 };
                prepared_draw_with_bounds(
                    space_id,
                    idx,
                    Vec3::new(base, base, base),
                    Vec3::new(base + 0.5, base + 0.5, base + 0.5),
                )
            })
            .collect::<Vec<_>>();
        let runs = (0..draws.len())
            .map(|idx| FramePreparedRun {
                start: idx as u32,
                end: idx as u32 + 1,
            })
            .collect::<Vec<_>>();
        let mut spatial = PreparedSpatialIndex::default();
        spatial.rebuild(&draws, &runs);
        assert!(spatial.space_uses_bvh_for_tests(space_id));

        let parallel = spatial.query_runs(&runs, &[space_id], &scene, Some(&culling));
        // Re-run inside a single-threaded pool so the same tree takes the serial path.
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("serial pool");
        let serial =
            pool.install(|| spatial.query_runs(&runs, &[space_id], &scene, Some(&culling)));

        assert_eq!(parallel.runs, serial.runs);
        assert!(!parallel.runs.is_empty(), "half the runs should survive cull");
    }

    #[test]
    fn sparse_candidate_gather_preserves_prepared_order_and_dedups() {
        let runs = [
            FramePreparedRun { start: 0, end: 1 },
            FramePreparedRun { start: 1, end: 2 },
            FramePreparedRun { start: 2, end: 3 },
            FramePreparedRun { start: 3, end: 4 },
        ];
        let mut candidate_indices = vec![2, 0, 2];

        let gathered = gather_spatial_candidate_runs_sparse(&runs, &mut candidate_indices);

        assert_eq!(
            gathered,
            vec![
                FramePreparedRun { start: 0, end: 1 },
                FramePreparedRun { start: 2, end: 3 },
            ]
        );
    }

    #[test]
    fn spatial_candidate_gather_switches_to_dense_for_high_density() {
        assert!(!should_gather_spatial_candidates_dense(1, 3));
        assert!(should_gather_spatial_candidates_dense(2, 3));
    }
}
