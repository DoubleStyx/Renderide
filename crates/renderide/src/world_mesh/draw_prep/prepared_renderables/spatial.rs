//! Per-render-space CPU spatial index for prepared renderer runs.

use glam::{Vec3, Vec3A};
use hashbrown::HashMap;

use crate::scene::{RenderSpaceId, SceneCoordinator};
use crate::world_mesh::culling::{WorldMeshCullInput, world_aabb_visible_for_cull};

use super::super::bitset::DenseBitSet;
use super::super::item::WorldMeshVisibilityStats;
use super::{FramePreparedDraw, FramePreparedRun};

const BVH_LEAF_SIZE: usize = 8;
pub(super) const SPATIAL_LINEAR_RUN_LIMIT: usize = 64;

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
#[derive(Default)]
pub(super) struct PreparedSpatialIndex {
    spaces: HashMap<RenderSpaceId, PreparedSpatialSpace>,
}

impl PreparedSpatialIndex {
    /// Rebuilds all per-space spatial data from the current prepared run snapshot.
    pub(super) fn rebuild(&mut self, draws: &[FramePreparedDraw], runs: &[FramePreparedRun]) {
        profiling::scope!("mesh::prepared_renderables::spatial_rebuild");
        self.spaces.clear();
        let mut builders: HashMap<RenderSpaceId, PreparedSpatialSpaceBuilder> = HashMap::new();
        for (run_index, run) in runs.iter().copied().enumerate() {
            let Some(first) = draws.get(run.start as usize) else {
                continue;
            };
            let slot_count = (run.end - run.start) as usize;
            let builder = builders.entry(first.space_id).or_default();
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
        for (space_id, builder) in builders {
            self.spaces.insert(space_id, builder.finish());
        }
    }

    /// Collects prepared renderer runs for the requested render spaces after optional frustum culling.
    pub(super) fn query_runs(
        &self,
        runs: &[FramePreparedRun],
        space_ids: &[RenderSpaceId],
        scene: &SceneCoordinator,
        culling: Option<&WorldMeshCullInput<'_>>,
    ) -> PreparedSpatialRunCandidates {
        profiling::scope!("mesh::prepared_renderables::spatial_query");
        let mut candidate_bits = DenseBitSet::default();
        candidate_bits.clear_and_resize(runs.len());
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
                        candidates: &mut candidate_bits,
                        raw_candidate_marks: &mut raw_candidate_marks,
                        cull_stats: &mut cull_stats,
                        visibility: &mut visibility,
                    },
                );
            }
        }

        let mut out = Vec::with_capacity(raw_candidate_marks.min(runs.len()));
        {
            profiling::scope!("mesh::prepared_renderables::spatial_query_gather");
            for (run_index, run) in runs.iter().copied().enumerate() {
                if !candidate_bits.contains(run_index) {
                    continue;
                }
                out.push(run);
            }
        }
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
    pub(super) fn refit_spaces<I>(
        &mut self,
        draws: &[FramePreparedDraw],
        runs: &[FramePreparedRun],
        space_ids: I,
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
            space.refit(draws, runs);
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
        let mut order = std::mem::take(&mut space.order);
        let end = order.len();
        space.root = Some(space.build_node(&mut order, 0, end));
        space.order = order;
        space
    }
}

/// One render space's BVH plus conservative linear fallback runs.
#[derive(Default)]
struct PreparedSpatialSpace {
    linear: Vec<LinearPreparedRun>,
    indexed: Vec<IndexedPreparedRun>,
    order: Vec<usize>,
    nodes: Vec<PreparedBvhNode>,
    root: Option<usize>,
    indexed_run_count: usize,
    fallback_run_count: usize,
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
        scene: &SceneCoordinator,
        culling: Option<&WorldMeshCullInput<'_>>,
        out: &mut PreparedSpatialQueryOutput<'_>,
    ) {
        out.visibility.indexed_runs += self.indexed_run_count;
        out.visibility.fallback_runs += self.fallback_run_count;
        out.visibility.linear_fallback_runs += self.linear.len();
        let cull_context = PreparedSpatialCullContext {
            space_id,
            scene,
            culling,
        };
        {
            profiling::scope!("mesh::prepared_renderables::spatial_query_linear");
            for entry in &self.linear {
                query_linear_run(
                    cull_context.space_id,
                    cull_context.scene,
                    cull_context.culling,
                    entry,
                    out,
                );
            }
        }
        if let Some(root) = self.root {
            profiling::scope!("mesh::prepared_renderables::spatial_query_bvh");
            self.query_node(root, &cull_context, out);
        }
    }

    fn query_node(
        &self,
        node_index: usize,
        cull_context: &PreparedSpatialCullContext<'_, '_, '_>,
        out: &mut PreparedSpatialQueryOutput<'_>,
    ) {
        let node = self.nodes[node_index];
        if let Some(culling) = cull_context.culling
            && !spatial_aabb_visible(
                cull_context.scene,
                cull_context.space_id,
                culling,
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
                query_indexed_run(
                    cull_context.space_id,
                    cull_context.scene,
                    cull_context.culling,
                    entry,
                    out,
                );
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
struct PreparedSpatialCullContext<'scene, 'cull_ref, 'cull_data> {
    /// Render space currently being queried.
    space_id: RenderSpaceId,
    /// Scene graph used to resolve world bounds.
    scene: &'scene SceneCoordinator,
    /// Optional CPU frustum and Hi-Z culling input.
    culling: Option<&'cull_ref WorldMeshCullInput<'cull_data>>,
}

/// Mutable query output shared across spatial traversal helpers.
struct PreparedSpatialQueryOutput<'a> {
    /// Unique candidate run marks.
    candidates: &'a mut DenseBitSet,
    /// Raw candidate marks before duplicate suppression.
    raw_candidate_marks: &'a mut usize,
    /// Slot-level cull counters.
    cull_stats: &'a mut (usize, usize, usize),
    /// Visibility broadphase counters.
    visibility: &'a mut WorldMeshVisibilityStats,
}

impl PreparedSpatialQueryOutput<'_> {
    /// Marks one visible run candidate.
    fn mark_candidate_run(&mut self, run_index: usize) {
        *self.raw_candidate_marks = self.raw_candidate_marks.saturating_add(1);
        self.candidates.insert(run_index);
    }
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
}

fn query_linear_run(
    space_id: RenderSpaceId,
    scene: &SceneCoordinator,
    culling: Option<&WorldMeshCullInput<'_>>,
    entry: &LinearPreparedRun,
    out: &mut PreparedSpatialQueryOutput<'_>,
) {
    if let (Some(culling), Some((aabb_min, aabb_max))) = (culling, entry.bounds)
        && !spatial_aabb_visible(scene, space_id, culling, aabb_min, aabb_max)
    {
        record_spatial_frustum_reject(out.cull_stats, out.visibility, 1, entry.slot_count);
        return;
    }
    out.mark_candidate_run(entry.run_index);
}

fn query_indexed_run(
    space_id: RenderSpaceId,
    scene: &SceneCoordinator,
    culling: Option<&WorldMeshCullInput<'_>>,
    entry: IndexedPreparedRun,
    out: &mut PreparedSpatialQueryOutput<'_>,
) {
    if let Some(culling) = culling
        && !spatial_aabb_visible(scene, space_id, culling, entry.aabb_min, entry.aabb_max)
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

fn spatial_aabb_visible(
    scene: &SceneCoordinator,
    space_id: RenderSpaceId,
    culling: &WorldMeshCullInput<'_>,
    aabb_min: Vec3A,
    aabb_max: Vec3A,
) -> bool {
    world_aabb_visible_for_cull(
        scene,
        space_id,
        false,
        culling,
        Vec3::from(aabb_min),
        Vec3::from(aabb_max),
    )
}

fn indexable_run_bounds(first: &FramePreparedDraw) -> Option<(Vec3A, Vec3A)> {
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
            sorting_order: 0,
            skinned: false,
            world_space_deformed: false,
            blendshape_deformed: false,
            tangent_blendshape_deform_active: false,
            slot_index: 0,
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
}
