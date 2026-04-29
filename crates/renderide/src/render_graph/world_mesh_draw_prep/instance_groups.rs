//! Bevy-style instance grouping for world-mesh forward draws.
//!
//! Produces an [`InstancePlan`] that groups `(batch_key, mesh, submesh)` runs into a
//! contiguous per-draw-slab range regardless of where the sort placed individual members.
//! The forward pass packs the per-draw slab in `slab_layout` order and emits one
//! `draw_indexed(.., 0, instance_range)` per [`DrawGroup`].
//!
//! Replaces the older `(regular_indices, intersect_indices) + for_each_instance_batch`
//! pipeline whose merge requirement was *adjacency in the sorted draw array* — that policy
//! silently fragmented instancing whenever the sort cascade interleaved same-mesh draws
//! with different-mesh draws (e.g. varying `sorting_order` within one material).
//!
//! References: Bevy's `RenderMeshInstances` / `GpuArrayBuffer<MeshUniform>` model
//! (`bevy_pbr/src/render/mesh.rs`, `bevy_render/src/batching/mod.rs::GetBatchData`).

use hashbrown::HashMap;
use std::ops::Range;

use super::WorldMeshDrawItem;

/// One emitted indexed draw covering a contiguous slab range of identical instances.
///
/// All members of a group share `batch_key`, `mesh_asset_id`, `first_index`, and
/// `index_count` by construction (see [`build_instance_plan`]), so the forward pass can
/// drive material binds, vertex streams, and stencil reference from any single member.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DrawGroup {
    /// Index in the sorted `draws` array of the group's first member in sort order.
    ///
    /// Used by the forward pass to advance the `precomputed_batches` cursor and to read
    /// material/state fields that are uniform across the group.
    pub representative_draw_idx: usize,
    /// Slab-coordinate range to pass as `first_instance..first_instance + count` to
    /// `draw_indexed`. Indexes into [`InstancePlan::slab_layout`], not into `draws`.
    pub instance_range: std::ops::Range<u32>,
}

/// Per-view instance plan: slab layout plus groups for regular, intersection, and grab-pass
/// transparent subpasses.
///
/// The forward pass packs the per-draw slab in `slab_layout` order — slot `i` holds the
/// per-draw uniforms for `draws[slab_layout[i]]` — and emits each group's `instance_range`
/// directly. `representative_draw_idx` for both group lists is monotonically increasing so
/// the existing `precomputed_batches` cursor in `draw_subset` advances in O(amortised 1).
#[derive(Clone, Debug, Default)]
pub struct InstancePlan {
    /// New slab order. `slab_layout[i]` is the sorted-draw index whose per-draw uniforms
    /// go into per-draw slot `i`. Length equals `draws.len()` (every draw gets one slot).
    pub slab_layout: Vec<usize>,
    /// Groups emitted by the regular opaque/transparent forward subpass (one
    /// `draw_indexed` each), in ascending `representative_draw_idx` order.
    pub regular_groups: Vec<DrawGroup>,
    /// Groups emitted by the intersection-pass subpass (post depth-snapshot), in
    /// ascending `representative_draw_idx` order.
    pub intersect_groups: Vec<DrawGroup>,
    /// Groups emitted by the grab-pass transparent subpass (post scene-color snapshot), in
    /// ascending `representative_draw_idx` order.
    pub transparent_groups: Vec<DrawGroup>,
}

/// Within-window key for grouping draws that share `batch_key` (already adjacent after sort)
/// by mesh and submesh. Cheap to hash because `batch_key` is implicit (constant within the
/// caller's window).
#[derive(Hash, Eq, PartialEq, Clone, Copy)]
struct MeshSubmeshKey {
    mesh_asset_id: i32,
    first_index: u32,
    index_count: u32,
}

/// Builds the per-view [`InstancePlan`] from a sorted draw list.
///
/// Walks `draws` once. Same-`batch_key` runs are already adjacent because of the sort, so
/// grouping happens in a small per-window `HashMap<MeshSubmeshKey, group_idx>` that is
/// cleared between windows. Singleton-per-draw groups are produced when:
/// - `supports_base_instance` is false (downlevel devices set `instance_count == 1`), or
/// - the run is `skinned` (vertex deform path differs per draw), or
/// - the run is `alpha_blended` (back-to-front order is load-bearing — must not collapse).
///
/// Group emit order matches the order of each group's first member in `draws`, so the
/// view's high-level sort intent (state-change minimisation, transparent depth) is
/// preserved while same-mesh members that landed later still merge in.
pub fn build_instance_plan(
    draws: &[WorldMeshDrawItem],
    supports_base_instance: bool,
) -> InstancePlan {
    profiling::scope!("mesh::build_instance_plan");
    if draws.is_empty() {
        return InstancePlan::default();
    }

    let mut builder = InstancePlanBuilder::with_capacity(draws.len());
    let mut i = 0usize;
    while i < draws.len() {
        let window = next_batch_window(draws, i, supports_base_instance);
        i = window.range.end;
        builder.process_window(draws, window);
    }

    builder.finish()
}

/// Mutable output and scratch buffers used while building one [`InstancePlan`].
struct InstancePlanBuilder {
    /// Per-draw slab order emitted for the frame.
    slab_layout: Vec<usize>,
    /// Regular forward draw groups.
    regular_groups: Vec<DrawGroup>,
    /// Intersection-pass draw groups.
    intersect_groups: Vec<DrawGroup>,
    /// Grab-pass transparent draw groups.
    transparent_groups: Vec<DrawGroup>,
    /// Reusable grouping scratch for one batch-key window.
    scratch: InstancePlanScratch,
}

impl InstancePlanBuilder {
    /// Creates a builder sized for `draw_count` sorted draws.
    fn with_capacity(draw_count: usize) -> Self {
        Self {
            slab_layout: Vec::with_capacity(draw_count),
            regular_groups: Vec::new(),
            intersect_groups: Vec::new(),
            transparent_groups: Vec::new(),
            scratch: InstancePlanScratch::default(),
        }
    }

    /// Emits all groups for one same-batch-key window.
    fn process_window(&mut self, draws: &[WorldMeshDrawItem], window: BatchWindow) {
        if window.singleton {
            self.emit_singletons(window);
        } else {
            self.emit_grouped_window(draws, window);
        }
    }

    /// Emits one GPU draw group per source draw.
    fn emit_singletons(&mut self, window: BatchWindow) {
        let target = subpass_groups(
            &mut self.regular_groups,
            &mut self.intersect_groups,
            &mut self.transparent_groups,
            window.intersect,
            window.grab_pass,
        );
        for draw_idx in window.range {
            emit_group(&mut self.slab_layout, target, draw_idx, &[draw_idx]);
        }
    }

    /// Groups non-transparent same-batch-key draws by mesh/submesh before emission.
    fn emit_grouped_window(&mut self, draws: &[WorldMeshDrawItem], window: BatchWindow) {
        self.scratch.rebuild(draws, window.range.clone());
        let target = subpass_groups(
            &mut self.regular_groups,
            &mut self.intersect_groups,
            &mut self.transparent_groups,
            window.intersect,
            window.grab_pass,
        );
        for group_idx in 0..self.scratch.group_counts.len() {
            let start = self.scratch.group_offsets[group_idx];
            let end = self.scratch.group_offsets[group_idx + 1];
            let members = &self.scratch.group_members[start..end];
            emit_group(
                &mut self.slab_layout,
                target,
                self.scratch.group_representative[group_idx],
                members,
            );
        }
    }

    /// Produces the final plan after debug-validating group order.
    fn finish(self) -> InstancePlan {
        // The cross-window walk visits regular and intersect groups interleaved by sort order,
        // so each list is already in ascending `representative_draw_idx` order — no resort.
        debug_assert!(self
            .regular_groups
            .windows(2)
            .all(|w| w[0].representative_draw_idx <= w[1].representative_draw_idx));
        debug_assert!(self
            .intersect_groups
            .windows(2)
            .all(|w| w[0].representative_draw_idx <= w[1].representative_draw_idx));
        debug_assert!(self
            .transparent_groups
            .windows(2)
            .all(|w| w[0].representative_draw_idx <= w[1].representative_draw_idx));

        InstancePlan {
            slab_layout: self.slab_layout,
            regular_groups: self.regular_groups,
            intersect_groups: self.intersect_groups,
            transparent_groups: self.transparent_groups,
        }
    }
}

/// Reusable temporary storage for grouping one batch-key window.
#[derive(Default)]
struct InstancePlanScratch {
    /// Map from mesh/submesh key to compact group index.
    window_groups: HashMap<MeshSubmeshKey, usize>,
    /// Member count per compact group.
    group_counts: Vec<usize>,
    /// Prefix-sum offsets into [`Self::group_members`].
    group_offsets: Vec<usize>,
    /// Mutable write cursors while filling [`Self::group_members`].
    group_write_offsets: Vec<usize>,
    /// Flat draw-index storage for every group in the current window.
    group_members: Vec<usize>,
    /// Representative sorted draw index per compact group.
    group_representative: Vec<usize>,
}

impl InstancePlanScratch {
    /// Rebuilds all scratch buffers for the supplied window.
    fn rebuild(&mut self, draws: &[WorldMeshDrawItem], range: Range<usize>) {
        self.clear_window();
        self.count_groups(draws, range.clone());
        self.build_offsets();
        self.fill_members(draws, range);
    }

    /// Clears previous-window scratch without releasing capacity.
    fn clear_window(&mut self) {
        self.window_groups.clear();
        self.group_counts.clear();
        self.group_offsets.clear();
        self.group_write_offsets.clear();
        self.group_members.clear();
        self.group_representative.clear();
    }

    /// Counts each mesh/submesh group in first-seen order.
    fn count_groups(&mut self, draws: &[WorldMeshDrawItem], range: Range<usize>) {
        for (offset, item) in draws[range.clone()].iter().enumerate() {
            let draw_idx = range.start + offset;
            let mk = mesh_submesh_key(item);
            if let Some(&group_idx) = self.window_groups.get(&mk) {
                self.group_counts[group_idx] += 1;
            } else {
                let group_idx = self.group_counts.len();
                self.window_groups.insert(mk, group_idx);
                self.group_representative.push(draw_idx);
                self.group_counts.push(1);
            }
        }
    }

    /// Builds prefix offsets and resets write cursors for the current group counts.
    fn build_offsets(&mut self) {
        self.group_offsets.reserve(self.group_counts.len() + 1);
        self.group_offsets.push(0);
        let mut next_offset = 0usize;
        for &count in &self.group_counts {
            next_offset += count;
            self.group_offsets.push(next_offset);
        }
        self.group_members.resize(next_offset, 0);
        self.group_write_offsets
            .extend_from_slice(&self.group_offsets[..self.group_counts.len()]);
    }

    /// Fills the flat member buffer using the offsets computed by [`Self::build_offsets`].
    fn fill_members(&mut self, draws: &[WorldMeshDrawItem], range: Range<usize>) {
        for (offset, item) in draws[range.clone()].iter().enumerate() {
            let Some(&group_idx) = self.window_groups.get(&mesh_submesh_key(item)) else {
                continue;
            };
            let write = self.group_write_offsets[group_idx];
            self.group_members[write] = range.start + offset;
            self.group_write_offsets[group_idx] += 1;
        }
    }
}

/// Same-batch-key draw window and its subpass routing metadata.
#[derive(Clone, Debug)]
struct BatchWindow {
    /// Draw index range covered by this window.
    range: Range<usize>,
    /// Whether the window belongs to the intersection subpass.
    intersect: bool,
    /// Whether the window belongs to the grab-pass transparent subpass.
    grab_pass: bool,
    /// Whether every draw must remain a singleton group.
    singleton: bool,
}

/// Returns the next same-batch-key window starting at `start`.
fn next_batch_window(
    draws: &[WorldMeshDrawItem],
    start: usize,
    supports_base_instance: bool,
) -> BatchWindow {
    let key = &draws[start].batch_key;
    let mut end = start + 1;
    while end < draws.len() && &draws[end].batch_key == key {
        end += 1;
    }

    let intersect = key.embedded_requires_intersection_pass;
    let grab_pass = key.embedded_requires_grab_pass;
    debug_assert!(
        !(intersect && grab_pass),
        "intersection and grab-pass subpasses are mutually exclusive"
    );

    BatchWindow {
        range: start..end,
        intersect,
        grab_pass,
        singleton: !supports_base_instance
            || draws[start].skinned
            || key.alpha_blended
            || grab_pass,
    }
}

/// Builds the grouping key for one draw item.
fn mesh_submesh_key(item: &WorldMeshDrawItem) -> MeshSubmeshKey {
    MeshSubmeshKey {
        mesh_asset_id: item.mesh_asset_id,
        first_index: item.first_index,
        index_count: item.index_count,
    }
}

/// Selects the subpass group vector for a batch window.
fn subpass_groups<'a>(
    regular_groups: &'a mut Vec<DrawGroup>,
    intersect_groups: &'a mut Vec<DrawGroup>,
    transparent_groups: &'a mut Vec<DrawGroup>,
    intersect: bool,
    grab_pass: bool,
) -> &'a mut Vec<DrawGroup> {
    if intersect {
        intersect_groups
    } else if grab_pass {
        transparent_groups
    } else {
        regular_groups
    }
}

/// Appends `members` to `slab_layout` and pushes a [`DrawGroup`] covering the new slab range.
#[inline]
fn emit_group(
    slab_layout: &mut Vec<usize>,
    target: &mut Vec<DrawGroup>,
    representative_draw_idx: usize,
    members: &[usize],
) {
    let first_instance = slab_layout.len() as u32;
    slab_layout.extend_from_slice(members);
    let count = members.len() as u32;
    target.push(DrawGroup {
        representative_draw_idx,
        instance_range: first_instance..first_instance + count,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materials::RasterFrontFace;
    use crate::render_graph::test_fixtures::{dummy_world_mesh_draw_item, DummyDrawItemSpec};
    use crate::render_graph::world_mesh_draw_prep::sort_world_mesh_draws;

    fn opaque(mesh: i32, mat: i32, sort: i32, node: i32) -> WorldMeshDrawItem {
        dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: mat,
            property_block: None,
            skinned: false,
            sorting_order: sort,
            mesh_asset_id: mesh,
            node_id: node,
            slot_index: 0,
            collect_order: node as usize,
            alpha_blended: false,
        })
    }

    #[test]
    fn empty_yields_empty_plan() {
        let plan = build_instance_plan(&[], true);
        assert!(plan.slab_layout.is_empty());
        assert!(plan.regular_groups.is_empty());
        assert!(plan.intersect_groups.is_empty());
        assert!(plan.transparent_groups.is_empty());
    }

    #[test]
    fn identical_opaque_draws_collapse_to_one_group() {
        let mut draws: Vec<_> = (0..6).map(|n| opaque(7, 1, 0, n)).collect();
        sort_world_mesh_draws(&mut draws);

        let plan = build_instance_plan(&draws, true);
        assert_eq!(plan.regular_groups.len(), 1);
        assert_eq!(plan.regular_groups[0].instance_range, 0..6);
        assert_eq!(plan.slab_layout.len(), 6);
        assert!(plan.intersect_groups.is_empty());
        assert!(plan.transparent_groups.is_empty());
    }

    #[test]
    fn mirrored_opaque_draws_split_instance_groups() {
        let normal = opaque(7, 1, 0, 0);
        let mut mirrored = opaque(7, 1, 0, 1);
        mirrored.batch_key.front_face = RasterFrontFace::CounterClockwise;
        let mut draws = vec![normal, mirrored];
        sort_world_mesh_draws(&mut draws);

        let plan = build_instance_plan(&draws, true);
        assert_eq!(plan.regular_groups.len(), 2);
        for group in &plan.regular_groups {
            assert_eq!(group.instance_range.end - group.instance_range.start, 1);
        }
        assert_eq!(plan.slab_layout.len(), 2);
        assert!(plan.intersect_groups.is_empty());
        assert!(plan.transparent_groups.is_empty());
    }

    #[test]
    fn stacked_duplicate_submesh_draws_keep_two_gpu_instances() {
        let mut first = opaque(7, 1, 0, 0);
        first.slot_index = 1;
        first.first_index = 3;
        first.index_count = 6;

        let mut stacked = opaque(7, 1, 0, 1);
        stacked.slot_index = 2;
        stacked.first_index = 3;
        stacked.index_count = 6;

        let mut draws = vec![stacked, first];
        sort_world_mesh_draws(&mut draws);

        let plan = build_instance_plan(&draws, true);
        assert_eq!(plan.regular_groups.len(), 1);
        assert_eq!(plan.regular_groups[0].instance_range, 0..2);
        assert_eq!(plan.slab_layout.len(), 2);
        assert!(plan.intersect_groups.is_empty());
    }

    #[test]
    fn varying_sorting_order_still_collapses_per_mesh() {
        // Same material, two meshes, interleaved sorting_orders. Pre-refactor this
        // fragmented to 5 singleton batches; post-refactor it should be 2 groups.
        let pattern: [(i32, i32); 5] = [(10, 10), (11, 8), (10, 6), (11, 4), (10, 2)];
        let mut draws: Vec<_> = pattern
            .iter()
            .enumerate()
            .map(|(i, &(mesh, sort))| opaque(mesh, 1, sort, i as i32))
            .collect();
        sort_world_mesh_draws(&mut draws);

        let plan = build_instance_plan(&draws, true);
        assert_eq!(plan.regular_groups.len(), 2);
        let total_instances: u32 = plan
            .regular_groups
            .iter()
            .map(|g| g.instance_range.end - g.instance_range.start)
            .sum();
        assert_eq!(total_instances, 5);
        assert_eq!(plan.slab_layout.len(), 5);
        assert!(plan.intersect_groups.is_empty());
        assert!(plan.transparent_groups.is_empty());
    }

    #[test]
    fn skinned_window_emits_singletons() {
        let mut draws: Vec<_> = (0..3)
            .map(|n| {
                dummy_world_mesh_draw_item(DummyDrawItemSpec {
                    material_asset_id: 1,
                    property_block: None,
                    skinned: true,
                    sorting_order: 0,
                    mesh_asset_id: 7,
                    node_id: n,
                    slot_index: 0,
                    collect_order: n as usize,
                    alpha_blended: false,
                })
            })
            .collect();
        sort_world_mesh_draws(&mut draws);

        let plan = build_instance_plan(&draws, true);
        assert_eq!(plan.regular_groups.len(), 3);
        for group in &plan.regular_groups {
            assert_eq!(group.instance_range.end - group.instance_range.start, 1);
        }
    }

    #[test]
    fn alpha_blended_window_emits_singletons() {
        let mut draws: Vec<_> = (0..3)
            .map(|n| {
                dummy_world_mesh_draw_item(DummyDrawItemSpec {
                    material_asset_id: 1,
                    property_block: None,
                    skinned: false,
                    sorting_order: 0,
                    mesh_asset_id: 7,
                    node_id: n,
                    slot_index: 0,
                    collect_order: n as usize,
                    alpha_blended: true,
                })
            })
            .collect();
        sort_world_mesh_draws(&mut draws);

        let plan = build_instance_plan(&draws, true);
        assert_eq!(plan.regular_groups.len(), 3);
    }

    #[test]
    fn grab_pass_window_emits_transparent_singletons() {
        let mut draws: Vec<_> = (0..3)
            .map(|n| {
                let mut item = dummy_world_mesh_draw_item(DummyDrawItemSpec {
                    material_asset_id: 1,
                    property_block: None,
                    skinned: false,
                    sorting_order: 0,
                    mesh_asset_id: 7,
                    node_id: n,
                    slot_index: 0,
                    collect_order: n as usize,
                    alpha_blended: false,
                });
                item.batch_key.embedded_requires_grab_pass = true;
                item
            })
            .collect();
        sort_world_mesh_draws(&mut draws);

        let plan = build_instance_plan(&draws, true);
        assert!(plan.regular_groups.is_empty());
        assert!(plan.intersect_groups.is_empty());
        assert_eq!(plan.transparent_groups.len(), 3);
        for group in &plan.transparent_groups {
            assert_eq!(group.instance_range.end - group.instance_range.start, 1);
        }
    }

    #[test]
    fn intersect_and_grab_pass_batches_stay_separate() {
        let mut intersect = opaque(7, 1, 0, 0);
        intersect.batch_key.embedded_requires_intersection_pass = true;
        let mut grab = opaque(7, 2, 0, 1);
        grab.batch_key.embedded_requires_grab_pass = true;
        let mut draws = vec![intersect, grab];
        sort_world_mesh_draws(&mut draws);

        let plan = build_instance_plan(&draws, true);
        assert!(plan.regular_groups.is_empty());
        assert_eq!(plan.intersect_groups.len(), 1);
        assert_eq!(plan.transparent_groups.len(), 1);
    }

    #[test]
    fn downlevel_disables_grouping() {
        let mut draws: Vec<_> = (0..4).map(|n| opaque(7, 1, 0, n)).collect();
        sort_world_mesh_draws(&mut draws);

        let plan = build_instance_plan(&draws, false);
        assert_eq!(plan.regular_groups.len(), 4);
        for group in &plan.regular_groups {
            assert_eq!(group.instance_range.end - group.instance_range.start, 1);
        }
    }

    #[test]
    fn slab_layout_is_a_permutation_of_draw_indices() {
        let pattern: [(i32, i32); 5] = [(10, 10), (11, 8), (10, 6), (11, 4), (10, 2)];
        let mut draws: Vec<_> = pattern
            .iter()
            .enumerate()
            .map(|(i, &(mesh, sort))| opaque(mesh, 1, sort, i as i32))
            .collect();
        sort_world_mesh_draws(&mut draws);

        let plan = build_instance_plan(&draws, true);
        let mut sorted = plan.slab_layout;
        sorted.sort_unstable();
        assert_eq!(sorted, (0..draws.len()).collect::<Vec<_>>());
    }

    #[test]
    fn group_representatives_are_monotonic() {
        let pattern: [(i32, i32); 5] = [(10, 10), (11, 8), (10, 6), (11, 4), (10, 2)];
        let mut draws: Vec<_> = pattern
            .iter()
            .enumerate()
            .map(|(i, &(mesh, sort))| opaque(mesh, 1, sort, i as i32))
            .collect();
        sort_world_mesh_draws(&mut draws);

        let plan = build_instance_plan(&draws, true);
        for w in plan.regular_groups.windows(2) {
            assert!(w[0].representative_draw_idx < w[1].representative_draw_idx);
        }
    }
}
