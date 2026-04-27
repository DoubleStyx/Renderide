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

    let mut slab_layout: Vec<usize> = Vec::with_capacity(draws.len());
    let mut regular_groups: Vec<DrawGroup> = Vec::new();
    let mut intersect_groups: Vec<DrawGroup> = Vec::new();
    let mut transparent_groups: Vec<DrawGroup> = Vec::new();
    let mut window_groups: HashMap<MeshSubmeshKey, usize> = HashMap::new();
    let mut group_members: Vec<Vec<usize>> = Vec::new();
    let mut group_representative: Vec<usize> = Vec::new();

    let mut i = 0usize;
    while i < draws.len() {
        let window_start = i;
        let key = &draws[i].batch_key;
        let mut j = i + 1;
        while j < draws.len() && &draws[j].batch_key == key {
            j += 1;
        }

        let intersect = key.embedded_requires_intersection_pass;
        let grab_pass = key.embedded_requires_grab_pass;
        debug_assert!(
            !(intersect && grab_pass),
            "intersection and grab-pass subpasses are mutually exclusive"
        );
        let window_singleton =
            !supports_base_instance || draws[i].skinned || key.alpha_blended || grab_pass;

        if window_singleton {
            // Every draw in this window becomes its own group, in sort order.
            for draw_idx in window_start..j {
                emit_group(
                    &mut slab_layout,
                    subpass_groups(
                        &mut regular_groups,
                        &mut intersect_groups,
                        &mut transparent_groups,
                        intersect,
                        grab_pass,
                    ),
                    draw_idx,
                    &[draw_idx],
                );
            }
        } else {
            // Group same `(mesh, first_index, index_count)` draws within this window.
            window_groups.clear();
            group_members.clear();
            group_representative.clear();
            for (offset, item) in draws[window_start..j].iter().enumerate() {
                let draw_idx = window_start + offset;
                let mk = MeshSubmeshKey {
                    mesh_asset_id: item.mesh_asset_id,
                    first_index: item.first_index,
                    index_count: item.index_count,
                };
                if let Some(&g) = window_groups.get(&mk) {
                    group_members[g].push(draw_idx);
                } else {
                    let g = group_members.len();
                    window_groups.insert(mk, g);
                    group_representative.push(draw_idx);
                    group_members.push(vec![draw_idx]);
                }
            }
            for (g, members) in group_members.iter().enumerate() {
                emit_group(
                    &mut slab_layout,
                    subpass_groups(
                        &mut regular_groups,
                        &mut intersect_groups,
                        &mut transparent_groups,
                        intersect,
                        grab_pass,
                    ),
                    group_representative[g],
                    members,
                );
            }
        }

        i = j;
    }

    // The cross-window walk visits regular and intersect groups interleaved by sort order,
    // so each list is already in ascending `representative_draw_idx` order — no resort.
    debug_assert!(regular_groups
        .windows(2)
        .all(|w| w[0].representative_draw_idx <= w[1].representative_draw_idx));
    debug_assert!(intersect_groups
        .windows(2)
        .all(|w| w[0].representative_draw_idx <= w[1].representative_draw_idx));
    debug_assert!(transparent_groups
        .windows(2)
        .all(|w| w[0].representative_draw_idx <= w[1].representative_draw_idx));

    InstancePlan {
        slab_layout,
        regular_groups,
        intersect_groups,
        transparent_groups,
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
