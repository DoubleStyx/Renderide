//! Integration test: GPU instance batching collapses repeated `(mesh, material)` draws into
//! a single `draw_indexed` regardless of where the sort placed them.
//!
//! These tests run through the public renderer surface (`stats_from_sorted`,
//! `sort_draws`) with no GPU and no IPC. The "fragmentation" cases below currently
//! report an under-merged batch count -- they are the regression guard for the Bevy-style
//! `InstancePlan` refactor that decouples per-draw slab layout from sort order.
//!
//! The simple-adjacent case asserts the existing batcher still merges what it always could.

use renderide::materials::ShaderPermutation;
use renderide::render_graph::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};
use renderide::world_mesh::{sort_draws, stats_from_sorted};

/// Baseline: N opaque draws sharing every batch-key field merge into a single instance batch
/// because the sort places them adjacent. Locks in the current batcher behaviour so the
/// Phase 2 refactor can't regress the easy case.
#[test]
fn identical_opaque_draws_collapse_to_one_instance_batch() {
    let mut draws: Vec<_> = (0..8)
        .map(|i| {
            dummy_world_mesh_draw_item(DummyDrawItemSpec {
                material_asset_id: 1,
                property_block: None,
                skinned: false,
                sorting_order: 0,
                mesh_asset_id: 7,
                node_id: i,
                slot_index: 0,
                collect_order: i as usize,
                alpha_blended: false,
            })
        })
        .collect();
    sort_draws(&mut draws);

    let stats = stats_from_sorted(&draws, None, true, ShaderPermutation(0));
    assert_eq!(stats.draws_total, 8);
    assert_eq!(
        stats.instance_batch_total, 1,
        "8 identical draws should merge into 1 batch"
    );
    assert_eq!(
        stats.gpu_instances_emitted, 8,
        "instance count should equal draws"
    );
    assert_eq!(stats.intersect_pass_batches, 0);
}

/// Skinned draws never instance -- each lands in its own singleton batch -- even when the sort
/// would otherwise place them adjacent. Locks in the `can_merge_instances` skinned guard.
#[test]
fn skinned_draws_do_not_merge() {
    let mut draws: Vec<_> = (0..4)
        .map(|i| {
            dummy_world_mesh_draw_item(DummyDrawItemSpec {
                material_asset_id: 1,
                property_block: None,
                skinned: true,
                sorting_order: 0,
                mesh_asset_id: 7,
                node_id: i,
                slot_index: 0,
                collect_order: i as usize,
                alpha_blended: false,
            })
        })
        .collect();
    sort_draws(&mut draws);

    let stats = stats_from_sorted(&draws, None, true, ShaderPermutation(0));
    assert_eq!(stats.draws_total, 4);
    assert_eq!(stats.instance_batch_total, 4);
    assert_eq!(stats.gpu_instances_emitted, 4);
    assert_eq!(stats.skinned_draws, 4);
}

/// Fragmentation case: same opaque material, two distinct meshes (A=10, B=11), but each draw
/// carries a different `sorting_order`. The opaque sort cascade orders within a `batch_key`
/// run by *descending* `sorting_order` first, then `mesh_asset_id` -- so when the host hands
/// the renderer renderers with varying sorting_orders, copies of mesh A get split apart by
/// intervening mesh-B draws and stop being adjacent. The current batcher requires adjacency,
/// so the same-mesh runs fragment into singleton batches.
///
/// Bevy-style grouping yields exactly **2 batches** (one per mesh) regardless of where the
/// sort placed individual members. Locks in the `InstancePlan` behaviour and guards
/// against sort-order regressions that would silently re-fragment instancing.
#[test]
fn varying_sorting_order_does_not_fragment_instancing() {
    // After sort (batch_key equal, then DESC sorting_order): [A(10), B(8), A(6), B(4), A(2)]
    // -- mesh A copies are non-adjacent, so adjacency-based merging falls back to 5 singletons.
    // Phase 2 must collapse to 2 groups (A with three instances, B with two).
    let pattern: [(i32, i32); 5] = [(10, 10), (11, 8), (10, 6), (11, 4), (10, 2)];
    let mut draws: Vec<_> = pattern
        .iter()
        .enumerate()
        .map(|(i, &(mesh, sort))| {
            dummy_world_mesh_draw_item(DummyDrawItemSpec {
                material_asset_id: 1,
                property_block: None,
                skinned: false,
                sorting_order: sort,
                mesh_asset_id: mesh,
                node_id: i as i32,
                slot_index: 0,
                collect_order: i,
                alpha_blended: false,
            })
        })
        .collect();
    sort_draws(&mut draws);

    let stats = stats_from_sorted(&draws, None, true, ShaderPermutation(0));
    assert_eq!(stats.draws_total, 5);
    assert_eq!(
        stats.instance_batch_total, 2,
        "two distinct meshes should yield two batches regardless of sort-order interleaving"
    );
    assert_eq!(stats.gpu_instances_emitted, 5);
}

/// Alpha-blended draws are singletons by design (`can_merge_instances` rejects
/// `alpha_blended`), even after Phase 2 -- back-to-front order matters, instancing must not
/// reorder them. Locks in the singleton-per-draw behaviour for the transparent path.
#[test]
fn alpha_blended_draws_stay_singletons() {
    let mut draws: Vec<_> = (0..3)
        .map(|i| {
            dummy_world_mesh_draw_item(DummyDrawItemSpec {
                material_asset_id: 1,
                property_block: None,
                skinned: false,
                sorting_order: 0,
                mesh_asset_id: 7,
                node_id: i,
                slot_index: 0,
                collect_order: i as usize,
                alpha_blended: true,
            })
        })
        .collect();
    sort_draws(&mut draws);

    let stats = stats_from_sorted(&draws, None, true, ShaderPermutation(0));
    assert_eq!(stats.draws_total, 3);
    assert_eq!(stats.instance_batch_total, 3);
    assert_eq!(stats.gpu_instances_emitted, 3);
}

/// Downlevel adapters (no `BASE_INSTANCE`) force `instance_count == 1` per batch via the
/// `supports_base_instance = false` gate. Asserts the gate still produces one batch per draw
/// even on the easy adjacent case.
#[test]
fn downlevel_disables_instancing() {
    let mut draws: Vec<_> = (0..4)
        .map(|i| {
            dummy_world_mesh_draw_item(DummyDrawItemSpec {
                material_asset_id: 1,
                property_block: None,
                skinned: false,
                sorting_order: 0,
                mesh_asset_id: 7,
                node_id: i,
                slot_index: 0,
                collect_order: i as usize,
                alpha_blended: false,
            })
        })
        .collect();
    sort_draws(&mut draws);

    let stats = stats_from_sorted(&draws, None, false, ShaderPermutation(0));
    assert_eq!(stats.draws_total, 4);
    assert_eq!(stats.instance_batch_total, 4);
    assert_eq!(stats.gpu_instances_emitted, 4);
}
