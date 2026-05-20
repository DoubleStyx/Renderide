//! Tests for world transform cache propagation.

use super::*;
use glam::{Quat, Vec3};

/// Identity local pose used as the default node TRS in test fixtures.
fn identity_xform() -> RenderTransform {
    RenderTransform {
        position: Vec3::ZERO,
        scale: Vec3::ONE,
        rotation: Quat::IDENTITY,
    }
}

/// Translation-only local pose, convenient for asserting world-matrix products.
fn translation_xform(x: f32, y: f32, z: f32) -> RenderTransform {
    RenderTransform {
        position: Vec3::new(x, y, z),
        scale: Vec3::ONE,
        rotation: Quat::IDENTITY,
    }
}

#[test]
fn bulk_level_parallel_gate_requires_meaningful_width() {
    assert_eq!(
        WORLD_BULK_REBUILD_PARALLEL_LEVEL_MIN,
        WORLD_BULK_REBUILD_PARALLEL_CHUNK_SIZE * 2
    );
    assert!(!should_parallelize_bulk_level(
        WORLD_BULK_REBUILD_PARALLEL_LEVEL_MIN - 1
    ));
    assert!(should_parallelize_bulk_level(
        WORLD_BULK_REBUILD_PARALLEL_LEVEL_MIN
    ));
}

#[test]
fn fixup_transform_id_remaps_last_to_removed() {
    assert_eq!(fixup_transform_id(7, 3, 7), 3);
}

#[test]
fn fixup_transform_id_returns_minus_one_when_old_equals_removed() {
    assert_eq!(fixup_transform_id(3, 3, 7), -1);
}

#[test]
fn fixup_transform_id_passes_through_unrelated_indices() {
    assert_eq!(fixup_transform_id(2, 3, 7), 2);
    assert_eq!(fixup_transform_id(-1, 3, 7), -1);
}

#[test]
fn rebuild_children_builds_parent_to_child_adjacency() {
    let parents = [-1, 0, 0, 1];
    let mut children = Vec::new();
    rebuild_children(&parents, 4, &mut children);
    assert_eq!(children.len(), 4);
    assert_eq!(children[0], vec![1, 2]);
    assert_eq!(children[1], vec![3]);
    assert!(children[2].is_empty());
    assert!(children[3].is_empty());
}

#[test]
fn rebuild_children_ignores_self_loops_and_out_of_bounds_parents() {
    let parents = [1, 1, 5];
    let mut children = Vec::new();
    rebuild_children(&parents, 3, &mut children);
    assert_eq!(children[1], vec![0]);
    assert!(
        children[0].is_empty() && children[2].is_empty(),
        "self-loop on 1 and out-of-bounds parent 5 must be skipped"
    );
}

#[test]
fn rebuild_children_clears_existing_children_before_rebuild() {
    let mut children = vec![vec![99usize]; 2];
    rebuild_children(&[-1, 0], 2, &mut children);
    assert_eq!(children[0], vec![1]);
    assert!(
        children[1].is_empty(),
        "stale child entries must be cleared"
    );
}

#[test]
fn mark_descendants_uncomputed_propagates_through_subtree() {
    let children = vec![vec![1, 2], vec![3], vec![], vec![]];
    let mut computed = vec![false, true, true, true];
    mark_descendants_uncomputed(&children, &mut computed);
    assert_eq!(computed, vec![false, false, false, false]);
}

#[test]
fn mark_descendants_uncomputed_no_op_when_all_computed() {
    let children = vec![vec![1], vec![]];
    let mut computed = vec![true, true];
    mark_descendants_uncomputed(&children, &mut computed);
    assert_eq!(computed, vec![true, true]);
}

#[test]
fn mark_descendants_uncomputed_handles_empty_input() {
    let children: Vec<Vec<usize>> = Vec::new();
    let mut computed: Vec<bool> = Vec::new();
    mark_descendants_uncomputed(&children, &mut computed);
    assert!(computed.is_empty());
}

#[test]
fn ensure_cache_shapes_resizes_and_clears_computed_on_grow() {
    let mut cache = WorldTransformCache::default();
    ensure_cache_shapes(&mut cache, 3, false);
    assert_eq!(cache.world_matrices.len(), 3);
    assert_eq!(cache.degenerate_scales.len(), 3);
    assert_eq!(cache.computed, vec![false, false, false]);
    assert!(
        cache.children_dirty,
        "growth must mark children adjacency dirty"
    );

    for c in &mut cache.computed {
        *c = true;
    }
    ensure_cache_shapes(&mut cache, 5, false);
    assert_eq!(cache.world_matrices.len(), 5);
    assert_eq!(cache.degenerate_scales.len(), 5);
    assert!(
        cache.computed.iter().all(|c| !*c),
        "resize must invalidate all computed flags"
    );
}

#[test]
fn ensure_cache_shapes_force_invalidate_clears_computed_without_resize() {
    let mut cache = WorldTransformCache::default();
    ensure_cache_shapes(&mut cache, 2, false);
    for c in &mut cache.computed {
        *c = true;
    }
    ensure_cache_shapes(&mut cache, 2, true);
    assert!(cache.computed.iter().all(|c| !*c));
}

#[test]
fn compute_world_matrices_for_space_empty_resets_cache() {
    let mut cache = WorldTransformCache::default();
    ensure_cache_shapes(&mut cache, 2, false);
    cache.computed[0] = true;
    compute_world_matrices_for_space(0, &[], &[], &mut cache).expect("ok");
    assert!(cache.world_matrices.is_empty());
    assert!(cache.computed.is_empty());
    assert!(cache.degenerate_scales.is_empty());
}

#[test]
fn compute_world_matrices_for_space_single_root_uses_local_matrix() {
    let nodes = vec![translation_xform(4.0, 0.0, 0.0)];
    let parents = vec![-1];
    let mut cache = WorldTransformCache::default();
    compute_world_matrices_for_space(0, &nodes, &parents, &mut cache).expect("ok");
    assert!(cache.computed[0]);
    let col3 = cache.world_matrices[0].col(3);
    assert!((col3.x - 4.0).abs() < 1e-5);
}

#[test]
fn compute_world_matrices_for_space_two_level_chain_multiplies_in_order() {
    let nodes = vec![
        translation_xform(1.0, 0.0, 0.0),
        translation_xform(2.0, 0.0, 0.0),
    ];
    let parents = vec![-1, 0];
    let mut cache = WorldTransformCache::default();
    compute_world_matrices_for_space(0, &nodes, &parents, &mut cache).expect("ok");
    let child_world = cache.world_matrices[1];
    let expected = render_transform_to_matrix(&nodes[0]) * render_transform_to_matrix(&nodes[1]);
    assert!(child_world.abs_diff_eq(expected, 1e-5));
}

/// Planar zero scale on a parent remains renderable for the whole transform chain.
#[test]
fn compute_world_matrices_for_space_keeps_planar_zero_scale_renderable() {
    let mut collapsed_parent = identity_xform();
    collapsed_parent.scale = Vec3::new(0.0, 1.0, 1.0);
    let nodes = vec![collapsed_parent, identity_xform()];
    let parents = vec![-1, 0];
    let mut cache = WorldTransformCache::default();

    compute_world_matrices_for_space(0, &nodes, &parents, &mut cache).expect("ok");

    assert_eq!(cache.degenerate_scales, vec![false, false]);
}

/// Parent and child planar scales on different axes can still collapse the effective matrix.
#[test]
fn compute_world_matrices_for_space_marks_effective_line_scale_degenerate() {
    let mut collapsed_parent = identity_xform();
    collapsed_parent.scale = Vec3::new(0.0, 1.0, 1.0);
    let mut collapsed_child = identity_xform();
    collapsed_child.scale = Vec3::new(1.0, 0.0, 1.0);
    let nodes = vec![collapsed_parent, collapsed_child];
    let parents = vec![-1, 0];
    let mut cache = WorldTransformCache::default();

    compute_world_matrices_for_space(0, &nodes, &parents, &mut cache).expect("ok");

    assert_eq!(cache.degenerate_scales, vec![false, true]);
}

/// Line-scale collapse on a parent marks every child in that transform chain.
#[test]
fn compute_world_matrices_for_space_propagates_line_scale_to_children() {
    let mut collapsed_parent = identity_xform();
    collapsed_parent.scale = Vec3::new(0.0, 0.0, 1.0);
    let nodes = vec![collapsed_parent, identity_xform()];
    let parents = vec![-1, 0];
    let mut cache = WorldTransformCache::default();

    compute_world_matrices_for_space(0, &nodes, &parents, &mut cache).expect("ok");

    assert_eq!(cache.degenerate_scales, vec![true, true]);
}

/// Negative nonzero object scale keeps the transform renderable for mirrored draw paths.
#[test]
fn compute_world_matrices_for_space_keeps_negative_nonzero_scale_renderable() {
    let mut mirrored = identity_xform();
    mirrored.scale = Vec3::new(-1.0, 1.0, 1.0);
    let nodes = vec![mirrored];
    let parents = vec![-1];
    let mut cache = WorldTransformCache::default();

    compute_world_matrices_for_space(0, &nodes, &parents, &mut cache).expect("ok");

    assert_eq!(cache.degenerate_scales, vec![false]);
}

#[test]
fn compute_world_matrices_for_space_cycle_falls_back_to_local_only() {
    let nodes = vec![identity_xform(), translation_xform(5.0, 0.0, 0.0)];
    let parents = vec![1, 0];
    let mut cache = WorldTransformCache::default();
    compute_world_matrices_for_space(42, &nodes, &parents, &mut cache).expect("cycle path");
    assert!(cache.computed.iter().all(|c| *c));
    let local1 = render_transform_to_matrix(&nodes[1]);
    assert!(
        cache.world_matrices[1].abs_diff_eq(local1, 1e-5),
        "cycle fallback must store local matrix unchanged"
    );
}

#[test]
fn parallel_bulk_rebuild_matches_serial_on_large_chain() {
    // Constructed chain: each node parents the previous one. Above
    // WORLD_BULK_REBUILD_PARALLEL_MIN, the bulk-rebuild path triggers; the result must
    // be bit-identical to the existing serial incremental algorithm.
    let n = WORLD_BULK_REBUILD_PARALLEL_MIN + 7;
    let mut nodes = Vec::with_capacity(n);
    let mut parents = Vec::with_capacity(n);
    for i in 0..n {
        nodes.push(translation_xform(0.5, 0.0, 0.0));
        parents.push(if i == 0 { -1 } else { (i - 1) as i32 });
    }

    let mut parallel = WorldTransformCache::default();
    compute_world_matrices_for_space(0, &nodes, &parents, &mut parallel).expect("parallel bulk");

    // Force the serial path: sub-threshold node count below WORLD_BULK_REBUILD_PARALLEL_MIN
    // still uses incremental, so we run it here and compare deeper subset
    // by directly invoking the incremental method via a fresh cache.
    let mut serial = WorldTransformCache::default();
    ensure_cache_shapes(&mut serial, n, false);
    if serial.children_dirty {
        rebuild_children(&parents, n, &mut serial.children);
        serial.children_dirty = false;
    }
    serial
        .compute_world_matrices_incremental(0, &nodes, &parents)
        .expect("serial");

    for i in 0..n {
        assert!(
            parallel.world_matrices[i].abs_diff_eq(serial.world_matrices[i], 1e-5),
            "world matrix mismatch at index {i}"
        );
        assert_eq!(parallel.degenerate_scales[i], serial.degenerate_scales[i]);
        assert!(parallel.computed[i] && serial.computed[i]);
    }
}

#[test]
fn parallel_bulk_rebuild_handles_wide_tree() {
    // Multiple roots and a wide layer at depth 1: exercises level fan-out.
    let root_count = 16;
    let children_per_root = WORLD_BULK_REBUILD_PARALLEL_MIN / root_count + 8;
    let n = root_count + root_count * children_per_root;
    let mut nodes = Vec::with_capacity(n);
    let mut parents = Vec::with_capacity(n);
    for _ in 0..root_count {
        nodes.push(translation_xform(1.0, 0.0, 0.0));
        parents.push(-1);
    }
    for r in 0..root_count {
        for _ in 0..children_per_root {
            nodes.push(translation_xform(0.0, 1.0, 0.0));
            parents.push(r as i32);
        }
    }

    let mut parallel = WorldTransformCache::default();
    compute_world_matrices_for_space(0, &nodes, &parents, &mut parallel).expect("parallel bulk");

    let mut serial = WorldTransformCache::default();
    ensure_cache_shapes(&mut serial, n, false);
    if serial.children_dirty {
        rebuild_children(&parents, n, &mut serial.children);
        serial.children_dirty = false;
    }
    serial
        .compute_world_matrices_incremental(0, &nodes, &parents)
        .expect("serial");

    for i in 0..n {
        assert!(
            parallel.world_matrices[i].abs_diff_eq(serial.world_matrices[i], 1e-5),
            "world matrix mismatch at index {i}"
        );
    }
}

#[test]
fn compute_world_matrices_for_space_incremental_recomputes_only_dirty() {
    let nodes = vec![
        translation_xform(1.0, 0.0, 0.0),
        translation_xform(2.0, 0.0, 0.0),
    ];
    let parents = vec![-1, 0];
    let mut cache = WorldTransformCache::default();
    compute_world_matrices_for_space(0, &nodes, &parents, &mut cache).expect("first solve");
    let parent_world_before = cache.world_matrices[0];

    cache.computed[1] = false;
    cache.local_dirty[1] = true;
    compute_world_matrices_for_space(0, &nodes, &parents, &mut cache).expect("incremental");

    assert_eq!(
        cache.world_matrices[0], parent_world_before,
        "parent world matrix must not be re-derived when only the child is dirty"
    );
    assert!(cache.computed[1]);
}
