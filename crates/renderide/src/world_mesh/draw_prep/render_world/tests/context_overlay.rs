//! Context-overlay replay invalidation and structural-fallback tests.

use super::*;

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the regression keeps two maintenance phases and their counter assertions together so the coalescing contract is visible end to end"
)]
fn base_and_overlay_prepare_coalesce_overlapping_refit_spaces() {
    let space_id = RenderSpaceId(78);
    let mesh_asset_id = 178;
    let render_context = RenderingContext::Camera;
    let mut first_renderer = StaticMeshRenderer {
        instance_id: MeshRendererInstanceId(1),
        node_id: 0,
        mesh_asset_id,
        material_slots: vec![MeshMaterialSlot {
            material_asset_id: 7,
            property_block_id: None,
        }],
        ..Default::default()
    };
    let second_renderer = StaticMeshRenderer {
        instance_id: MeshRendererInstanceId(2),
        node_id: 1,
        mesh_asset_id,
        material_slots: vec![MeshMaterialSlot {
            material_asset_id: 8,
            property_block_id: None,
        }],
        ..Default::default()
    };
    let mut scene = SceneCoordinator::new();
    scene.test_seed_space_identity_worlds(
        space_id,
        vec![identity_transform(), identity_transform()],
        vec![-1, -1],
    );
    scene.test_set_static_mesh_renderers(
        space_id,
        vec![first_renderer.clone(), second_renderer.clone()],
    );
    let mut mesh_pool = MeshPool::default_pool();
    mesh_pool.insert(crate::assets::mesh::GpuMesh::test_draw_prep_mesh(
        mesh_asset_id,
    ));
    let point_render_buffers = HashMap::new();
    let mut base = RenderWorld::new_context_invariant(render_context);
    let mut overlay = RenderWorld::new_context_overlay(render_context);
    base.prepare_for_frame(
        &scene.context_invariant_read(),
        &mesh_pool,
        &point_render_buffers,
        render_context,
    );
    overlay.prepare_context_overlay_from(
        &base,
        &scene,
        &mesh_pool,
        &point_render_buffers,
        render_context,
    );

    // One stable mesh-row patch and one bounds-only patch both target the same space. The base and
    // overlay each need the final spatial/LOD state once, not once per intermediate patch source.
    let mut moved = identity_transform();
    moved.position = Vec3::new(5.0, 0.0, 0.0);
    scene.test_seed_space_identity_worlds(
        space_id,
        vec![identity_transform(), moved],
        vec![-1, -1],
    );
    first_renderer.sorting_order = 41;
    scene.test_set_static_mesh_renderers(space_id, vec![first_renderer, second_renderer]);
    base.note_renderer_dirty(
        dirty_static(space_id, 0),
        RenderWorldDirtyReason::MaterialOverride,
    );
    base.note_bounds_dirty(
        RenderWorldBoundsDirty {
            space_id,
            kind: RenderWorldRendererKind::Static,
            renderable_index: 1,
        },
        RenderWorldDirtyReason::TransformOnly,
    );
    base.prepare_for_frame(
        &scene.context_invariant_read(),
        &mesh_pool,
        &point_render_buffers,
        render_context,
    );
    let base_stats = base.maintenance_stats();
    assert_eq!(base_stats.mesh_renderer_patch_count, 1);
    assert_eq!(base_stats.bounds_refreshed_renderer_count, 1);
    assert_eq!(
        base_stats.spatial_refit_count, 1,
        "base maintenance must refit the touched space once"
    );

    overlay.prepare_context_overlay_from(
        &base,
        &scene,
        &mesh_pool,
        &point_render_buffers,
        render_context,
    );
    let overlay_stats = overlay.maintenance_stats();
    assert_eq!(overlay_stats.context_overlay_clone_count, 0);
    assert_eq!(overlay_stats.mesh_renderer_patch_count, 1);
    assert_eq!(
        overlay_stats.spatial_refit_count, 1,
        "bounds and mesh replay must share one final overlay refit"
    );
    assert_eq!(overlay.prepared.draws()[0].sorting_order, 41);
}

#[test]
fn mesh_override_only_replay_advances_gpu_static_generation() {
    let space_id = RenderSpaceId(77);
    let render_context = RenderingContext::Camera;
    let (mut scene, mesh_pool, base, mut overlay) = drawing_overlay_pair(space_id, render_context);
    let point_render_buffers = HashMap::new();
    let prepared_generation = overlay.prepared_generation();
    let static_generation = overlay.static_generation();

    scene.test_clear_material_overrides(space_id);
    overlay.note_scene_apply_report(&context_override_report(space_id, render_context));
    overlay.prepare_context_overlay_from(
        &base,
        &scene,
        &mesh_pool,
        &point_render_buffers,
        render_context,
    );

    assert_ne!(
        overlay.prepared_generation(),
        prepared_generation,
        "the changed prepared row must invalidate ordinary draw-plan caches"
    );
    assert_ne!(
        overlay.static_generation(),
        static_generation,
        "a mesh override replay must invalidate GPU-static draw-plan caches too"
    );
}

#[test]
fn a_removed_space_still_forces_a_full_overlay_clone() {
    // Row layout changed, so patching individual renderers cannot repair the overlay.
    let space_id = RenderSpaceId(76);
    let render_context = RenderingContext::Camera;
    let (scene, mesh_pool, base, mut overlay) = synced_overlay_pair(space_id, render_context);
    let point_render_buffers = HashMap::new();

    overlay.note_scene_apply_report(&SceneApplyReport {
        removed_spaces: vec![RenderSpaceId(999)],
        ..Default::default()
    });
    overlay.prepare_context_overlay_from(
        &base,
        &scene,
        &mesh_pool,
        &point_render_buffers,
        render_context,
    );

    let stats = overlay.maintenance_stats();
    assert_eq!(stats.context_overlay_clone_count, 1);
    assert_eq!(
        stats.context_overlay_clone_reason,
        OVERLAY_CLONE_REASON_DIRTY
    );
    assert_eq!(stats.context_overlay_override_replay_count, 0);
}
