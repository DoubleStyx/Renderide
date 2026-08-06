//! Unit tests for retained render-world dirty tracking and snapshot maintenance.

use super::refresh::refresh_render_world_space;
use super::snapshot::{SnapshotRebuildSource, SnapshotRendererTable, build_snapshot_rebuild_tasks};
use super::state::{RenderWorldRendererRef, RenderWorldRendererTemplate};
use super::*;
use crate::cpu_parallelism::{FrameParallelPolicy, ParallelAdmission};
use crate::scene::{
    BillboardRenderBufferEntry, MeshMaterialSlot, MeshRenderBufferEntry, MeshRendererInstanceId,
    SceneCacheFlushReport, SceneCoordinator, SkinnedMeshRenderer, StaticMeshRenderer,
};
use crate::shared::{RenderTransform, ShadowCastMode};
use crate::world_mesh::culling::MeshCullGeometry;
use crate::world_mesh::draw_prep::prepared_renderables::FramePreparedDraw;
use glam::{Mat4, Quat, Vec3};

/// Returns an identity host transform for scene fixtures.
fn identity_transform() -> RenderTransform {
    RenderTransform {
        position: Vec3::ZERO,
        scale: Vec3::ONE,
        rotation: Quat::IDENTITY,
    }
}

/// Builds a dirty renderer key for tests.
fn dirty_static(space_id: RenderSpaceId, renderable_index: usize) -> RenderWorldRendererDirty {
    RenderWorldRendererDirty {
        space_id,
        kind: RenderWorldRendererKind::Static,
        renderable_index,
    }
}

/// Builds a retained static renderer table reference for tests.
fn static_ref(index: usize) -> RenderWorldRendererRef {
    RenderWorldRendererRef {
        kind: RenderWorldRendererKind::Static,
        index,
    }
}

/// Builds a retained draw template entry for snapshot chunking tests.
fn prepared_draw(space_id: RenderSpaceId, renderable_index: usize) -> FramePreparedDraw {
    FramePreparedDraw {
        space_id,
        renderable_index,
        instance_id: Default::default(),
        renderer_ordinal: 0,
        node_id: -1,
        mesh_asset_id: 1,
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
        cull_geometry: None,
        rigid_world_matrix_override: None,
        particle_draw: crate::particles::ParticleDrawParams::default(),
    }
}

/// Marks a prepared test row as generated particle content.
fn particle_prepared_draw(space_id: RenderSpaceId, renderable_index: usize) -> FramePreparedDraw {
    let mut draw = prepared_draw(space_id, renderable_index);
    draw.particle_draw.kind = crate::render_contract::ParticleDrawKind::Billboard;
    draw
}

/// Builds test cull geometry with a recognizable AABB.
fn test_cull_geometry(min_x: f32, max_x: f32) -> MeshCullGeometry {
    MeshCullGeometry {
        world_aabb: Some((Vec3::new(min_x, -0.25, -0.25), Vec3::new(max_x, 0.25, 0.25))),
        rigid_world_matrix: Some(Mat4::IDENTITY),
        front_face_world_matrix: Some(Mat4::IDENTITY),
    }
}

#[test]
fn recreated_render_world_has_distinct_cross_frame_cache_identity() {
    let first = RenderWorld::new(RenderingContext::Camera);
    let replacement = RenderWorld::new(RenderingContext::Camera);

    assert_eq!(
        first.prepared_generation(),
        replacement.prepared_generation()
    );
    assert_ne!(
        first.cache_identity(),
        replacement.cache_identity(),
        "a replacement at the same local generation must invalidate retained draw-item arcs"
    );
}

#[test]
fn apply_report_marks_changed_spaces_dirty_without_fine_report() {
    let mut world = RenderWorld::default();
    world.note_scene_apply_report(&SceneApplyReport {
        frame_index: 7,
        submitted_spaces: vec![RenderSpaceId(1)],
        changed_spaces: vec![RenderSpaceId(1), RenderSpaceId(2)],
        removed_spaces: Vec::new(),
        render_world_dirty: Default::default(),
        ..Default::default()
    });

    assert!(world.dirty_spaces.contains(&RenderSpaceId(1)));
    assert!(world.dirty_spaces.contains(&RenderSpaceId(2)));
}

#[test]
fn classified_noop_report_does_not_fall_back_to_full_space_dirty() {
    let space_id = RenderSpaceId(1);
    let mut world = RenderWorld::default();
    world.note_scene_apply_report(&SceneApplyReport {
        frame_index: 7,
        submitted_spaces: vec![space_id],
        changed_spaces: vec![space_id],
        render_world_classified_spaces: vec![space_id],
        removed_spaces: Vec::new(),
        render_world_dirty: Default::default(),
        ..Default::default()
    });

    assert!(!world.dirty_spaces.contains(&space_id));
    assert!(world.dirty_renderers.is_empty());
}

#[test]
fn apply_report_uses_fine_renderer_dirty_instead_of_changed_space() {
    let mut world = RenderWorld::default();
    let mut report = SceneApplyReport {
        frame_index: 7,
        submitted_spaces: vec![RenderSpaceId(1)],
        changed_spaces: vec![RenderSpaceId(1)],
        removed_spaces: Vec::new(),
        render_world_dirty: Default::default(),
        ..Default::default()
    };
    report
        .render_world_dirty
        .renderers
        .push(dirty_static(RenderSpaceId(1), 3));
    world.note_scene_apply_report(&report);

    assert!(!world.dirty_spaces.contains(&RenderSpaceId(1)));
    assert!(
        world
            .dirty_renderers
            .contains_key(&dirty_static(RenderSpaceId(1), 3))
    );
}

#[test]
fn deformation_candidate_uses_non_topology_dirty_reason() {
    let space_id = RenderSpaceId(2);
    let dirty = RenderWorldRendererDirty {
        space_id,
        kind: RenderWorldRendererKind::Skinned,
        renderable_index: 3,
    };
    let mut report = SceneApplyReport {
        frame_index: 7,
        submitted_spaces: vec![space_id],
        changed_spaces: vec![space_id],
        render_world_classified_spaces: vec![space_id],
        ..Default::default()
    };
    report.render_world_dirty.deform_renderers.push(dirty);
    let mut world = RenderWorld::default();

    world.note_scene_apply_report(&report);

    assert_eq!(
        world.dirty_renderers.get(&dirty),
        Some(&RenderWorldDirtyReason::Deformation)
    );
    let counts = RenderWorldDirtyReasonCounts::from_dirty_sets(
        &world.dirty_renderers,
        &world.dirty_bounds_renderers,
    );
    assert_eq!(counts.deformation, 1);
    assert_eq!(counts.topology, 0);
}

#[test]
fn removed_space_evicts_cached_rows_and_requests_snapshot_rebuild() {
    let mut world = RenderWorld::default();
    world
        .spaces
        .insert(RenderSpaceId(3), RenderWorldSpace::default());
    world.dirty_spaces.insert(RenderSpaceId(3));

    world.note_scene_apply_report(&SceneApplyReport {
        frame_index: 8,
        submitted_spaces: Vec::new(),
        changed_spaces: Vec::new(),
        removed_spaces: vec![RenderSpaceId(3)],
        render_world_dirty: Default::default(),
        ..Default::default()
    });

    assert!(!world.spaces.contains_key(&RenderSpaceId(3)));
    assert!(!world.dirty_spaces.contains(&RenderSpaceId(3)));
    assert!(world.full_rebuild_requested);
}

#[test]
fn cache_flush_no_longer_marks_whole_space_dirty() {
    let world = RenderWorld::default();
    world.note_cache_flush_report(&SceneCacheFlushReport {
        flushed_spaces: vec![RenderSpaceId(9)],
    });

    assert!(!world.dirty_spaces.contains(&RenderSpaceId(9)));
}

#[test]
fn removed_space_wins_over_changed_space_in_apply_report() {
    let mut world = RenderWorld::default();
    world
        .spaces
        .insert(RenderSpaceId(5), RenderWorldSpace::default());

    world.note_scene_apply_report(&SceneApplyReport {
        frame_index: 9,
        submitted_spaces: vec![RenderSpaceId(5)],
        changed_spaces: vec![RenderSpaceId(5)],
        removed_spaces: vec![RenderSpaceId(5)],
        render_world_dirty: Default::default(),
        ..Default::default()
    });

    assert!(!world.spaces.contains_key(&RenderSpaceId(5)));
    assert!(!world.dirty_spaces.contains(&RenderSpaceId(5)));
    assert!(world.full_rebuild_requested);
}

#[test]
fn full_space_dirty_discards_redundant_fine_grained_work() {
    let space_id = RenderSpaceId(6);
    let mut world = RenderWorld::default();
    world.spaces.insert(space_id, RenderWorldSpace::default());
    world.note_renderer_dirty(dirty_static(space_id, 0), RenderWorldDirtyReason::Topology);
    world.note_bounds_dirty(
        RenderWorldBoundsDirty {
            space_id,
            kind: RenderWorldRendererKind::Static,
            renderable_index: 1,
        },
        RenderWorldDirtyReason::TransformOnly,
    );
    world.dirty_transform_roots.push(RenderWorldTransformDirty {
        space_id,
        root_node_ids: vec![0],
    });

    world.note_space_dirty(space_id);

    assert!(world.dirty_spaces.contains(&space_id));
    assert!(world.dirty_renderers.is_empty());
    assert!(world.dirty_bounds_renderers.is_empty());
    assert!(world.dirty_transform_roots.is_empty());
}

/// A world root that ticks every frame reaches this path. Marking the space dirty refreshes every
/// template and rebuilds the prepared snapshot; transforms only move renderers, so the retained
/// rows are patched through the bounds path instead.
#[test]
fn space_wide_transform_change_patches_bounds_instead_of_rebuilding() {
    let space_id = RenderSpaceId(61);
    let mut world = RenderWorld::default();
    let mut space = RenderWorldSpace::default();
    space
        .static_renderers
        .push(RenderWorldRendererTemplate::default());
    space
        .static_renderers
        .push(RenderWorldRendererTemplate::default());
    space
        .skinned_renderers
        .push(RenderWorldRendererTemplate::default());
    world.spaces.insert(space_id, space);

    let marked = world.note_space_transform_bounds_dirty(space_id);

    assert_eq!(marked, 3);
    assert!(
        !world.dirty_spaces.contains(&space_id),
        "a transform-only change must not force a full space rebuild"
    );
    assert!(world.dirty_renderers.is_empty());
    assert_eq!(world.dirty_bounds_renderers.len(), 3);
}

/// Nothing retained means there are no prepared rows to patch, so the full path is still required.
#[test]
fn space_wide_transform_change_falls_back_when_nothing_is_retained() {
    let space_id = RenderSpaceId(62);
    let mut world = RenderWorld::default();

    let marked = world.note_space_transform_bounds_dirty(space_id);

    assert_eq!(marked, 0);
    assert!(world.dirty_spaces.contains(&space_id));
}

#[test]
fn renderer_dirty_supersedes_existing_bounds_dirty() {
    let space_id = RenderSpaceId(7);
    let mut world = RenderWorld::default();
    world.spaces.insert(space_id, RenderWorldSpace::default());
    let bounds_dirty = RenderWorldBoundsDirty {
        space_id,
        kind: RenderWorldRendererKind::Static,
        renderable_index: 0,
    };
    let renderer_dirty = dirty_static(space_id, 0);

    world.note_bounds_dirty(bounds_dirty, RenderWorldDirtyReason::TransformOnly);
    world.note_renderer_dirty(renderer_dirty, RenderWorldDirtyReason::Topology);

    assert!(world.dirty_renderers.contains_key(&renderer_dirty));
    assert!(!world.dirty_bounds_renderers.contains_key(&bounds_dirty));
    let counts = RenderWorldDirtyReasonCounts::from_dirty_sets(
        &world.dirty_renderers,
        &world.dirty_bounds_renderers,
    );
    assert_eq!(counts.topology, 1);
    assert_eq!(counts.transform_only, 0);
}

#[test]
fn mark_all_scene_spaces_dirty_retain_only_existing_scene_spaces() {
    let mut scene = SceneCoordinator::new();
    let keep = RenderSpaceId(10);
    scene.test_seed_space_identity_worlds(keep, vec![identity_transform()], vec![-1]);
    let mut world = RenderWorld::default();
    world.spaces.insert(keep, RenderWorldSpace::default());
    world
        .spaces
        .insert(RenderSpaceId(11), RenderWorldSpace::default());

    world.mark_all_scene_spaces_dirty(&scene);

    assert!(world.spaces.contains_key(&keep));
    assert!(!world.spaces.contains_key(&RenderSpaceId(11)));
    assert!(world.dirty_spaces.contains(&keep));
}

#[test]
fn rebuild_prepared_snapshot_skips_inactive_cached_spaces() {
    let mut scene = SceneCoordinator::new();
    let active = RenderSpaceId(20);
    let inactive = RenderSpaceId(21);
    scene.test_seed_space_identity_worlds(active, vec![identity_transform()], vec![-1]);
    scene.test_seed_space_identity_worlds(inactive, vec![identity_transform()], vec![-1]);
    let mut world = RenderWorld::default();
    world.spaces.insert(
        active,
        RenderWorldSpace {
            active: true,
            ..Default::default()
        },
    );
    world.spaces.insert(
        inactive,
        RenderWorldSpace {
            active: false,
            ..Default::default()
        },
    );

    let mesh_pool = MeshPool::default_pool();
    let point_render_buffers = HashMap::new();
    world.rebuild_prepared_snapshot(
        &scene,
        &mesh_pool,
        &point_render_buffers,
        RenderingContext::RenderToAsset,
        None,
        &HashSet::new(),
    );

    assert_eq!(
        world.prepared.render_context(),
        RenderingContext::RenderToAsset
    );
    assert_eq!(world.prepared.active_space_ids(), &[active]);
    assert!(world.prepared.draws().is_empty());
}

#[test]
fn transform_roots_expand_to_descendant_renderer_records() {
    let mut scene = SceneCoordinator::new();
    let space_id = RenderSpaceId(30);
    scene.test_seed_space_identity_worlds(
        space_id,
        vec![RenderTransform::default(), RenderTransform::default()],
        vec![-1, 0],
    );
    let mut world = RenderWorld::default();
    let mut cached = RenderWorldSpace::default();
    cached.static_renderers.push(RenderWorldRendererTemplate {
        node_id: 1,
        ..Default::default()
    });
    cached.push_reverse_indexes_for_ref(static_ref(0));
    world.spaces.insert(space_id, cached);
    world.dirty_transform_roots.push(RenderWorldTransformDirty {
        space_id,
        root_node_ids: vec![0],
    });

    world.expand_dirty_transform_roots(&scene);

    assert!(world.dirty_renderers.is_empty());
    assert!(
        world
            .dirty_bounds_renderers
            .contains_key(&RenderWorldBoundsDirty {
                space_id,
                kind: RenderWorldRendererKind::Static,
                renderable_index: 0,
            })
    );
}

#[test]
fn transform_root_space_cover_requires_single_scene_root() {
    assert!(transform_roots_cover_space(&[-1, 0, 1], &[0]));
    assert!(!transform_roots_cover_space(&[-1, -1], &[0]));
}

#[test]
fn mesh_asset_dirties_use_reverse_index() {
    let space_id = RenderSpaceId(40);
    let mut world = RenderWorld::default();
    let mut cached = RenderWorldSpace::default();
    cached.static_renderers.push(RenderWorldRendererTemplate {
        mesh_asset_id: 55,
        ..Default::default()
    });
    cached.push_reverse_indexes_for_ref(static_ref(0));
    world.spaces.insert(space_id, cached);
    world.dirty_mesh_assets.insert(55);

    world.expand_dirty_mesh_assets();

    assert!(
        world
            .dirty_renderers
            .contains_key(&dirty_static(space_id, 0))
    );
}

#[test]
fn identical_mesh_metadata_mutation_suppresses_renderer_dirty() {
    let asset_id = 56;
    let mut mesh_pool = MeshPool::default_pool();
    mesh_pool.insert(crate::assets::mesh::GpuMesh::test_draw_prep_mesh(asset_id));
    let mut world = RenderWorld {
        full_rebuild_requested: false,
        mesh_pool_generation: mesh_pool.mutation_generation(),
        ..Default::default()
    };
    world.mesh_draw_prep_states.insert(
        asset_id,
        MeshDrawPrepState::capture(mesh_pool.get(asset_id)),
    );

    mesh_pool.insert(crate::assets::mesh::GpuMesh::test_draw_prep_mesh(asset_id));
    let mut stats = RenderWorldMaintenanceStats::default();
    world.note_mesh_pool_delta(&mesh_pool, &mut stats);

    assert_eq!(stats.mesh_asset_invalidation_count, 1);
    assert_eq!(stats.mesh_asset_draw_prep_noop_count, 1);
    assert_eq!(stats.mesh_asset_draw_prep_change_count, 0);
    assert!(world.dirty_mesh_assets.is_empty());
}

#[test]
fn changed_mesh_draw_prep_metadata_marks_asset_dirty() {
    let asset_id = 57;
    let mut mesh_pool = MeshPool::default_pool();
    mesh_pool.insert(crate::assets::mesh::GpuMesh::test_draw_prep_mesh(asset_id));
    let mut world = RenderWorld {
        full_rebuild_requested: false,
        mesh_pool_generation: mesh_pool.mutation_generation(),
        ..Default::default()
    };
    world.mesh_draw_prep_states.insert(
        asset_id,
        MeshDrawPrepState::capture(mesh_pool.get(asset_id)),
    );

    let mut changed = crate::assets::mesh::GpuMesh::test_draw_prep_mesh(asset_id);
    changed.submeshes[0].1 = 6;
    mesh_pool.insert(changed);
    let mut stats = RenderWorldMaintenanceStats::default();
    world.note_mesh_pool_delta(&mesh_pool, &mut stats);

    assert_eq!(stats.mesh_asset_draw_prep_noop_count, 0);
    assert_eq!(stats.mesh_asset_draw_prep_change_count, 1);
    assert!(world.dirty_mesh_assets.contains(&asset_id));
}

#[test]
fn renderer_state_change_patches_prepared_range_without_snapshot_rebuild() {
    let space_id = RenderSpaceId(59);
    let mesh_asset_id = 159;
    let instance_id = MeshRendererInstanceId(1);
    let render_context = RenderingContext::UserView;
    let renderer = StaticMeshRenderer {
        instance_id,
        node_id: 0,
        mesh_asset_id,
        material_slots: vec![MeshMaterialSlot {
            material_asset_id: 7,
            property_block_id: None,
        }],
        ..Default::default()
    };
    let mut scene = SceneCoordinator::new();
    scene.test_insert_static_mesh_renderers(space_id, vec![renderer.clone()]);
    scene.test_set_space_active(space_id, true);
    let mut mesh_pool = MeshPool::default_pool();
    mesh_pool.insert(crate::assets::mesh::GpuMesh::test_draw_prep_mesh(
        mesh_asset_id,
    ));
    let point_render_buffers = HashMap::new();
    let mut world = RenderWorld::new(render_context);
    world.prepare_for_frame(&scene, &mesh_pool, &point_render_buffers, render_context);
    assert_eq!(world.prepared.draws().len(), 1);
    let old_prepared_generation = world.prepared_generation();
    let old_static_generation = world.static_generation();

    scene.test_set_static_mesh_renderers(
        space_id,
        vec![StaticMeshRenderer {
            sorting_order: 23,
            ..renderer
        }],
    );
    let dirty = dirty_static(space_id, 0);
    let mut report = SceneApplyReport {
        changed_spaces: vec![space_id],
        render_world_classified_spaces: vec![space_id],
        ..Default::default()
    };
    report.render_world_dirty.renderers.push(dirty);
    world.note_scene_apply_report(&report);
    world.prepare_for_frame(&scene, &mesh_pool, &point_render_buffers, render_context);

    let stats = world.maintenance_stats();
    assert_eq!(stats.mesh_renderer_patch_count, 1);
    assert_eq!(stats.snapshot_rebuild_task_count, 0);
    assert_eq!(stats.mesh_patch_structural_rebuild_count, 0);
    assert_eq!(world.prepared.draws()[0].sorting_order, 23);
    assert_ne!(world.prepared_generation(), old_prepared_generation);
    assert_ne!(world.static_generation(), old_static_generation);
}

#[test]
fn unchanged_deformation_candidate_preserves_generations_and_overlay_key() {
    let space_id = RenderSpaceId(60);
    let mesh_asset_id = 160;
    let render_context = RenderingContext::UserView;
    let renderer = SkinnedMeshRenderer {
        base: StaticMeshRenderer {
            instance_id: MeshRendererInstanceId(2),
            node_id: 0,
            mesh_asset_id,
            material_slots: vec![MeshMaterialSlot {
                material_asset_id: 8,
                property_block_id: None,
            }],
            ..Default::default()
        },
        ..Default::default()
    };
    let mut scene = SceneCoordinator::new();
    scene.test_insert_skinned_mesh_renderers(space_id, vec![renderer]);
    scene.test_set_space_active(space_id, true);
    let mut mesh_pool = MeshPool::default_pool();
    mesh_pool.insert(crate::assets::mesh::GpuMesh::test_draw_prep_mesh(
        mesh_asset_id,
    ));
    let point_render_buffers = HashMap::new();
    let mut world = RenderWorld::new(render_context);
    world.prepare_for_frame(&scene, &mesh_pool, &point_render_buffers, render_context);
    assert_eq!(world.prepared.draws().len(), 1);
    let old_prepared_generation = world.prepared_generation();
    let old_static_generation = world.static_generation();
    let dirty = RenderWorldRendererDirty {
        space_id,
        kind: RenderWorldRendererKind::Skinned,
        renderable_index: 0,
    };
    let mut report = SceneApplyReport {
        changed_spaces: vec![space_id],
        render_world_classified_spaces: vec![space_id],
        ..Default::default()
    };
    report.render_world_dirty.deform_renderers.push(dirty);
    world.note_scene_apply_report(&report);

    world.prepare_for_frame(&scene, &mesh_pool, &point_render_buffers, render_context);

    let stats = world.maintenance_stats();
    assert_eq!(stats.deformation_dirty_renderer_count, 1);
    assert_eq!(stats.mesh_renderer_patch_count, 0);
    assert_eq!(stats.mesh_renderer_patch_noop_count, 1);
    assert_eq!(stats.snapshot_rebuild_task_count, 0);
    assert_eq!(stats.steady_state_skip_count, 1);
    assert_eq!(world.prepared_generation(), old_prepared_generation);
    assert_eq!(world.static_generation(), old_static_generation);
}

#[test]
fn mesh_removal_and_reappearance_each_invalidate_draw_prep_state() {
    let asset_id = 58;
    let mut mesh_pool = MeshPool::default_pool();
    mesh_pool.insert(crate::assets::mesh::GpuMesh::test_draw_prep_mesh(asset_id));
    let mut world = RenderWorld {
        full_rebuild_requested: false,
        mesh_pool_generation: mesh_pool.mutation_generation(),
        ..Default::default()
    };
    world.mesh_draw_prep_states.insert(
        asset_id,
        MeshDrawPrepState::capture(mesh_pool.get(asset_id)),
    );

    assert!(mesh_pool.remove(asset_id));
    let mut removed_stats = RenderWorldMaintenanceStats::default();
    world.note_mesh_pool_delta(&mesh_pool, &mut removed_stats);
    assert_eq!(removed_stats.mesh_asset_draw_prep_change_count, 1);
    assert!(world.dirty_mesh_assets.remove(&asset_id));

    mesh_pool.insert(crate::assets::mesh::GpuMesh::test_draw_prep_mesh(asset_id));
    let mut reappeared_stats = RenderWorldMaintenanceStats::default();
    world.note_mesh_pool_delta(&mesh_pool, &mut reappeared_stats);
    assert_eq!(reappeared_stats.mesh_asset_draw_prep_change_count, 1);
    assert_eq!(reappeared_stats.mesh_asset_draw_prep_noop_count, 0);
    assert!(world.dirty_mesh_assets.contains(&asset_id));
}

#[test]
fn full_space_refresh_builds_reverse_indexes_while_refreshing_records() {
    let mut scene = SceneCoordinator::new();
    let space_id = RenderSpaceId(41);
    scene.test_insert_static_mesh_renderers(
        space_id,
        vec![StaticMeshRenderer {
            node_id: 7,
            mesh_asset_id: 88,
            ..Default::default()
        }],
    );
    scene.test_set_space_active(space_id, true);
    let mesh_pool = MeshPool::default_pool();
    let mut cached = RenderWorldSpace::default();

    let outcome = refresh_render_world_space(
        &mut cached,
        &scene,
        &mesh_pool,
        RenderingContext::UserView,
        space_id,
    );

    assert_eq!(outcome.full_space_count, 1);
    assert_eq!(cached.mesh_asset_index.get(&88), Some(&vec![static_ref(0)]));
    assert_eq!(cached.node_index.get(&7), Some(&vec![static_ref(0)]));
}

#[test]
fn generated_particle_mesh_delta_marks_only_snapshot_dirty() {
    let mut mesh_pool = MeshPool::default_pool();
    let mut world = RenderWorld {
        full_rebuild_requested: false,
        mesh_pool_generation: mesh_pool.mutation_generation(),
        ..Default::default()
    };

    let mesh_asset_id = crate::particles::billboard_render_buffer_mesh_asset_id(44)
        .expect("generated billboard mesh id should fit");
    mesh_pool.test_record_mutation(mesh_asset_id);
    let mut stats = RenderWorldMaintenanceStats::default();

    world.note_mesh_pool_delta(&mesh_pool, &mut stats);

    assert_eq!(stats.mesh_asset_invalidation_count, 1);
    assert!(world.particle_snapshot_dirty);
    assert!(!world.full_rebuild_requested);
    assert!(world.dirty_mesh_assets.is_empty());
}

#[test]
fn generated_point_mesh_delta_classifies_mesh_particle_renderer_using_same_buffer() {
    let space_id = RenderSpaceId(66);
    let point_buffer_asset_id = 45;
    let generated_mesh_id =
        crate::particles::billboard_render_buffer_mesh_asset_id(point_buffer_asset_id)
            .expect("generated point-buffer mesh id");
    let mut scene = SceneCoordinator::new();
    scene.test_seed_space_identity_worlds(space_id, vec![identity_transform()], vec![-1]);
    scene.test_push_mesh_render_buffers(
        space_id,
        [MeshRenderBufferEntry {
            node_id: 0,
            point_render_buffer_asset_id: point_buffer_asset_id,
            material_asset_id: 7,
            mesh_asset_id: 8,
            ..Default::default()
        }],
    );
    let mut world = RenderWorld {
        full_rebuild_requested: false,
        particle_snapshot_dirty: true,
        dirty_generated_particle_mesh_assets: HashSet::from([generated_mesh_id]),
        ..Default::default()
    };

    world.classify_generated_particle_mesh_dirties(&scene);

    assert!(world.dirty_generated_particle_mesh_assets.is_empty());
    assert_eq!(
        world.dirty_particle_renderers,
        HashSet::from([RenderWorldParticleRendererDirty {
            space_id,
            kind: RenderWorldParticleRendererKind::Mesh,
            renderable_index: 0,
        }])
    );
    assert!(world.dirty_particle_spaces.is_empty());
}

#[test]
fn source_mesh_delta_classifies_mesh_particle_renderer_without_static_reference() {
    let space_id = RenderSpaceId(67);
    let source_mesh_asset_id = 812;
    let mut scene = SceneCoordinator::new();
    scene.test_seed_space_identity_worlds(space_id, vec![identity_transform()], vec![-1]);
    scene.test_push_mesh_render_buffers(
        space_id,
        [MeshRenderBufferEntry {
            node_id: 0,
            point_render_buffer_asset_id: 45,
            material_asset_id: 7,
            mesh_asset_id: source_mesh_asset_id,
            ..Default::default()
        }],
    );
    let mut world = RenderWorld {
        full_rebuild_requested: false,
        dirty_particle_source_mesh_assets: HashSet::from([source_mesh_asset_id]),
        ..Default::default()
    };

    world.classify_generated_particle_mesh_dirties(&scene);

    assert!(world.particle_snapshot_dirty);
    assert!(world.dirty_particle_source_mesh_assets.is_empty());
    assert_eq!(
        world.dirty_particle_renderers,
        HashSet::from([RenderWorldParticleRendererDirty {
            space_id,
            kind: RenderWorldParticleRendererKind::Mesh,
            renderable_index: 0,
        }])
    );
}

#[test]
fn generated_particle_mesh_delta_patches_only_matching_prepared_renderer_run() {
    let space_id = RenderSpaceId(65);
    let point_buffer_asset_id = 44;
    let generated_mesh_id =
        crate::particles::billboard_render_buffer_mesh_asset_id(point_buffer_asset_id)
            .expect("generated billboard id");
    let mut scene = SceneCoordinator::new();
    scene.test_seed_space_identity_worlds(space_id, vec![identity_transform()], vec![-1]);
    scene.test_push_billboard_render_buffers(
        space_id,
        [BillboardRenderBufferEntry {
            node_id: 0,
            point_render_buffer_asset_id: point_buffer_asset_id,
            material_asset_id: 17,
            ..Default::default()
        }],
    );
    let mut mesh_pool = MeshPool::default_pool();
    mesh_pool.insert(crate::assets::mesh::GpuMesh::test_draw_prep_mesh(
        generated_mesh_id,
    ));
    let point_render_buffers = HashMap::new();
    let render_context = RenderingContext::UserView;
    let mut world = RenderWorld::new(render_context);

    world.prepare_for_frame(&scene, &mesh_pool, &point_render_buffers, render_context);
    let static_generation = world.static_generation();
    let prepared_generation = world.prepared_generation();
    assert_eq!(world.prepared.draws().len(), 1);

    let mut changed_mesh = crate::assets::mesh::GpuMesh::test_draw_prep_mesh(generated_mesh_id);
    changed_mesh.submeshes[0] = (7, 12);
    mesh_pool.insert(changed_mesh);
    world.prepare_for_frame(&scene, &mesh_pool, &point_render_buffers, render_context);

    let stats = world.maintenance_stats();
    assert_eq!(stats.particle_renderer_patch_count, 1);
    assert_eq!(stats.particle_patch_draw_count, 1);
    assert_eq!(stats.particle_snapshot_rebuild_count, 0);
    assert_eq!(stats.snapshot_rebuild_task_count, 0);
    assert_eq!(stats.snapshot_retained_draw_count, 0);
    assert_eq!(world.static_generation(), static_generation);
    assert_ne!(world.prepared_generation(), prepared_generation);
    assert_eq!(world.prepared.draws()[0].first_index, 7);
    assert_eq!(world.prepared.draws()[0].index_count, 12);
}

#[test]
fn particle_membership_refreshes_cached_context_overlay_targets() {
    let space_id = RenderSpaceId(66);
    let point_buffer_asset_id = 45;
    let generated_mesh_id =
        crate::particles::billboard_render_buffer_mesh_asset_id(point_buffer_asset_id)
            .expect("generated billboard id");
    let render_context = RenderingContext::Camera;
    let mut scene = SceneCoordinator::new();
    scene.test_seed_space_identity_worlds(
        space_id,
        vec![identity_transform(), identity_transform()],
        vec![-1, 0],
    );
    scene.test_push_scale_render_transform_override(space_id, 0, render_context, Vec3::splat(2.0));
    let mut mesh_pool = MeshPool::default_pool();
    let mut generated_mesh = crate::assets::mesh::GpuMesh::test_draw_prep_mesh(generated_mesh_id);
    generated_mesh.bounds.extents = Vec3::ONE;
    mesh_pool.insert(generated_mesh);
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
    assert!(overlay.overlay_particle_override_targets.is_empty());
    let static_generation = base.static_generation();

    scene.test_push_billboard_render_buffers(
        space_id,
        [BillboardRenderBufferEntry {
            node_id: 1,
            point_render_buffer_asset_id: point_buffer_asset_id,
            material_asset_id: 17,
            ..Default::default()
        }],
    );
    let mut report = SceneApplyReport::default();
    report.render_world_dirty.particle_spaces.push(space_id);
    base.note_scene_apply_report(&report);
    overlay.note_scene_apply_report(&report);

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

    let dirty = RenderWorldParticleRendererDirty {
        space_id,
        kind: RenderWorldParticleRendererKind::Billboard,
        renderable_index: 0,
    };
    assert_eq!(base.static_generation(), static_generation);
    assert!(overlay.overlay_particle_override_targets.contains(&dirty));
    assert!(!overlay.overlay_particle_targets_dirty);
    let base_matrix = base.prepared.draws()[0]
        .cull_geometry
        .and_then(|geometry| geometry.rigid_world_matrix)
        .expect("base billboard rigid matrix");
    let overlay_matrix = overlay.prepared.draws()[0]
        .cull_geometry
        .and_then(|geometry| geometry.rigid_world_matrix)
        .expect("context billboard rigid matrix");
    assert_eq!(base_matrix, Mat4::IDENTITY);
    assert_eq!(overlay_matrix, Mat4::from_scale(Vec3::splat(2.0)));
    assert_eq!(overlay.maintenance_stats().context_override_patch_count, 1);
}

#[test]
fn empty_particle_invalidation_skips_snapshot_without_static_refresh() {
    let scene = SceneCoordinator::new();
    let mesh_pool = MeshPool::default_pool();
    let point_render_buffers = HashMap::new();
    let mut world = RenderWorld {
        full_rebuild_requested: false,
        particle_snapshot_dirty: true,
        ..Default::default()
    };
    let render_context = world.prepared.render_context();

    world.prepare_for_frame(&scene, &mesh_pool, &point_render_buffers, render_context);

    let stats = world.maintenance_stats();
    assert_eq!(stats.particle_snapshot_rebuild_count, 0);
    assert_eq!(stats.snapshot_rebuild_task_count, 0);
    assert_eq!(stats.full_world_rebuild_count, 0);
    assert_eq!(stats.full_space_rebuild_count, 0);
    assert_eq!(stats.steady_state_skip_count, 1);
    assert!(!world.particle_snapshot_dirty);
}

#[test]
fn parallel_snapshot_assembly_keeps_particle_outputs_for_earlier_spaces() {
    let first_space = RenderSpaceId(61);
    let second_space = RenderSpaceId(62);
    let mut world = RenderWorld::default();
    let render_context = world.prepared.render_context();
    world.prepared.begin_cached_rebuild(render_context);
    let outputs = vec![
        (0usize, vec![prepared_draw(first_space, 0)]),
        (1usize, vec![prepared_draw(second_space, 0)]),
        (0usize, vec![prepared_draw(first_space, 7)]),
    ];

    snapshot::rebuild_snapshot_parallel(
        &mut world,
        &[first_space, second_space],
        &HashSet::new(),
        &HashSet::new(),
        outputs,
    );

    let draws = world.prepared.draws();
    assert_eq!(draws.len(), 3);
    assert!(
        draws
            .iter()
            .any(|draw| draw.space_id == first_space && draw.renderable_index == 7)
    );
    let first_range = draws
        .iter()
        .position(|draw| draw.space_id == second_space)
        .expect("second space draw present");
    assert!(
        draws[..first_range]
            .iter()
            .all(|draw| draw.space_id == first_space),
        "first space draws must stay contiguous"
    );
}

#[test]
fn particle_only_snapshot_reuses_static_and_skinned_rows_then_replaces_particle_rows() {
    let space_id = RenderSpaceId(63);
    let render_context = RenderingContext::UserView;
    let scene = SceneCoordinator::new();
    let mut world = RenderWorld::new(render_context);
    let mut old_static = prepared_draw(space_id, 0);
    old_static.material_asset_id = 11;
    old_static.cull_geometry = Some(test_cull_geometry(-2.0, 2.0));
    let mut old_skinned = prepared_draw(space_id, 1);
    old_skinned.skinned = true;
    old_skinned.material_asset_id = 12;
    let mut old_particle = particle_prepared_draw(space_id, 2);
    old_particle.material_asset_id = 22;

    world.prepared.begin_cached_rebuild(render_context);
    world.prepared.push_cached_space(space_id);
    world
        .prepared
        .extend_cached_draws(&[old_static, old_skinned, old_particle]);
    world.prepared.finish_cached_rebuild(&scene);
    world.prepared.begin_cached_rebuild(render_context);

    let mut fresh_particle = particle_prepared_draw(space_id, 3);
    fresh_particle.material_asset_id = 33;
    snapshot::rebuild_snapshot_parallel(
        &mut world,
        &[space_id],
        &HashSet::new(),
        &HashSet::from([space_id]),
        vec![(0, vec![fresh_particle])],
    );

    let draws = world.prepared.draws();
    assert_eq!(draws.len(), 3);
    assert_eq!(
        draws
            .iter()
            .map(|draw| draw.material_asset_id)
            .collect::<Vec<_>>(),
        vec![11, 12, 33],
        "stable static/skinned rows keep their order and fresh particles remain the suffix"
    );
    assert_eq!(
        draws[0]
            .cull_geometry
            .and_then(|geometry| geometry.world_aabb),
        test_cull_geometry(-2.0, 2.0).world_aabb
    );
    assert!(
        draws.iter().all(|draw| draw.material_asset_id != 22),
        "the previous generated particle row must not survive a particle-only rebuild"
    );
}

#[test]
fn particle_only_serial_rebuild_copies_stable_prefix_and_drops_previous_particle_suffix() {
    let space_id = RenderSpaceId(64);
    let render_context = RenderingContext::UserView;
    let mut scene = SceneCoordinator::new();
    scene.test_seed_space_identity_worlds(space_id, vec![identity_transform()], vec![-1]);
    scene.test_set_space_active(space_id, true);
    let mut world = RenderWorld::new(render_context);
    world.spaces.insert(
        space_id,
        RenderWorldSpace {
            active: true,
            ..Default::default()
        },
    );
    let mut old_static = prepared_draw(space_id, 0);
    old_static.material_asset_id = 41;
    let mut old_particle = particle_prepared_draw(space_id, 1);
    old_particle.material_asset_id = 42;
    world.prepared.begin_cached_rebuild(render_context);
    world.prepared.push_cached_space(space_id);
    world
        .prepared
        .extend_cached_draws(&[old_static, old_particle]);
    world.prepared.finish_cached_rebuild(&scene);

    let dirty_spaces = HashSet::from([space_id]);
    let particle_only_spaces = HashSet::from([space_id]);
    let stats = world.rebuild_prepared_snapshot(
        &scene,
        &MeshPool::default_pool(),
        &HashMap::new(),
        render_context,
        Some(&dirty_spaces),
        &particle_only_spaces,
    );

    assert_eq!(stats.retained_draw_count, 0);
    assert_eq!(stats.reused_space_count, 1);
    assert_eq!(world.prepared.draws().len(), 1);
    assert_eq!(world.prepared.draws()[0].material_asset_id, 41);
    assert_eq!(
        world.prepared.draws()[0].particle_draw.kind,
        crate::render_contract::ParticleDrawKind::None
    );
}

#[test]
fn reverse_index_delta_replaces_stale_renderer_identity() {
    let mut cached = RenderWorldSpace::default();
    cached.static_renderers.push(RenderWorldRendererTemplate {
        mesh_asset_id: 55,
        node_id: 10,
        ..Default::default()
    });
    cached.static_renderers.push(RenderWorldRendererTemplate {
        mesh_asset_id: 55,
        node_id: 11,
        ..Default::default()
    });
    let first = static_ref(0);
    let second = static_ref(1);
    cached.push_reverse_indexes_for_ref(first);
    cached.push_reverse_indexes_for_ref(second);
    cached.remove_reverse_indexes_for_ref(first);
    cached.static_renderers[0].mesh_asset_id = 99;
    cached.static_renderers[0].node_id = 20;
    cached.push_reverse_indexes_for_ref(first);

    assert_eq!(cached.mesh_asset_index.get(&55), Some(&vec![second]));
    assert_eq!(cached.node_index.get(&11), Some(&vec![second]));
    assert_eq!(cached.mesh_asset_index.get(&99), Some(&vec![first]));
    assert_eq!(cached.node_index.get(&20), Some(&vec![first]));
    assert!(!cached.node_index.contains_key(&10));
}

#[test]
fn retained_templates_store_cull_geometry_outside_stable_draws() {
    let space_id = RenderSpaceId(50);
    let cull_geometry = test_cull_geometry(-1.0, 1.0);
    let mut draw = prepared_draw(space_id, 0);
    draw.cull_geometry = Some(cull_geometry);
    let mut record = RenderWorldRendererTemplate {
        draws: vec![draw],
        ..Default::default()
    };

    record.retain_stable_draw_templates_only();

    assert_eq!(
        record
            .cull_geometry
            .and_then(|geometry| geometry.world_aabb),
        cull_geometry.world_aabb
    );
    assert!(record.draws[0].cull_geometry.is_none());
    let mut out = Vec::new();
    let space = RenderWorldSpace {
        active: true,
        static_renderers: vec![record],
        ..Default::default()
    };
    space.append_static_draws_range_to(0..1, &mut out);
    assert_eq!(
        out[0]
            .cull_geometry
            .and_then(|geometry| geometry.world_aabb),
        cull_geometry.world_aabb
    );
}

#[test]
fn prepared_snapshot_reuses_unchanged_space_draw_ranges() {
    let first_space = RenderSpaceId(51);
    let second_space = RenderSpaceId(52);
    let mut scene = SceneCoordinator::new();
    scene.test_seed_space_identity_worlds(first_space, vec![identity_transform()], vec![-1]);
    scene.test_seed_space_identity_worlds(second_space, vec![identity_transform()], vec![-1]);
    let mesh_pool = MeshPool::default_pool();
    let point_render_buffers = HashMap::new();
    let mut world = RenderWorld::default();
    let mut first_draw = prepared_draw(first_space, 0);
    first_draw.material_asset_id = 11;
    let mut second_draw = prepared_draw(second_space, 0);
    second_draw.material_asset_id = 22;
    world.spaces.insert(
        first_space,
        RenderWorldSpace {
            active: true,
            static_renderers: vec![RenderWorldRendererTemplate {
                draws: vec![first_draw],
                ..Default::default()
            }],
            ..Default::default()
        },
    );
    world.spaces.insert(
        second_space,
        RenderWorldSpace {
            active: true,
            static_renderers: vec![RenderWorldRendererTemplate {
                draws: vec![second_draw],
                ..Default::default()
            }],
            ..Default::default()
        },
    );
    world.rebuild_prepared_snapshot(
        &scene,
        &mesh_pool,
        &point_render_buffers,
        RenderingContext::UserView,
        None,
        &HashSet::new(),
    );
    world.spaces.get_mut(&first_space).unwrap().static_renderers[0].draws[0].material_asset_id = 33;
    world
        .spaces
        .get_mut(&second_space)
        .unwrap()
        .static_renderers[0]
        .draws[0]
        .material_asset_id = 44;
    let dirty_spaces = HashSet::from([first_space]);

    world.rebuild_prepared_snapshot(
        &scene,
        &mesh_pool,
        &point_render_buffers,
        RenderingContext::UserView,
        Some(&dirty_spaces),
        &HashSet::new(),
    );

    assert_eq!(
        world.prepared.active_space_ids(),
        &[first_space, second_space]
    );
    assert_eq!(world.prepared.draws()[0].material_asset_id, 33);
    assert_eq!(world.prepared.draws()[1].material_asset_id, 22);
}

#[test]
fn dirty_refresh_parallelism_requires_enough_spaces_and_work_units() {
    let policy = FrameParallelPolicy::new(2);

    assert_eq!(
        dirty_refresh_admission(policy, 2, DIRTY_SPACE_REFRESH_PARALLEL_MIN_WORK_UNITS - 1,),
        ParallelAdmission::Serial
    );
    assert_eq!(
        dirty_refresh_admission(policy, 1, DIRTY_SPACE_REFRESH_PARALLEL_MIN_WORK_UNITS),
        ParallelAdmission::Serial
    );
    assert!(
        dirty_refresh_admission(policy, 2, DIRTY_SPACE_REFRESH_PARALLEL_MIN_WORK_UNITS)
            .is_parallel()
    );
}

#[test]
fn snapshot_rebuild_parallelism_requires_enough_tasks_and_draws() {
    let policy = FrameParallelPolicy::new(2);

    assert_eq!(
        snapshot_rebuild_admission(policy, 2, SNAPSHOT_REBUILD_PARALLEL_MIN_DRAWS - 1),
        ParallelAdmission::Serial
    );
    assert_eq!(
        snapshot_rebuild_admission(policy, 1, SNAPSHOT_REBUILD_PARALLEL_MIN_DRAWS),
        ParallelAdmission::Serial
    );
    assert!(
        snapshot_rebuild_admission(policy, 2, SNAPSHOT_REBUILD_PARALLEL_MIN_DRAWS).is_parallel()
    );
}

#[test]
fn snapshot_rebuild_tasks_chunk_by_retained_template_count() {
    let space_id = RenderSpaceId(60);
    let mut space = RenderWorldSpace::default();
    space.static_renderers.push(RenderWorldRendererTemplate {
        draws: vec![prepared_draw(space_id, 0); 800],
        ..Default::default()
    });
    space.static_renderers.push(RenderWorldRendererTemplate {
        draws: vec![prepared_draw(space_id, 1); 224],
        ..Default::default()
    });
    space.static_renderers.push(RenderWorldRendererTemplate {
        draws: vec![prepared_draw(space_id, 2); 1],
        ..Default::default()
    });

    let tasks = build_snapshot_rebuild_tasks(&[(0, space_id, &space)]);

    assert_eq!(tasks.len(), 5);
    assert_eq!(
        &tasks[0].source,
        &SnapshotRebuildSource::RendererDrawRange {
            table: SnapshotRendererTable::Static,
            renderer_index: 0,
            range: 0..256,
        }
    );
    assert_eq!(
        &tasks[1].source,
        &SnapshotRebuildSource::RendererDrawRange {
            table: SnapshotRendererTable::Static,
            renderer_index: 0,
            range: 256..512,
        }
    );
    assert_eq!(
        &tasks[2].source,
        &SnapshotRebuildSource::RendererDrawRange {
            table: SnapshotRendererTable::Static,
            renderer_index: 0,
            range: 512..768,
        }
    );
    assert_eq!(
        &tasks[3].source,
        &SnapshotRebuildSource::RendererDrawRange {
            table: SnapshotRendererTable::Static,
            renderer_index: 0,
            range: 768..800,
        }
    );
    assert_eq!(
        &tasks[4].source,
        &SnapshotRebuildSource::RendererRange {
            table: SnapshotRendererTable::Static,
            range: 1..3,
        }
    );
    assert_eq!(tasks[0].retained_template_count(), 256);
    assert_eq!(tasks[3].retained_template_count(), 32);
    assert_eq!(tasks[4].retained_template_count(), 225);
}
