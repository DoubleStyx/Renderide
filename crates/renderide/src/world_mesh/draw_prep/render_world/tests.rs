//! Unit tests for retained render-world dirty tracking and snapshot maintenance.

use super::refresh::refresh_render_world_space;
use super::snapshot::{SnapshotRebuildSource, SnapshotRendererTable, build_snapshot_rebuild_tasks};
use super::state::{RenderWorldRendererRef, RenderWorldRendererTemplate};
use super::*;
use crate::cpu_parallelism::{FrameParallelPolicy, ParallelAdmission};
use crate::scene::{SceneCacheFlushReport, StaticMeshRenderer};
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

/// Builds test cull geometry with a recognizable AABB.
fn test_cull_geometry(min_x: f32, max_x: f32) -> MeshCullGeometry {
    MeshCullGeometry {
        world_aabb: Some((Vec3::new(min_x, -0.25, -0.25), Vec3::new(max_x, 0.25, 0.25))),
        rigid_world_matrix: Some(Mat4::IDENTITY),
        front_face_world_matrix: Some(Mat4::IDENTITY),
    }
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
    });

    assert!(world.dirty_spaces.contains(&RenderSpaceId(1)));
    assert!(world.dirty_spaces.contains(&RenderSpaceId(2)));
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
            .contains(&dirty_static(RenderSpaceId(1), 3))
    );
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
    });

    assert!(!world.spaces.contains_key(&RenderSpaceId(5)));
    assert!(!world.dirty_spaces.contains(&RenderSpaceId(5)));
    assert!(world.full_rebuild_requested);
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
            .contains(&RenderWorldBoundsDirty {
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

    assert!(world.dirty_renderers.contains(&dirty_static(space_id, 0)));
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
fn particle_snapshot_dirty_rebuilds_snapshot_without_static_refresh() {
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
    assert_eq!(stats.particle_snapshot_rebuild_count, 1);
    assert_eq!(stats.full_world_rebuild_count, 0);
    assert_eq!(stats.full_space_rebuild_count, 0);
    assert!(!world.particle_snapshot_dirty);
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
