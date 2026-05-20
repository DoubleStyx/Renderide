//! Unit tests for retained render-world dirty tracking and snapshot maintenance.

use super::state::RenderWorldRendererTemplate;
use super::*;
use crate::scene::SceneCacheFlushReport;
use crate::shared::RenderTransform;

/// Builds a dirty renderer key for tests.
fn dirty_static(space_id: RenderSpaceId, renderable_index: usize) -> RenderWorldRendererDirty {
    RenderWorldRendererDirty {
        space_id,
        kind: RenderWorldRendererKind::Static,
        renderable_index,
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
    scene.test_seed_space_identity_worlds(keep, vec![RenderTransform::default()], vec![-1]);
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
    scene.test_seed_space_identity_worlds(active, vec![RenderTransform::default()], vec![-1]);
    scene.test_seed_space_identity_worlds(inactive, vec![RenderTransform::default()], vec![-1]);
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

    world.rebuild_prepared_snapshot(&scene, RenderingContext::RenderToAsset);

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
    cached.rebuild_reverse_indexes();
    world.spaces.insert(space_id, cached);
    world.dirty_transform_roots.push(RenderWorldTransformDirty {
        space_id,
        root_node_ids: vec![0],
    });

    world.expand_dirty_transform_roots(&scene);

    assert!(world.dirty_renderers.contains(&dirty_static(space_id, 0)));
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
    cached.rebuild_reverse_indexes();
    world.spaces.insert(space_id, cached);
    world.dirty_mesh_assets.insert(55);

    world.expand_dirty_mesh_assets();

    assert!(world.dirty_renderers.contains(&dirty_static(space_id, 0)));
}
