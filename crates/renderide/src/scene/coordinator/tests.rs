//! Unit tests for [`super::SceneCoordinator`].

use glam::{Mat4, Quat, Vec3};

use crate::camera::{view_matrix_for_world_mesh_render_space, view_matrix_from_render_transform};
use crate::scene::render_overrides::RenderTransformOverrideEntry;
use crate::scene::render_space::RenderSpaceState;
use crate::shared::{RenderTransform, RenderingContext};

use super::super::ids::RenderSpaceId;
use super::super::world::{WorldTransformCache, compute_world_matrices_for_space};
use super::SceneCoordinator;

impl SceneCoordinator {
    /// Overrides [`RenderSpaceState::is_active`] for a seeded space (unit tests only).
    pub(crate) fn test_set_space_active(&mut self, id: RenderSpaceId, is_active: bool) {
        if let Some(space) = self.spaces.get_mut(&id) {
            space.is_active = is_active;
        }
    }

    /// Inserts a render space and solves world matrices from the given locals (for unit tests).
    pub(crate) fn test_seed_space_identity_worlds(
        &mut self,
        id: RenderSpaceId,
        nodes: Vec<RenderTransform>,
        node_parents: Vec<i32>,
    ) {
        assert_eq!(
            nodes.len(),
            node_parents.len(),
            "nodes and node_parents length must match"
        );
        self.spaces.insert(
            id,
            RenderSpaceState {
                id,
                is_active: true,
                nodes,
                node_parents,
                ..Default::default()
            },
        );
        let space = self.spaces.get(&id).expect("inserted space");
        let mut cache = WorldTransformCache::default();
        let _ =
            compute_world_matrices_for_space(id.0, &space.nodes, &space.node_parents, &mut cache);
        self.world_caches.insert(id, cache);
    }
}

/// Builds a unit-scale test transform at the origin.
fn identity_transform() -> RenderTransform {
    RenderTransform {
        position: Vec3::ZERO,
        scale: Vec3::ONE,
        rotation: Quat::IDENTITY,
    }
}

/// Render-space iteration is stable so draw collection and transparent fallback ordering do not
/// depend on hash seed or host insertion order.
#[test]
fn render_space_ids_are_sorted_by_host_id() {
    let mut scene = SceneCoordinator::new();
    for id in [RenderSpaceId(42), RenderSpaceId(-2), RenderSpaceId(7)] {
        scene.spaces.insert(
            id,
            RenderSpaceState {
                id,
                is_active: true,
                ..Default::default()
            },
        );
    }

    let ids: Vec<RenderSpaceId> = scene.render_space_ids().collect();
    assert_eq!(
        ids,
        vec![RenderSpaceId(-2), RenderSpaceId(7), RenderSpaceId(42)]
    );

    scene.spaces.remove(&RenderSpaceId(7));
    scene.spaces.insert(
        RenderSpaceId(3),
        RenderSpaceState {
            id: RenderSpaceId(3),
            is_active: true,
            ..Default::default()
        },
    );

    let ids_after_reinsert: Vec<RenderSpaceId> = scene.render_space_ids().collect();
    assert_eq!(
        ids_after_reinsert,
        vec![RenderSpaceId(-2), RenderSpaceId(3), RenderSpaceId(42)]
    );
}

#[test]
fn world_matrix_excludes_render_space_root() {
    let mut scene = SceneCoordinator::new();
    let id = RenderSpaceId(1);
    scene.spaces.insert(
        id,
        RenderSpaceState {
            id,
            is_active: true,
            root_transform: RenderTransform {
                position: Vec3::new(100.0, 0.0, 0.0),
                scale: Vec3::ONE,
                rotation: Quat::IDENTITY,
            },
            nodes: vec![RenderTransform {
                position: Vec3::new(1.0, 2.0, 3.0),
                scale: Vec3::ONE,
                rotation: Quat::IDENTITY,
            }],
            node_parents: vec![-1],
            ..Default::default()
        },
    );
    let space = scene.spaces.get(&id).expect("space");
    let mut cache = WorldTransformCache::default();
    compute_world_matrices_for_space(id.0, &space.nodes, &space.node_parents, &mut cache)
        .expect("solve");
    scene.world_caches.insert(id, cache);

    let world = scene.world_matrix(id, 0).expect("matrix");
    let t = world.col(3);
    assert!(
        (t.x - 1.0).abs() < 1e-4,
        "world_matrix must not include root_transform translation (got x={})",
        t.x
    );

    let with_root = scene
        .world_matrix_including_space_root(id, 0)
        .expect("with root");
    let t2 = with_root.col(3);
    assert!(
        (t2.x - 101.0).abs() < 0.1,
        "world_matrix_including_space_root should add root translation (got x={})",
        t2.x
    );
}

#[test]
fn overlay_render_matrix_tracks_head_output_transform() {
    let mut scene = SceneCoordinator::new();
    let id = RenderSpaceId(7);
    scene.spaces.insert(
        id,
        RenderSpaceState {
            id,
            is_active: true,
            is_overlay: true,
            root_transform: RenderTransform {
                position: Vec3::new(2.0, 3.0, 4.0),
                scale: Vec3::ONE,
                rotation: Quat::IDENTITY,
            },
            nodes: vec![RenderTransform {
                position: Vec3::new(1.0, 0.0, 0.0),
                scale: Vec3::ONE,
                rotation: Quat::IDENTITY,
            }],
            node_parents: vec![-1],
            ..Default::default()
        },
    );
    let space = scene.spaces.get(&id).expect("space");
    let mut cache = WorldTransformCache::default();
    compute_world_matrices_for_space(id.0, &space.nodes, &space.node_parents, &mut cache)
        .expect("solve");
    scene.world_caches.insert(id, cache);

    let head_output =
        Mat4::from_scale_rotation_translation(Vec3::ONE, Quat::IDENTITY, Vec3::new(10.0, 0.0, 0.0));
    let world = scene
        .world_matrix_for_render_context(id, 0, RenderingContext::UserView, head_output)
        .expect("render matrix");
    let t = world.col(3);
    assert!(
        (t.x - 9.0).abs() < 1e-4,
        "overlay x should follow head output"
    );
    assert!(
        (t.y + 3.0).abs() < 1e-4,
        "overlay y should subtract space root"
    );
    assert!(
        (t.z + 4.0).abs() < 1e-4,
        "overlay z should subtract space root"
    );
}

/// Cached zero-scale state reports the selected node as non-renderable for draw collection.
#[test]
fn transform_has_degenerate_scale_reads_cached_world_state() {
    let mut scene = SceneCoordinator::new();
    let id = RenderSpaceId(11);
    let mut collapsed = identity_transform();
    collapsed.scale = Vec3::new(0.0, 1.0, 1.0);
    scene.test_seed_space_identity_worlds(id, vec![collapsed], vec![-1]);

    assert!(scene.transform_has_degenerate_scale(id, 0));
    assert!(scene.transform_has_degenerate_scale_for_context(id, 0, RenderingContext::UserView));
}

/// A zero-scale render-context override hides only the context that owns the override.
#[test]
fn transform_override_zero_scale_is_context_local_degenerate_state() {
    let mut scene = SceneCoordinator::new();
    let id = RenderSpaceId(12);
    scene.test_seed_space_identity_worlds(id, vec![identity_transform()], vec![-1]);
    scene
        .spaces
        .get_mut(&id)
        .expect("space")
        .render_transform_overrides
        .push(RenderTransformOverrideEntry {
            node_id: 0,
            context: RenderingContext::UserView,
            scale_override: Some(Vec3::ZERO),
            ..Default::default()
        });

    assert!(scene.transform_has_degenerate_scale_for_context(id, 0, RenderingContext::UserView));
    assert!(!scene.transform_has_degenerate_scale_for_context(
        id,
        0,
        RenderingContext::ExternalView
    ));
}

/// A context scale override can restore a base zero-scale transform for that context only.
#[test]
fn transform_override_unit_scale_replaces_cached_zero_scale_for_context() {
    let mut scene = SceneCoordinator::new();
    let id = RenderSpaceId(13);
    let mut collapsed = identity_transform();
    collapsed.scale = Vec3::ZERO;
    scene.test_seed_space_identity_worlds(id, vec![collapsed], vec![-1]);
    scene
        .spaces
        .get_mut(&id)
        .expect("space")
        .render_transform_overrides
        .push(RenderTransformOverrideEntry {
            node_id: 0,
            context: RenderingContext::UserView,
            scale_override: Some(Vec3::ONE),
            ..Default::default()
        });

    assert!(!scene.transform_has_degenerate_scale_for_context(id, 0, RenderingContext::UserView));
    assert!(scene.transform_has_degenerate_scale_for_context(
        id,
        0,
        RenderingContext::ExternalView
    ));
}

/// [`super::parallel_apply::apply_extracted_render_space_update`] mutates only the per-space
/// inputs it is given: pre-extracted payloads with non-identity poses commit into the right
/// dense slots and report a dirty world cache so the caller can flag the space for re-flush.
#[test]
fn parallel_apply_extracted_commits_pose_writes_and_marks_dirty() {
    use super::parallel_apply::{
        ExtractedRenderSpaceUpdate, PerSpaceApplyInputs, apply_extracted_render_space_update,
    };
    use crate::scene::transforms_apply::ExtractedTransformsUpdate;
    use crate::shared::{RenderTransform, TransformPoseUpdate};

    let mut space = RenderSpaceState {
        id: RenderSpaceId(7),
        is_active: true,
        nodes: vec![RenderTransform::default(); 3],
        node_parents: vec![-1, 0, 1],
        ..Default::default()
    };
    let mut cache = WorldTransformCache::default();
    compute_world_matrices_for_space(7, &space.nodes, &space.node_parents, &mut cache)
        .expect("solve");

    let new_pose = RenderTransform {
        position: Vec3::new(5.0, 0.0, 0.0),
        scale: Vec3::ONE,
        rotation: Quat::IDENTITY,
    };
    let extracted = ExtractedRenderSpaceUpdate {
        space_id: RenderSpaceId(7),
        cameras: None,
        transforms: Some(ExtractedTransformsUpdate {
            removals: Vec::new(),
            parent_updates: Vec::new(),
            pose_updates: vec![
                TransformPoseUpdate {
                    transform_id: 1,
                    pose: new_pose,
                },
                TransformPoseUpdate {
                    transform_id: -1,
                    pose: RenderTransform::default(),
                },
            ],
            target_transform_count: 3,
            frame_index: 0,
        }),
        meshes: None,
        skinned_meshes: None,
        reflection_probes: None,
        layers: None,
        mesh_render_buffers: None,
        transform_overrides: None,
        material_overrides: None,
    };
    let mut removal_events = Vec::new();
    let dirty = apply_extracted_render_space_update(
        &extracted,
        PerSpaceApplyInputs {
            space: &mut space,
            cache: &mut cache,
            removal_events: &mut removal_events,
        },
    );
    assert!(dirty, "pose write must invalidate the world cache");
    assert!((space.nodes[1].position.x - 5.0).abs() < 1e-5);
    assert!(
        !cache.computed[1],
        "node 1 must be marked uncomputed after pose write"
    );
    assert!(removal_events.is_empty());
}

/// Overlay spaces use the main camera view because object matrices are in main-world coordinates.
#[test]
fn overlay_render_space_view_matrix_matches_main_space() {
    let mut scene = SceneCoordinator::new();
    let main_id = RenderSpaceId(1);
    let overlay_id = RenderSpaceId(0);
    scene.spaces.insert(
        main_id,
        RenderSpaceState {
            id: main_id,
            is_active: true,
            is_overlay: false,
            override_view_position: true,
            root_transform: RenderTransform {
                position: Vec3::new(10.0, 0.0, 0.0),
                scale: Vec3::ONE,
                rotation: Quat::IDENTITY,
            },
            view_transform: RenderTransform {
                position: Vec3::new(10.0, 1.7, 5.0),
                scale: Vec3::ONE,
                rotation: Quat::IDENTITY,
            },
            ..Default::default()
        },
    );
    scene.spaces.insert(
        overlay_id,
        RenderSpaceState {
            id: overlay_id,
            is_active: true,
            is_overlay: true,
            override_view_position: true,
            root_transform: RenderTransform {
                position: Vec3::new(2.0, 0.0, 0.0),
                scale: Vec3::ONE,
                rotation: Quat::IDENTITY,
            },
            view_transform: RenderTransform {
                position: Vec3::new(99.0, 0.0, 0.0),
                scale: Vec3::ONE,
                rotation: Quat::IDENTITY,
            },
            ..Default::default()
        },
    );

    let overlay = scene.space(overlay_id).expect("overlay space");
    let main = scene.active_main_space().expect("main space");
    let v_overlay_rule = view_matrix_for_world_mesh_render_space(&scene, overlay);
    let v_main = view_matrix_from_render_transform(&main.view_transform);
    let diff = (v_overlay_rule - v_main).to_cols_array();
    let err: f32 = diff.iter().map(|&x| x.abs()).sum();
    assert!(
        err < 1e-4,
        "overlay space view matrix must match main space (got err sum {err})"
    );

    let v_from_overlay_only = view_matrix_from_render_transform(&overlay.view_transform);
    let diff_wrong = (v_overlay_rule - v_from_overlay_only).to_cols_array();
    let err_wrong: f32 = diff_wrong.iter().map(|&x| x.abs()).sum();
    assert!(
        err_wrong > 0.1,
        "sanity: overlay-only view must differ from main when positions differ"
    );
}
