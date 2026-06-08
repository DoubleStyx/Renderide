//! Phase-orchestration tests: render-world header / extracted update dirtiness plus the
//! per-space apply commit that the parallel apply path drives.

use glam::{Quat, Vec3};

use crate::scene::render_space::RenderSpaceState;
use crate::shared::{RenderSpaceUpdate, RenderTransform};

use super::super::super::ids::RenderSpaceId;
use super::super::super::world::{WorldTransformCache, compute_world_matrices_for_space};
use super::super::apply::ExtractedRenderSpaceUpdate;
use super::super::{
    RenderWorldRendererKind, SceneApplyReport, extracted_update_affects_render_world,
    note_render_world_dirty_for_extracted_update, render_world_header_changed,
};

fn empty_extracted_render_space_update() -> ExtractedRenderSpaceUpdate {
    ExtractedRenderSpaceUpdate {
        space_id: RenderSpaceId(1),
        cameras: None,
        camera_portals: None,
        reflection_probes: None,
        transforms: None,
        meshes: None,
        skinned_meshes: None,
        layers: None,
        lod_groups: None,
        transform_overrides: None,
        material_overrides: None,
        blit_to_displays: None,
        billboard_render_buffers: None,
        mesh_render_buffers: None,
        trail_render_buffers: None,
    }
}

#[test]
fn render_world_header_dirty_ignores_view_only_header_changes() {
    let space = RenderSpaceState {
        is_active: true,
        is_overlay: false,
        view_position_is_external: false,
        root_transform: RenderTransform {
            position: Vec3::new(1.0, 2.0, 3.0),
            scale: Vec3::ONE,
            rotation: Quat::IDENTITY,
        },
        ..Default::default()
    };
    let update = RenderSpaceUpdate {
        is_active: true,
        is_overlay: false,
        view_position_is_external: false,
        root_transform: RenderTransform {
            position: Vec3::new(9.0, 8.0, 7.0),
            scale: Vec3::ONE,
            rotation: Quat::IDENTITY,
        },
        ..RenderSpaceUpdate::default()
    };

    assert!(!render_world_header_changed(Some(&space), &update));
}

#[test]
fn render_world_header_dirty_tracks_draw_prep_header_changes() {
    let space = RenderSpaceState {
        is_active: true,
        is_overlay: false,
        view_position_is_external: false,
        ..Default::default()
    };

    assert!(render_world_header_changed(
        Some(&space),
        &RenderSpaceUpdate {
            is_active: false,
            is_overlay: false,
            view_position_is_external: false,
            ..RenderSpaceUpdate::default()
        },
    ));
    assert!(render_world_header_changed(
        Some(&space),
        &RenderSpaceUpdate {
            is_active: true,
            is_overlay: true,
            view_position_is_external: false,
            ..RenderSpaceUpdate::default()
        },
    ));
    assert!(render_world_header_changed(
        Some(&space),
        &RenderSpaceUpdate {
            is_active: true,
            is_overlay: false,
            view_position_is_external: true,
            ..RenderSpaceUpdate::default()
        },
    ));
}

#[test]
fn extracted_render_world_dirty_ignores_camera_only_updates() {
    let mut update = empty_extracted_render_space_update();
    update.cameras = Some(crate::scene::camera::ExtractedCameraRenderablesUpdate::default());

    assert!(!extracted_update_affects_render_world(&update));
}

#[test]
fn extracted_render_world_dirty_tracks_transform_updates() {
    let mut update = empty_extracted_render_space_update();
    update.transforms = Some(crate::scene::transforms::ExtractedTransformsUpdate::default());

    assert!(extracted_update_affects_render_world(&update));
}

#[test]
fn extracted_render_world_dirty_tracks_lod_group_updates() {
    let mut update = empty_extracted_render_space_update();
    update.lod_groups =
        Some(crate::scene::lod_groups::ExtractedLodGroupRenderablesUpdate::default());

    assert!(extracted_update_affects_render_world(&update));
}

#[test]
fn render_world_dirty_report_tracks_static_state_rows() {
    let mut update = empty_extracted_render_space_update();
    update.meshes = Some(crate::scene::meshes::ExtractedMeshRenderablesUpdate {
        mesh_states: vec![
            crate::shared::MeshRendererState {
                renderable_index: 4,
                ..Default::default()
            },
            crate::shared::MeshRendererState {
                renderable_index: -1,
                ..Default::default()
            },
        ],
        ..Default::default()
    });
    let mut report = SceneApplyReport::default();

    note_render_world_dirty_for_extracted_update(&mut report, RenderSpaceId(3), false, 0, &update);

    assert_eq!(report.render_world_dirty.full_spaces, Vec::new());
    assert_eq!(report.render_world_dirty.renderers.len(), 1);
    assert_eq!(
        report.render_world_dirty.renderers[0].kind,
        RenderWorldRendererKind::Static
    );
    assert_eq!(report.render_world_dirty.renderers[0].renderable_index, 4);
}

#[test]
fn render_world_dirty_report_tracks_skinned_bounds_separately() {
    let mut update = empty_extracted_render_space_update();
    update.skinned_meshes = Some(
        crate::scene::meshes::ExtractedSkinnedMeshRenderablesUpdate {
            bounds_updates: vec![
                crate::shared::SkinnedMeshBoundsUpdate {
                    renderable_index: 2,
                    local_bounds: crate::shared::RenderBoundingBox::default(),
                },
                crate::shared::SkinnedMeshBoundsUpdate {
                    renderable_index: -1,
                    local_bounds: crate::shared::RenderBoundingBox::default(),
                },
            ],
            ..Default::default()
        },
    );
    let mut report = SceneApplyReport::default();

    note_render_world_dirty_for_extracted_update(&mut report, RenderSpaceId(3), false, 0, &update);

    assert_eq!(report.render_world_dirty.bounds.len(), 1);
    assert_eq!(
        report.render_world_dirty.bounds[0].kind,
        RenderWorldRendererKind::Skinned
    );
    assert_eq!(report.render_world_dirty.bounds[0].renderable_index, 2);
    assert!(report.render_world_dirty.renderers.is_empty());
    assert!(report.render_world_dirty.full_spaces.is_empty());
}

#[test]
fn render_world_dirty_report_marks_mesh_membership_as_full_space() {
    let mut update = empty_extracted_render_space_update();
    update.meshes = Some(crate::scene::meshes::ExtractedMeshRenderablesUpdate {
        removals: vec![1, -1],
        ..Default::default()
    });
    let mut report = SceneApplyReport::default();

    note_render_world_dirty_for_extracted_update(&mut report, RenderSpaceId(3), false, 0, &update);

    assert_eq!(
        report.render_world_dirty.full_spaces,
        vec![RenderSpaceId(3)]
    );
    assert!(report.render_world_dirty.renderers.is_empty());
}

#[test]
fn render_world_dirty_report_marks_lod_groups_as_full_space() {
    let mut update = empty_extracted_render_space_update();
    update.lod_groups =
        Some(crate::scene::lod_groups::ExtractedLodGroupRenderablesUpdate::default());
    let mut report = SceneApplyReport::default();

    note_render_world_dirty_for_extracted_update(&mut report, RenderSpaceId(3), false, 0, &update);

    assert_eq!(
        report.render_world_dirty.full_spaces,
        vec![RenderSpaceId(3)]
    );
    assert!(report.render_world_dirty.renderers.is_empty());
}

#[test]
fn render_world_dirty_report_tracks_transform_pose_roots() {
    let mut update = empty_extracted_render_space_update();
    update.transforms = Some(crate::scene::transforms::ExtractedTransformsUpdate {
        pose_updates: vec![
            crate::shared::TransformPoseUpdate {
                transform_id: 2,
                pose: RenderTransform::default(),
            },
            crate::shared::TransformPoseUpdate {
                transform_id: -1,
                pose: RenderTransform::default(),
            },
        ],
        target_transform_count: 5,
        ..Default::default()
    });
    let mut report = SceneApplyReport::default();

    note_render_world_dirty_for_extracted_update(&mut report, RenderSpaceId(4), false, 5, &update);

    assert_eq!(
        report.render_world_dirty.transform_roots[0].root_node_ids,
        vec![2]
    );
    assert!(report.render_world_dirty.full_spaces.is_empty());
}

#[test]
fn render_world_dirty_report_tracks_material_override_targets() {
    let mut update = empty_extracted_render_space_update();
    update.material_overrides = Some(
        crate::scene::overrides::ExtractedRenderMaterialOverridesUpdate {
            states: vec![
                crate::shared::RenderMaterialOverrideState {
                    renderable_index: 0,
                    packed_mesh_renderer_index: (1 << 30) | 7,
                    context: crate::shared::RenderingContext::Camera,
                    ..Default::default()
                },
                crate::shared::RenderMaterialOverrideState {
                    renderable_index: -1,
                    ..Default::default()
                },
            ],
            ..Default::default()
        },
    );
    let mut report = SceneApplyReport::default();

    note_render_world_dirty_for_extracted_update(&mut report, RenderSpaceId(5), false, 0, &update);

    assert_eq!(report.render_world_dirty.material_overrides.len(), 1);
    assert!(report.render_world_dirty.full_spaces.is_empty());
}

/// [`super::super::apply::apply_extracted_render_space_update`] mutates only the per-space
/// inputs it is given: pre-extracted payloads with non-identity poses commit into the right
/// dense slots and report a dirty world cache so the caller can flag the space for re-flush.
#[test]
fn parallel_apply_extracted_commits_pose_writes_and_marks_dirty() {
    use crate::scene::transforms::ExtractedTransformsUpdate;
    use crate::shared::TransformPoseUpdate;

    use super::super::apply::{PerSpaceApplyInputs, apply_extracted_render_space_update};

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
        camera_portals: None,
        reflection_probes: None,
        layers: None,
        lod_groups: None,
        transform_overrides: None,
        material_overrides: None,
        blit_to_displays: None,
        billboard_render_buffers: None,
        mesh_render_buffers: None,
        trail_render_buffers: None,
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
