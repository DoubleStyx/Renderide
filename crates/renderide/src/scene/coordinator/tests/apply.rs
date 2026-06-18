//! Phase-orchestration tests: render-world header / extracted update dirtiness plus the
//! per-space apply commit that the parallel apply path drives.

use std::sync::atomic::{AtomicU32, Ordering};

use glam::{Quat, Vec3};
use renderide_shared::wire_writer::{TransformPoseRow, encode_transform_pose_updates};
use renderide_shared::{SharedMemoryWriter, SharedMemoryWriterConfig};

use crate::ipc::SharedMemoryAccessor;
use crate::scene::camera_portal::CameraPortalEntry;
use crate::scene::meshes::types::StaticMeshRenderer;
use crate::scene::overrides::{MeshRendererOverrideTarget, RenderMaterialOverrideEntry};
use crate::scene::render_space::RenderSpaceState;
use crate::shared::{
    CameraPortalState, FrameSubmitData, RenderSpaceUpdate, RenderTransform, RenderingContext,
    TransformsUpdate,
};

use super::super::super::ids::RenderSpaceId;
use super::super::super::world::{WorldTransformCache, compute_world_matrices_for_space};
use super::super::apply::ExtractedRenderSpaceUpdate;
use super::super::{
    RenderWorldRendererKind, SceneApplyReport, SceneCoordinator,
    extracted_update_affects_reflection_probes, extracted_update_affects_render_world,
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

static SHARED_MEMORY_PREFIX_SEQ: AtomicU32 = AtomicU32::new(0);

fn unique_shared_memory_prefix(label: &str) -> String {
    let seq = SHARED_MEMORY_PREFIX_SEQ.fetch_add(1, Ordering::Relaxed);
    format!(
        "renderide_scene_coordinator_{label}_{}_{}",
        std::process::id(),
        seq
    )
}

fn identity_transform() -> RenderTransform {
    RenderTransform {
        position: Vec3::ZERO,
        scale: Vec3::ONE,
        rotation: Quat::IDENTITY,
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
fn apply_frame_submit_applies_inactive_render_space_payloads() {
    let mut scene = SceneCoordinator::new();
    let space_id = RenderSpaceId(7);
    scene.test_seed_space_identity_worlds(space_id, vec![identity_transform()], vec![-1]);

    let updated_pose = RenderTransform {
        position: Vec3::new(4.0, 5.0, 6.0),
        scale: Vec3::ONE,
        rotation: Quat::IDENTITY,
    };
    let rows = [
        TransformPoseRow {
            transform_id: 0,
            pose: updated_pose,
        },
        TransformPoseRow {
            transform_id: -1,
            pose: RenderTransform::default(),
        },
    ];
    let bytes = encode_transform_pose_updates(&rows);
    let prefix = unique_shared_memory_prefix("inactive_space_payload");
    let cfg = SharedMemoryWriterConfig {
        prefix: prefix.clone(),
        destroy_on_drop: true,
        ..SharedMemoryWriterConfig::default()
    };
    let mut writer = SharedMemoryWriter::open(cfg, 1, bytes.len()).expect("open writer");
    writer.write_at(0, &bytes).expect("write transform rows");
    writer.flush();
    let pose_descriptor = writer.descriptor_for(0, bytes.len() as i32);
    let mut shm = SharedMemoryAccessor::new(prefix);

    let report = scene
        .apply_frame_submit(
            &mut shm,
            &FrameSubmitData {
                frame_index: 42,
                render_spaces: vec![RenderSpaceUpdate {
                    id: space_id.0,
                    is_active: false,
                    transforms_update: Some(TransformsUpdate {
                        target_transform_count: 1,
                        pose_updates: pose_descriptor,
                        ..TransformsUpdate::default()
                    }),
                    ..RenderSpaceUpdate::default()
                }],
                ..FrameSubmitData::default()
            },
        )
        .expect("apply inactive submitted payload");

    let space = scene.space(space_id).expect("space remains tracked");
    assert!(!space.is_active());
    assert_eq!(space.local_transforms()[0].position, updated_pose.position);
    assert_eq!(report.changed_spaces, vec![space_id]);
    assert_eq!(report.removed_spaces, Vec::new());
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
fn extracted_reflection_probe_dirty_tracks_probe_updates() {
    let mut update = empty_extracted_render_space_update();
    update.reflection_probes =
        Some(crate::scene::reflection_probe::ExtractedReflectionProbeRenderablesUpdate::default());

    assert!(extracted_update_affects_reflection_probes(&update));
}

#[test]
fn extracted_reflection_probe_dirty_tracks_transform_updates() {
    let mut update = empty_extracted_render_space_update();
    update.transforms = Some(crate::scene::transforms::ExtractedTransformsUpdate::default());

    assert!(extracted_update_affects_reflection_probes(&update));
}

#[test]
fn extracted_reflection_probe_dirty_tracks_transform_overrides() {
    let mut update = empty_extracted_render_space_update();
    update.transform_overrides =
        Some(crate::scene::overrides::ExtractedRenderTransformOverridesUpdate::default());

    assert!(extracted_update_affects_reflection_probes(&update));
}

#[test]
fn reflection_probe_dirty_spaces_deduplicate() {
    let mut report = SceneApplyReport::default();
    report.note_reflection_probe_dirty_space(RenderSpaceId(4));
    report.note_reflection_probe_dirty_space(RenderSpaceId(4));

    assert_eq!(report.reflection_probe_dirty_spaces, vec![RenderSpaceId(4)]);
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

    note_render_world_dirty_for_extracted_update(
        &mut report,
        RenderSpaceId(3),
        false,
        0,
        None,
        &update,
    );

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

    note_render_world_dirty_for_extracted_update(
        &mut report,
        RenderSpaceId(3),
        false,
        0,
        None,
        &update,
    );

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

    note_render_world_dirty_for_extracted_update(
        &mut report,
        RenderSpaceId(3),
        false,
        0,
        None,
        &update,
    );

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

    note_render_world_dirty_for_extracted_update(
        &mut report,
        RenderSpaceId(3),
        false,
        0,
        None,
        &update,
    );

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

    note_render_world_dirty_for_extracted_update(
        &mut report,
        RenderSpaceId(4),
        false,
        5,
        None,
        &update,
    );

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
                    context: RenderingContext::Camera,
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

    note_render_world_dirty_for_extracted_update(
        &mut report,
        RenderSpaceId(5),
        false,
        0,
        None,
        &update,
    );

    assert_eq!(report.render_world_dirty.material_overrides.len(), 1);
    assert!(report.render_world_dirty.full_spaces.is_empty());
}

#[test]
fn render_world_dirty_report_tracks_material_override_previous_and_new_targets() {
    let mut space = RenderSpaceState::default();
    space
        .render_material_overrides
        .push(RenderMaterialOverrideEntry {
            node_id: 2,
            context: RenderingContext::Camera,
            target: MeshRendererOverrideTarget::Static(4),
            ..Default::default()
        });
    let mut update = empty_extracted_render_space_update();
    update.material_overrides = Some(
        crate::scene::overrides::ExtractedRenderMaterialOverridesUpdate {
            states: vec![
                crate::shared::RenderMaterialOverrideState {
                    renderable_index: 0,
                    packed_mesh_renderer_index: (1 << 30) | 7,
                    context: RenderingContext::Mirror,
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

    note_render_world_dirty_for_extracted_update(
        &mut report,
        RenderSpaceId(5),
        false,
        0,
        Some(&space),
        &update,
    );

    let dirty_targets: Vec<_> = report
        .render_world_dirty
        .material_overrides
        .iter()
        .map(|dirty| (dirty.context, dirty.target))
        .collect();
    assert_eq!(
        dirty_targets,
        vec![
            (
                RenderingContext::Camera,
                MeshRendererOverrideTarget::Static(4)
            ),
            (
                RenderingContext::Mirror,
                MeshRendererOverrideTarget::Skinned(7)
            ),
        ]
    );
    assert!(report.render_world_dirty.full_spaces.is_empty());
}

#[test]
fn render_world_dirty_report_marks_unknown_previous_material_override_target_as_full_space() {
    let mut space = RenderSpaceState::default();
    space
        .render_material_overrides
        .push(RenderMaterialOverrideEntry {
            node_id: 2,
            context: RenderingContext::Portal,
            target: MeshRendererOverrideTarget::Unknown,
            ..Default::default()
        });
    let mut update = empty_extracted_render_space_update();
    update.material_overrides = Some(
        crate::scene::overrides::ExtractedRenderMaterialOverridesUpdate {
            states: vec![crate::shared::RenderMaterialOverrideState {
                renderable_index: 0,
                packed_mesh_renderer_index: 9,
                context: RenderingContext::UserView,
                ..Default::default()
            }],
            ..Default::default()
        },
    );
    let mut report = SceneApplyReport::default();

    note_render_world_dirty_for_extracted_update(
        &mut report,
        RenderSpaceId(5),
        false,
        0,
        Some(&space),
        &update,
    );

    assert_eq!(
        report.render_world_dirty.full_spaces,
        vec![RenderSpaceId(5)]
    );
    assert!(report.render_world_dirty.material_overrides.is_empty());
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

#[test]
fn parallel_apply_extracted_remaps_camera_portal_static_mesh_target_before_mesh_removal() {
    use super::super::apply::{PerSpaceApplyInputs, apply_extracted_render_space_update};

    let mut space = RenderSpaceState {
        id: RenderSpaceId(7),
        static_mesh_renderers: vec![StaticMeshRenderer::default(); 4],
        camera_portals: vec![CameraPortalEntry {
            renderable_index: 0,
            transform_id: 0,
            state: CameraPortalState {
                mesh_renderer_index: 3,
                ..CameraPortalState::default()
            },
        }],
        ..Default::default()
    };
    let mut cache = WorldTransformCache::default();
    let mut extracted = empty_extracted_render_space_update();
    extracted.space_id = RenderSpaceId(7);
    extracted.meshes = Some(crate::scene::meshes::ExtractedMeshRenderablesUpdate {
        removals: vec![1, -1],
        ..Default::default()
    });
    let mut removal_events = Vec::new();

    let dirty = apply_extracted_render_space_update(
        &extracted,
        PerSpaceApplyInputs {
            space: &mut space,
            cache: &mut cache,
            removal_events: &mut removal_events,
        },
    );

    assert!(!dirty, "mesh removal must not dirty the world matrix cache");
    assert_eq!(space.static_mesh_renderers.len(), 3);
    assert_eq!(space.camera_portals[0].state.mesh_renderer_index, 1);
}
