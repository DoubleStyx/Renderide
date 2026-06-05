//! Builds scene-state shared memory and pumps lockstep until the renderer has consumed it.

use std::path::Path;
use std::time::{Duration, Instant};

use glam::{Quat, Vec3, Vec4};
use renderide_shared::buffer::SharedMemoryBufferDescriptor;
use renderide_shared::ipc::HostDualQueueIpc;
use renderide_shared::packing::memory_packable::MemoryPackable;
use renderide_shared::packing::memory_packer::MemoryPacker;
use renderide_shared::shared::{
    LIGHT_STATE_HOST_ROW_BYTES, LightRenderablesUpdate, LightState, LightType,
    MeshRenderablesUpdate, MeshRendererState, MotionVectorMode, RenderSH2, RenderSpaceUpdate,
    RenderTransform, ShadowCastMode, ShadowType, TransformsUpdate,
};
use renderide_shared::wire_writer::mesh_renderers::{
    encode_additions, encode_mesh_states, encode_packed_material_ids,
};
use renderide_shared::wire_writer::transforms::{TransformPoseRow, encode_transform_pose_updates};
use renderide_shared::{SharedMemoryWriter, SharedMemoryWriterConfig};

use crate::error::HarnessError;

use super::super::lockstep::LockstepDriver;
use super::consts::{asset_ids, timing};

/// One mesh renderer submitted by a test scene.
#[derive(Clone, Debug)]
pub(super) struct SceneRenderable {
    /// Dense transform index this renderer attaches to.
    pub transform_id: i32,
    /// Local transform pose for the renderer.
    pub pose: RenderTransform,
    /// Uploaded mesh asset id.
    pub mesh_asset_id: i32,
    /// Material asset id bound to the renderer.
    pub material_asset_id: i32,
    /// Unity-style renderer sorting order.
    pub sorting_order: i32,
}

/// One regular light submitted by a test scene.
#[derive(Clone, Debug)]
pub(super) struct SceneLight {
    /// Dense transform index this light attaches to.
    pub transform_id: i32,
    /// Local transform pose for the light.
    pub pose: RenderTransform,
    /// Light state; [`build_scene_state`] rewrites `renderable_index` to match submission order.
    pub state: LightState,
}

/// Full scene payload written into one render-space update.
#[derive(Clone, Debug)]
pub(super) struct SceneSubmission {
    /// World-space camera pose used as `RenderSpaceUpdate.root_transform`.
    pub camera_world_pose: RenderTransform,
    /// Mesh renderers to add and update.
    pub renderables: Vec<SceneRenderable>,
    /// Regular lights to add and update.
    pub lights: Vec<SceneLight>,
}

impl SceneSubmission {
    /// Builds the historical one-mesh scene used by the original smoke cases.
    pub fn single_mesh(mesh_asset_id: i32, material_asset_id: i32) -> Self {
        Self {
            camera_world_pose: default_camera_world_pose(),
            renderables: vec![SceneRenderable {
                transform_id: 0,
                pose: identity_transform(),
                mesh_asset_id,
                material_asset_id,
                sorting_order: 0,
            }],
            lights: Vec::new(),
        }
    }
}

/// Holds the scene SHM writer alive so the renderer can keep reading the descriptor over many
/// lockstep ticks. [`Drop`] releases the shared-memory mapping.
pub(super) struct SceneState {
    /// Live writer keeping the scene-state SHM region alive.
    _writer: SharedMemoryWriter,
    /// Region bytes retained with the writer so the one-shot scene delta remains readable until
    /// the renderer consumes it.
    _regions: SceneSharedMemoryRegions,
}

#[derive(Clone, Debug)]
struct SceneSharedMemoryRegions {
    pose_updates_bytes: Vec<u8>,
    mesh_additions_bytes: Vec<u8>,
    mesh_states_bytes: Vec<u8>,
    packed_material_ids_bytes: Vec<u8>,
    light_additions_bytes: Vec<u8>,
    light_states_bytes: Vec<u8>,
}

impl SceneSharedMemoryRegions {
    fn build(scene: &SceneSubmission) -> Self {
        let pose_updates_bytes = encode_transform_pose_updates(&pose_rows(scene));
        let mesh_additions_bytes = encode_additions(
            &scene
                .renderables
                .iter()
                .map(|renderable| renderable.transform_id)
                .collect::<Vec<_>>(),
        );
        let mesh_states_bytes = encode_mesh_states(&mesh_state_rows(scene));
        let packed_material_ids_bytes = encode_packed_material_ids(
            &scene
                .renderables
                .iter()
                .map(|renderable| renderable.material_asset_id)
                .collect::<Vec<_>>(),
        );
        let light_additions_bytes = encode_additions(
            &scene
                .lights
                .iter()
                .map(|light| light.transform_id)
                .collect::<Vec<_>>(),
        );
        let light_states_bytes = encode_light_states(&light_state_rows(scene));
        Self {
            pose_updates_bytes,
            mesh_additions_bytes,
            mesh_states_bytes,
            packed_material_ids_bytes,
            light_additions_bytes,
            light_states_bytes,
        }
    }

    const fn total_bytes(&self) -> usize {
        self.pose_updates_bytes.len()
            + self.mesh_additions_bytes.len()
            + self.mesh_states_bytes.len()
            + self.packed_material_ids_bytes.len()
            + self.light_additions_bytes.len()
            + self.light_states_bytes.len()
    }
}

#[derive(Clone, Copy, Debug)]
struct SceneSharedMemoryLayout {
    pose_updates: SharedMemoryBufferDescriptor,
    mesh_additions: SharedMemoryBufferDescriptor,
    mesh_states: SharedMemoryBufferDescriptor,
    packed_material_ids: SharedMemoryBufferDescriptor,
    light_additions: SharedMemoryBufferDescriptor,
    light_states: SharedMemoryBufferDescriptor,
}

impl SceneSharedMemoryLayout {
    fn pack_back_to_back(buffer_id: i32, regions: &SceneSharedMemoryRegions) -> Self {
        let capacity = regions.total_bytes() as i32;
        let mut offset = 0i32;
        let pose_updates = descriptor_for_region(
            buffer_id,
            capacity,
            &mut offset,
            regions.pose_updates_bytes.len(),
        );
        let mesh_additions = descriptor_for_region(
            buffer_id,
            capacity,
            &mut offset,
            regions.mesh_additions_bytes.len(),
        );
        let mesh_states = descriptor_for_region(
            buffer_id,
            capacity,
            &mut offset,
            regions.mesh_states_bytes.len(),
        );
        let packed_material_ids = descriptor_for_region(
            buffer_id,
            capacity,
            &mut offset,
            regions.packed_material_ids_bytes.len(),
        );
        let light_additions = descriptor_for_region(
            buffer_id,
            capacity,
            &mut offset,
            regions.light_additions_bytes.len(),
        );
        let light_states = descriptor_for_region(
            buffer_id,
            capacity,
            &mut offset,
            regions.light_states_bytes.len(),
        );
        Self {
            pose_updates,
            mesh_additions,
            mesh_states,
            packed_material_ids,
            light_additions,
            light_states,
        }
    }
}

/// Builds the scene-state SHM region and latches a `RenderSpaceUpdate` into the lockstep driver.
pub(super) fn build_scene_state(
    prefix: &str,
    backing_dir: &Path,
    scene: &SceneSubmission,
    lockstep: &mut LockstepDriver,
) -> Result<SceneState, HarnessError> {
    let regions = SceneSharedMemoryRegions::build(scene);
    let total_bytes = regions.total_bytes();
    let cfg = SharedMemoryWriterConfig {
        prefix: prefix.to_string(),
        destroy_on_drop: true,
        dir_override: Some(backing_dir.to_path_buf()),
    };
    let mut writer = SharedMemoryWriter::open(cfg, asset_ids::SCENE_STATE_BUFFER, total_bytes)
        .map_err(|e| {
            HarnessError::QueueOptions(format!("open scene-state SHM (cap={total_bytes}): {e}"))
        })?;

    let layout =
        SceneSharedMemoryLayout::pack_back_to_back(asset_ids::SCENE_STATE_BUFFER, &regions);
    write_scene_regions(&mut writer, &layout, &regions)?;
    writer.flush();

    let render_space = build_render_space_update(scene, &layout);
    lockstep.set_render_space(Some(render_space));

    Ok(SceneState {
        _writer: writer,
        _regions: regions,
    })
}

/// Pumps the lockstep until at least one `FrameSubmitData` carrying the scene has been enqueued.
/// Returns the `frame_index` of that submission so callers can log it.
pub(super) fn ensure_scene_submitted(
    queues: &mut HostDualQueueIpc,
    lockstep: &mut LockstepDriver,
    timeout: Duration,
) -> Result<i32, HarnessError> {
    let deadline = Instant::now() + timeout;
    let frame_index_before = lockstep.current_frame_index();
    while Instant::now() < deadline {
        let tick = lockstep.tick(queues);
        if tick.frame_submits_sent > 0 {
            return Ok(frame_index_before);
        }
        std::thread::sleep(timing::SCENE_SUBMIT_POLL);
    }
    Err(HarnessError::AssetAckTimeout(
        deadline.elapsed(),
        "renderer never sent FrameStartData after scene was loaded",
    ))
}

/// Builds an identity transform.
pub(super) fn identity_transform() -> RenderTransform {
    RenderTransform {
        position: Vec3::ZERO,
        scale: Vec3::ONE,
        rotation: Quat::IDENTITY,
    }
}

/// Builds the default camera pose used by the original integration harness.
pub(super) fn default_camera_world_pose() -> RenderTransform {
    RenderTransform {
        position: Vec3::new(0.0, 0.0, -3.0),
        scale: Vec3::ONE,
        rotation: Quat::IDENTITY,
    }
}

/// Builds a no-shadow directional light row suitable for CI golden cases.
pub(super) fn directional_light(color: [f32; 3], intensity: f32) -> LightState {
    light_state(LightType::Directional, color, intensity, 50.0, 30.0)
}

/// Builds a no-shadow point light row suitable for CI golden cases.
pub(super) fn point_light(color: [f32; 3], intensity: f32, range: f32) -> LightState {
    light_state(LightType::Point, color, intensity, range, 30.0)
}

fn light_state(
    light_type: LightType,
    color: [f32; 3],
    intensity: f32,
    range: f32,
    spot_angle: f32,
) -> LightState {
    LightState {
        renderable_index: 0,
        intensity,
        range,
        spot_angle,
        color: Vec4::new(color[0], color[1], color[2], 1.0),
        shadow_strength: 0.0,
        shadow_near_plane: 0.1,
        shadow_map_resolution_override: -1,
        shadow_bias: 0.05,
        shadow_normal_bias: 0.4,
        cookie_texture_asset_id: -1,
        r#type: light_type,
        shadow_type: ShadowType::None,
        _padding: [0; 2],
    }
}

fn pose_rows(scene: &SceneSubmission) -> Vec<TransformPoseRow> {
    scene
        .renderables
        .iter()
        .map(|renderable| TransformPoseRow {
            transform_id: renderable.transform_id,
            pose: renderable.pose,
        })
        .chain(scene.lights.iter().map(|light| TransformPoseRow {
            transform_id: light.transform_id,
            pose: light.pose,
        }))
        .collect()
}

fn mesh_state_rows(scene: &SceneSubmission) -> Vec<MeshRendererState> {
    scene
        .renderables
        .iter()
        .enumerate()
        .map(|(index, renderable)| MeshRendererState {
            renderable_index: index as i32,
            mesh_asset_id: renderable.mesh_asset_id,
            material_count: 1,
            material_property_block_count: 0,
            sorting_order: renderable.sorting_order,
            shadow_cast_mode: ShadowCastMode::Off,
            motion_vector_mode: MotionVectorMode::NoMotion,
            _padding: [0; 2],
        })
        .collect()
}

fn light_state_rows(scene: &SceneSubmission) -> Vec<LightState> {
    scene
        .lights
        .iter()
        .enumerate()
        .map(|(index, light)| LightState {
            renderable_index: index as i32,
            ..light.state
        })
        .collect()
}

fn descriptor_for_region(
    buffer_id: i32,
    buffer_capacity: i32,
    offset: &mut i32,
    len: usize,
) -> SharedMemoryBufferDescriptor {
    let desc = SharedMemoryBufferDescriptor {
        buffer_id,
        buffer_capacity,
        offset: *offset,
        length: len as i32,
    };
    *offset += len as i32;
    desc
}

fn write_scene_regions(
    writer: &mut SharedMemoryWriter,
    layout: &SceneSharedMemoryLayout,
    regions: &SceneSharedMemoryRegions,
) -> Result<(), HarnessError> {
    write_region(
        writer,
        layout.pose_updates,
        &regions.pose_updates_bytes,
        "pose_updates",
    )?;
    write_region(
        writer,
        layout.mesh_additions,
        &regions.mesh_additions_bytes,
        "mesh additions",
    )?;
    write_region(
        writer,
        layout.mesh_states,
        &regions.mesh_states_bytes,
        "mesh_states",
    )?;
    write_region(
        writer,
        layout.packed_material_ids,
        &regions.packed_material_ids_bytes,
        "packed_material_ids",
    )?;
    write_region(
        writer,
        layout.light_additions,
        &regions.light_additions_bytes,
        "light additions",
    )?;
    write_region(
        writer,
        layout.light_states,
        &regions.light_states_bytes,
        "light_states",
    )
}

fn write_region(
    writer: &mut SharedMemoryWriter,
    desc: SharedMemoryBufferDescriptor,
    bytes: &[u8],
    label: &str,
) -> Result<(), HarnessError> {
    writer
        .write_at(desc.offset as usize, bytes)
        .map_err(|e| HarnessError::QueueOptions(format!("write {label}: {e}")))
}

fn build_render_space_update(
    scene: &SceneSubmission,
    layout: &SceneSharedMemoryLayout,
) -> RenderSpaceUpdate {
    RenderSpaceUpdate {
        id: asset_ids::RENDER_SPACE,
        is_active: true,
        is_overlay: false,
        is_private: false,
        root_transform: scene.camera_world_pose,
        view_position_is_external: false,
        override_view_position: false,
        skybox_material_asset_id: -1,
        ambient_light: RenderSH2::default(),
        overriden_view_transform: RenderTransform::default(),
        transforms_update: Some(TransformsUpdate {
            target_transform_count: target_transform_count(scene),
            removals: SharedMemoryBufferDescriptor::default(),
            parent_updates: SharedMemoryBufferDescriptor::default(),
            pose_updates: layout.pose_updates,
        }),
        mesh_renderers_update: Some(MeshRenderablesUpdate {
            mesh_states: layout.mesh_states,
            mesh_materials_and_property_blocks: layout.packed_material_ids,
            removals: SharedMemoryBufferDescriptor::default(),
            additions: layout.mesh_additions,
        }),
        skinned_mesh_renderers_update: None,
        lights_update: (!scene.lights.is_empty()).then_some(LightRenderablesUpdate {
            states: layout.light_states,
            removals: SharedMemoryBufferDescriptor::default(),
            additions: layout.light_additions,
        }),
        cameras_update: None,
        camera_portals_update: None,
        reflection_probes_update: None,
        reflection_probe_sh2_taks: None,
        layers_update: None,
        billboard_buffers_update: None,
        mesh_render_buffers_update: None,
        trail_renderers_update: None,
        lights_buffer_renderers_update: None,
        render_transform_overrides_update: None,
        render_material_overrides_update: None,
        blit_to_displays_update: None,
        lod_group_update: None,
        gaussian_splat_renderers_update: None,
        reflection_probe_render_tasks: Vec::new(),
    }
}

fn target_transform_count(scene: &SceneSubmission) -> i32 {
    pose_rows(scene)
        .iter()
        .map(|row| row.transform_id)
        .max()
        .map_or(0, |max_id| max_id + 1)
}

fn encode_light_states(rows: &[LightState]) -> Vec<u8> {
    let row_size = LIGHT_STATE_HOST_ROW_BYTES;
    let mut out = vec![0u8; (rows.len() + 1) * row_size];
    for (index, row) in rows.iter().enumerate() {
        pack_light_state_row(&mut out[index * row_size..(index + 1) * row_size], *row);
    }
    let sentinel = LightState {
        renderable_index: -1,
        ..LightState::default()
    };
    let start = rows.len() * row_size;
    pack_light_state_row(&mut out[start..start + row_size], sentinel);
    out
}

fn pack_light_state_row(dest: &mut [u8], mut row: LightState) {
    let mut packer = MemoryPacker::new(dest);
    row.pack(&mut packer);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_submission_single_mesh_matches_historical_region_sizes() {
        let scene = SceneSubmission::single_mesh(2, 4);
        let regions = SceneSharedMemoryRegions::build(&scene);
        assert_eq!(regions.pose_updates_bytes.len(), 44);
        assert_eq!(regions.mesh_additions_bytes.len(), 8);
        assert_eq!(regions.mesh_states_bytes.len(), 48);
        assert_eq!(regions.packed_material_ids_bytes.len(), 4);
        assert_eq!(regions.light_additions_bytes.len(), 4);
        assert_eq!(regions.light_states_bytes.len(), LIGHT_STATE_HOST_ROW_BYTES);
    }

    #[test]
    fn multi_renderable_scene_uses_transform_ids_and_material_order() {
        let scene = SceneSubmission {
            camera_world_pose: default_camera_world_pose(),
            renderables: vec![
                SceneRenderable {
                    transform_id: 2,
                    pose: identity_transform(),
                    mesh_asset_id: 10,
                    material_asset_id: 20,
                    sorting_order: 0,
                },
                SceneRenderable {
                    transform_id: 4,
                    pose: identity_transform(),
                    mesh_asset_id: 11,
                    material_asset_id: 21,
                    sorting_order: 3,
                },
            ],
            lights: Vec::new(),
        };
        let regions = SceneSharedMemoryRegions::build(&scene);
        assert_eq!(regions.mesh_additions_bytes.len(), 12);
        assert_eq!(regions.mesh_states_bytes.len(), 72);
        assert_eq!(regions.packed_material_ids_bytes.len(), 8);
        assert_eq!(target_transform_count(&scene), 5);
    }

    #[test]
    fn light_state_rows_are_indexed_and_terminated() {
        let scene = SceneSubmission {
            camera_world_pose: default_camera_world_pose(),
            renderables: Vec::new(),
            lights: vec![
                SceneLight {
                    transform_id: 0,
                    pose: identity_transform(),
                    state: point_light([1.0, 0.5, 0.25], 2.0, 3.0),
                },
                SceneLight {
                    transform_id: 1,
                    pose: identity_transform(),
                    state: directional_light([0.5, 0.5, 1.0], 1.0),
                },
            ],
        };
        let bytes = encode_light_states(&light_state_rows(&scene));
        assert_eq!(bytes.len(), 3 * LIGHT_STATE_HOST_ROW_BYTES);
        assert_eq!(i32::from_le_bytes(bytes[0..4].try_into().unwrap()), 0);
        assert_eq!(
            i32::from_le_bytes(
                bytes[LIGHT_STATE_HOST_ROW_BYTES..LIGHT_STATE_HOST_ROW_BYTES + 4]
                    .try_into()
                    .unwrap()
            ),
            1
        );
        assert_eq!(
            i32::from_le_bytes(
                bytes[2 * LIGHT_STATE_HOST_ROW_BYTES..2 * LIGHT_STATE_HOST_ROW_BYTES + 4]
                    .try_into()
                    .unwrap()
            ),
            -1
        );
    }
}
