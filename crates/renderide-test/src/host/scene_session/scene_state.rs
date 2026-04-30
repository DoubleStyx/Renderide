//! Builds the scene-state shared-memory region and pumps the lockstep until the renderer has
//! seen at least one frame submission carrying the scene.

use std::time::{Duration, Instant};

use renderide_shared::ipc::HostDualQueueIpc;
use renderide_shared::wire_writer::render_space::{
    build_sphere_render_space_update, SphereSceneInputs, SphereSceneSharedMemoryLayout,
    SphereSceneSharedMemoryRegions,
};
use renderide_shared::{SharedMemoryWriter, SharedMemoryWriterConfig};

use crate::error::HarnessError;

use super::super::lockstep::LockstepDriver;
use super::consts::{asset_ids, timing};

/// Holds the scene SHM writer alive so the renderer can keep reading the descriptor over many
/// lockstep ticks. [`Drop`] releases the shared-memory mapping.
pub(super) struct SceneState {
    /// Live writer keeping the scene-state SHM region alive.
    _writer: SharedMemoryWriter,
    /// Region descriptors retained for potential future re-sends; currently inert after the
    /// initial `set_render_space`.
    _regions: SphereSceneSharedMemoryRegions,
}

/// Builds the scene-state SHM region, writes the four sub-regions, and latches the resulting
/// `RenderSpaceUpdate` into the lockstep driver so subsequent `FrameSubmitData` carries the scene.
pub(super) fn build_scene_state(
    prefix: &str,
    lockstep: &mut LockstepDriver,
) -> Result<SceneState, HarnessError> {
    let defaults = SphereSceneInputs::default();
    let inputs = SphereSceneInputs {
        render_space_id: asset_ids::RENDER_SPACE,
        camera_world_pose: defaults.camera_world_pose,
        object_pose: defaults.object_pose,
        mesh_asset_id: asset_ids::SPHERE_MESH,
        material_asset_id: asset_ids::SPHERE_MATERIAL,
    };
    let regions = SphereSceneSharedMemoryRegions::build(&inputs);
    let total_bytes = regions.total_bytes();
    let cfg = SharedMemoryWriterConfig {
        prefix: prefix.to_string(),
        destroy_on_drop: true,
    };
    let mut writer = SharedMemoryWriter::open(cfg, asset_ids::SCENE_STATE_BUFFER, total_bytes)
        .map_err(|e| {
            HarnessError::QueueOptions(format!("open scene-state SHM (cap={total_bytes}): {e}"))
        })?;

    let layout = SphereSceneSharedMemoryLayout::pack_back_to_back(
        asset_ids::SCENE_STATE_BUFFER,
        total_bytes as i32,
        &regions,
    );
    writer
        .write_at(
            layout.pose_updates_offset as usize,
            &regions.pose_updates_bytes,
        )
        .map_err(|e| HarnessError::QueueOptions(format!("write pose_updates: {e}")))?;
    writer
        .write_at(layout.additions_offset as usize, &regions.additions_bytes)
        .map_err(|e| HarnessError::QueueOptions(format!("write additions: {e}")))?;
    writer
        .write_at(
            layout.mesh_states_offset as usize,
            &regions.mesh_states_bytes,
        )
        .map_err(|e| HarnessError::QueueOptions(format!("write mesh_states: {e}")))?;
    writer
        .write_at(
            layout.packed_material_ids_offset as usize,
            &regions.packed_material_ids_bytes,
        )
        .map_err(|e| HarnessError::QueueOptions(format!("write packed_material_ids: {e}")))?;
    writer.flush();

    let render_space = build_sphere_render_space_update(&inputs, &regions, &layout);
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
