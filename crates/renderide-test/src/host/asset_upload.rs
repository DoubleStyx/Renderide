//! Background-channel asset upload helpers.
//!
//! Writes the sphere mesh bytes via [`SharedMemoryWriter`], sends `MeshUploadData` on the
//! Background queue, and pumps the lockstep loop while waiting for `MeshUploadResult` so the
//! renderer's frame-start lockstep doesn't deadlock during the upload.

use std::time::{Duration, Instant};

use renderide_shared::ipc::HostDualQueueIpc;
use renderide_shared::shared::RendererCommand;
use renderide_shared::{SharedMemoryWriter, SharedMemoryWriterConfig};

use crate::error::HarnessError;
use crate::scene::mesh_payload::{make_mesh_upload_data, SphereMeshUpload};

use super::lockstep::LockstepDriver;

/// Default deadline for receiving `MeshUploadResult` after sending `MeshUploadData`.
pub const DEFAULT_ASSET_UPLOAD_TIMEOUT: Duration = Duration::from_secs(10);

/// Owns the open `SharedMemoryWriter` for the sphere mesh buffer so the harness can keep the
/// shared memory alive until the renderer is shut down (the renderer's `SharedMemoryAccessor`
/// only holds a read mapping; the host owns the backing).
///
/// Fields are intentionally accessed only via [`Drop`] (writer keeps SHM alive); we expose them
/// publicly so future host-side code can re-derive descriptors without re-opening the writer.
#[allow(dead_code)]
pub struct UploadedMesh {
    /// Asset id assigned to the uploaded mesh.
    pub asset_id: i32,
    /// Live writer keeping the SHM buffer alive.
    pub writer: SharedMemoryWriter,
}

/// Uploads `mesh` as a `MeshUploadData` against `asset_id`, blocking on `MeshUploadResult` while
/// pumping the lockstep loop.
pub fn upload_sphere_mesh(
    queues: &mut HostDualQueueIpc,
    lockstep: &mut LockstepDriver,
    shared_memory_prefix: &str,
    buffer_id: i32,
    asset_id: i32,
    mesh: &SphereMeshUpload,
    timeout: Duration,
) -> Result<UploadedMesh, HarnessError> {
    let cfg = SharedMemoryWriterConfig {
        prefix: shared_memory_prefix.to_string(),
        destroy_on_drop: true,
    };
    let capacity = mesh.payload.bytes.len();
    let mut writer = SharedMemoryWriter::open(cfg, buffer_id, capacity).map_err(|e| {
        HarnessError::QueueOptions(format!(
            "SharedMemoryWriter::open(prefix={shared_memory_prefix}, buffer={buffer_id}, cap={capacity}): {e}"
        ))
    })?;
    writer
        .write_at(0, &mesh.payload.bytes)
        .map_err(|e| HarnessError::QueueOptions(format!("write mesh bytes: {e}")))?;
    writer.flush();

    let buffer_descriptor = writer.descriptor_for(0, mesh.payload.bytes.len() as i32);
    let upload = make_mesh_upload_data(mesh, asset_id, buffer_descriptor)
        .map_err(|e| HarnessError::QueueOptions(format!("compose MeshUploadData: {e}")))?;

    if !queues.send_background(RendererCommand::MeshUploadData(upload)) {
        return Err(HarnessError::QueueOptions(
            "send_background(MeshUploadData) returned false (queue full?)".to_string(),
        ));
    }
    logger::info!(
        "AssetUpload: sent MeshUploadData(asset_id={asset_id}, bytes={})",
        mesh.payload.bytes.len()
    );

    wait_for_mesh_upload_result(queues, lockstep, asset_id, timeout)?;
    logger::info!("AssetUpload: received MeshUploadResult(asset_id={asset_id})");

    Ok(UploadedMesh { asset_id, writer })
}

fn wait_for_mesh_upload_result(
    queues: &mut HostDualQueueIpc,
    lockstep: &mut LockstepDriver,
    asset_id: i32,
    timeout: Duration,
) -> Result<(), HarnessError> {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        let tick = lockstep.tick(queues);
        for msg in tick.other_messages {
            if let RendererCommand::MeshUploadResult(r) = msg {
                if r.asset_id == asset_id {
                    return Ok(());
                }
            }
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    Err(HarnessError::AssetAckTimeout(
        timeout,
        "MeshUploadResult never arrived",
    ))
}
