//! Background-channel asset upload helpers.
//!
//! Writes the sphere mesh bytes via [`SharedMemoryWriter`], sends `MeshUploadData` on the
//! Background queue, and pumps the lockstep loop while waiting for `MeshUploadResult` so the
//! renderer's frame-start lockstep doesn't deadlock during the upload.

use std::path::Path;
use std::time::{Duration, Instant};

use renderide_shared::ipc::HostDualQueueIpc;
use renderide_shared::shared::RendererCommand;
use renderide_shared::{SharedMemoryWriter, SharedMemoryWriterConfig};

use crate::error::HarnessError;
use crate::scene::mesh_payload::{SphereMeshUpload, make_mesh_upload_data};

use super::lockstep::LockstepDriver;

/// Default deadline for receiving `MeshUploadResult` after sending `MeshUploadData`.
pub(super) const DEFAULT_ASSET_UPLOAD_TIMEOUT: Duration = Duration::from_secs(10);

/// Owns the open `SharedMemoryWriter` for the sphere mesh buffer so the harness can keep the
/// shared memory alive until the renderer is shut down (the renderer's `SharedMemoryAccessor`
/// only holds a read mapping; the host owns the backing).
///
/// The writer is held purely for its [`Drop`] semantics -- releasing it tears down the SHM
/// mapping the renderer is reading.
pub(super) struct UploadedMesh {
    /// Live writer keeping the SHM buffer alive; released on `Drop`.
    _writer: SharedMemoryWriter,
}

/// Per-call inputs for [`upload_sphere_mesh`]. Bundled to keep the function under clippy's
/// `too_many_arguments` cap and to give callers one focused place to set up an upload.
pub(super) struct MeshUploadRequest<'a> {
    /// Shared-memory prefix matching `RendererInitData.shared_memory_prefix`.
    pub shared_memory_prefix: &'a str,
    /// Per-session backing directory passed to [`SharedMemoryWriterConfig::dir_override`].
    pub backing_dir: &'a Path,
    /// Buffer id assigned to this mesh's SHM region.
    pub buffer_id: i32,
    /// Renderer-side asset id echoed back in the `MeshUploadResult` ack.
    pub asset_id: i32,
    /// Packed mesh payload to upload.
    pub mesh: &'a SphereMeshUpload,
    /// Deadline for receiving `MeshUploadResult`.
    pub timeout: Duration,
}

/// Uploads `request.mesh` as a `MeshUploadData` against `request.asset_id`, blocking on
/// `MeshUploadResult` while pumping the lockstep loop.
pub(super) fn upload_sphere_mesh(
    queues: &mut HostDualQueueIpc,
    lockstep: &mut LockstepDriver,
    request: MeshUploadRequest<'_>,
) -> Result<UploadedMesh, HarnessError> {
    let cfg = SharedMemoryWriterConfig {
        prefix: request.shared_memory_prefix.to_string(),
        destroy_on_drop: true,
        dir_override: Some(request.backing_dir.to_path_buf()),
    };
    let capacity = request.mesh.payload.bytes.len();
    let mut writer = SharedMemoryWriter::open(cfg, request.buffer_id, capacity).map_err(|e| {
        HarnessError::QueueOptions(format!(
            "SharedMemoryWriter::open(prefix={prefix}, buffer={buffer}, cap={capacity}): {e}",
            prefix = request.shared_memory_prefix,
            buffer = request.buffer_id,
        ))
    })?;
    writer
        .write_at(0, &request.mesh.payload.bytes)
        .map_err(|e| HarnessError::QueueOptions(format!("write mesh bytes: {e}")))?;
    writer.flush();

    let buffer_descriptor = writer.descriptor_for(0, request.mesh.payload.bytes.len() as i32);
    let upload = make_mesh_upload_data(request.mesh, request.asset_id, buffer_descriptor)
        .map_err(|e| HarnessError::QueueOptions(format!("compose MeshUploadData: {e}")))?;

    if !queues.send_background(RendererCommand::MeshUploadData(upload)) {
        return Err(HarnessError::QueueOptions(
            "send_background(MeshUploadData) returned false (queue full?)".to_string(),
        ));
    }
    logger::info!(
        "AssetUpload: sent MeshUploadData(asset_id={asset}, bytes={})",
        request.mesh.payload.bytes.len(),
        asset = request.asset_id,
    );

    wait_for_mesh_upload_result(queues, lockstep, request.asset_id, request.timeout)?;
    logger::info!(
        "AssetUpload: received MeshUploadResult(asset_id={asset})",
        asset = request.asset_id
    );

    Ok(UploadedMesh { _writer: writer })
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
            if let RendererCommand::MeshUploadResult(r) = msg
                && r.asset_id == asset_id
            {
                return Ok(());
            }
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    Err(HarnessError::AssetAckTimeout(
        timeout,
        "MeshUploadResult never arrived",
    ))
}
