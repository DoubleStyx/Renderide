//! Mesh shared-memory upload and acknowledgement handling.

use std::path::Path;
use std::time::Duration;

use renderide_shared::SharedMemoryWriter;
use renderide_shared::ipc::HostDualQueueIpc;
use renderide_shared::shared::RendererCommand;

use crate::error::HarnessError;
use crate::scene::mesh_payload::{MeshUpload, make_mesh_upload_data};

use super::super::command_wait::wait_for_command;
use super::super::lockstep::LockstepDriver;
use super::shared::open_writer;

/// Owns the open `SharedMemoryWriter` for the mesh buffer so the harness can keep the shared
/// memory alive until the renderer is shut down.
pub(in crate::host) struct UploadedMesh {
    /// Live writer keeping the SHM buffer alive; released on `Drop`.
    _writer: SharedMemoryWriter,
}

/// Per-call inputs for [`upload_mesh`].
pub(in crate::host) struct MeshUploadRequest<'a> {
    /// Shared-memory prefix matching `RendererInitData.shared_memory_prefix`.
    pub shared_memory_prefix: &'a str,
    /// Per-session backing directory passed to `SharedMemoryWriterConfig::dir_override`.
    pub backing_dir: &'a Path,
    /// Buffer id assigned to this mesh's SHM region.
    pub buffer_id: i32,
    /// Renderer-side asset id echoed back in the `MeshUploadResult` ack.
    pub asset_id: i32,
    /// Packed mesh payload to upload.
    pub mesh: &'a MeshUpload,
    /// Deadline for receiving `MeshUploadResult`.
    pub timeout: Duration,
}

/// Uploads `request.mesh` as a `MeshUploadData` against `request.asset_id`.
pub(in crate::host) fn upload_mesh(
    queues: &mut HostDualQueueIpc,
    lockstep: &mut LockstepDriver,
    request: MeshUploadRequest<'_>,
) -> Result<UploadedMesh, HarnessError> {
    let writer = open_writer(
        request.shared_memory_prefix,
        request.backing_dir,
        request.buffer_id,
        &request.mesh.payload.bytes,
        "mesh",
    )?;
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

/// Waits for the matching `MeshUploadResult`.
fn wait_for_mesh_upload_result(
    queues: &mut HostDualQueueIpc,
    lockstep: &mut LockstepDriver,
    asset_id: i32,
    timeout: Duration,
) -> Result<(), HarnessError> {
    wait_for_command(
        queues,
        lockstep,
        timeout,
        |wait| HarnessError::AssetAckTimeout(wait, "MeshUploadResult never arrived"),
        |msg| match msg {
            RendererCommand::MeshUploadResult(result) if result.asset_id == asset_id => Some(()),
            _ => None,
        },
    )
}
