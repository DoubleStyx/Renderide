//! Background-channel asset upload helpers.
//!
//! Writes mesh bytes via [`SharedMemoryWriter`], sends `MeshUploadData`/`ShaderUpload`/
//! `MaterialsUpdateBatch` on the Background queue, and pumps the lockstep loop while waiting
//! for the matching ack so the renderer's frame-start lockstep doesn't deadlock during the
//! upload.

use std::path::Path;
use std::time::{Duration, Instant};

use renderide_shared::ipc::HostDualQueueIpc;
use renderide_shared::shared::{MaterialsUpdateBatch, RendererCommand, ShaderUpload};
use renderide_shared::{RENDERIDE_TEST_STEM_PREFIX, SharedMemoryWriter, SharedMemoryWriterConfig};

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

/// Sends a `ShaderUpload` carrying the test-only `RENDERIDE_TEST_STEM:` sentinel so the
/// renderer routes the asset directly to an embedded WGSL stem (skipping AssetBundle parsing).
/// The `shader_name` is taken in production-style form (e.g. `"Unlit.shader"`); the renderer
/// strips the optional `.shader` extension and lowercases internally.
///
/// Blocks on `ShaderUploadResult` while pumping the lockstep loop.
pub(super) fn upload_shader(
    queues: &mut HostDualQueueIpc,
    lockstep: &mut LockstepDriver,
    asset_id: i32,
    shader_name: &str,
    timeout: Duration,
) -> Result<(), HarnessError> {
    let upload = ShaderUpload {
        asset_id,
        file: Some(format!("{RENDERIDE_TEST_STEM_PREFIX}{shader_name}")),
    };
    if !queues.send_background(RendererCommand::ShaderUpload(upload)) {
        return Err(HarnessError::QueueOptions(
            "send_background(ShaderUpload) returned false (queue full?)".to_string(),
        ));
    }
    logger::info!("AssetUpload: sent ShaderUpload(asset_id={asset_id}, sentinel={shader_name:?})");
    wait_for_shader_upload_result(queues, lockstep, asset_id, timeout)?;
    logger::info!("AssetUpload: received ShaderUploadResult(asset_id={asset_id})");
    Ok(())
}

fn wait_for_shader_upload_result(
    queues: &mut HostDualQueueIpc,
    lockstep: &mut LockstepDriver,
    asset_id: i32,
    timeout: Duration,
) -> Result<(), HarnessError> {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        let tick = lockstep.tick(queues);
        for msg in tick.other_messages {
            if let RendererCommand::ShaderUploadResult(r) = msg
                && r.asset_id == asset_id
            {
                return Ok(());
            }
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    Err(HarnessError::AssetAckTimeout(
        timeout,
        "ShaderUploadResult never arrived",
    ))
}

/// Owns the open `SharedMemoryWriter` backing a [`MaterialsUpdateBatch`] so the SHM region
/// stays mapped for as long as the renderer might re-read it. [`Drop`] tears down the mapping.
pub(super) struct BoundMaterial {
    /// Live writer keeping the material-update SHM region alive.
    _writer: SharedMemoryWriter,
}

/// Per-call inputs for [`bind_material_shader`]. Bundled to keep the function under clippy's
/// `too_many_arguments` cap.
pub(super) struct MaterialBindRequest<'a> {
    /// Shared-memory prefix matching `RendererInitData.shared_memory_prefix`.
    pub shared_memory_prefix: &'a str,
    /// Per-session backing directory passed to [`SharedMemoryWriterConfig::dir_override`].
    pub backing_dir: &'a Path,
    /// Buffer id assigned to the material-update SHM region.
    pub buffer_id: i32,
    /// `update_batch_id` echoed back in the `MaterialsUpdateBatchResult` ack.
    pub update_batch_id: i32,
    /// Material asset id that the bound shader applies to.
    pub material_asset_id: i32,
    /// Shader asset id previously registered via [`upload_shader`].
    pub shader_asset_id: i32,
    /// Deadline for receiving `MaterialsUpdateBatchResult`.
    pub timeout: Duration,
}

/// Binds `request.shader_asset_id` as the shader for `request.material_asset_id` by writing a
/// minimal three-row [`MaterialsUpdateBatch`] stream into shared memory: `SelectTarget(material)`,
/// `SetShader(shader)`, `UpdateBatchEnd`. Each row is 8 bytes (4-byte property id + 1-byte opcode
/// + 3 bytes padding), matching the renderer's host interop wire row.
///
/// Blocks on `MaterialsUpdateBatchResult` while pumping the lockstep loop.
pub(super) fn bind_material_shader(
    queues: &mut HostDualQueueIpc,
    lockstep: &mut LockstepDriver,
    request: MaterialBindRequest<'_>,
) -> Result<BoundMaterial, HarnessError> {
    use renderide_shared::shared::{
        MATERIAL_PROPERTY_UPDATE_HOST_ROW_BYTES, MaterialPropertyUpdateType,
    };
    const ROW_BYTES: usize = MATERIAL_PROPERTY_UPDATE_HOST_ROW_BYTES;
    let stream_bytes = encode_material_update_stream(&[
        (
            request.material_asset_id,
            MaterialPropertyUpdateType::SelectTarget,
        ),
        (
            request.shader_asset_id,
            MaterialPropertyUpdateType::SetShader,
        ),
        (0, MaterialPropertyUpdateType::UpdateBatchEnd),
    ]);
    debug_assert_eq!(stream_bytes.len(), 3 * ROW_BYTES);

    let cfg = SharedMemoryWriterConfig {
        prefix: request.shared_memory_prefix.to_string(),
        destroy_on_drop: true,
        dir_override: Some(request.backing_dir.to_path_buf()),
    };
    let mut writer =
        SharedMemoryWriter::open(cfg, request.buffer_id, stream_bytes.len()).map_err(|e| {
            HarnessError::QueueOptions(format!(
                "SharedMemoryWriter::open(material_updates buffer={buffer}, cap={cap}): {e}",
                buffer = request.buffer_id,
                cap = stream_bytes.len()
            ))
        })?;
    writer
        .write_at(0, &stream_bytes)
        .map_err(|e| HarnessError::QueueOptions(format!("write material update stream: {e}")))?;
    writer.flush();

    let descriptor = writer.descriptor_for(0, stream_bytes.len() as i32);
    let batch = MaterialsUpdateBatch {
        update_batch_id: request.update_batch_id,
        material_updates: vec![descriptor],
        material_update_count: 1,
        int_buffers: Vec::new(),
        float_buffers: Vec::new(),
        float4_buffers: Vec::new(),
        matrix_buffers: Vec::new(),
        instance_changed_buffer: Default::default(),
    };

    if !queues.send_background(RendererCommand::MaterialsUpdateBatch(batch)) {
        return Err(HarnessError::QueueOptions(
            "send_background(MaterialsUpdateBatch) returned false (queue full?)".to_string(),
        ));
    }
    logger::info!(
        "AssetUpload: sent MaterialsUpdateBatch(batch={batch_id}, material={mat}, shader={shader})",
        batch_id = request.update_batch_id,
        mat = request.material_asset_id,
        shader = request.shader_asset_id,
    );
    wait_for_materials_update_batch_result(
        queues,
        lockstep,
        request.update_batch_id,
        request.timeout,
    )?;
    logger::info!(
        "AssetUpload: received MaterialsUpdateBatchResult(batch={batch_id})",
        batch_id = request.update_batch_id
    );

    Ok(BoundMaterial { _writer: writer })
}

/// Encodes a sequence of (property_id, opcode) pairs into the contiguous 8-byte rows expected
/// by the renderer's `MaterialsUpdateBatch` parser (see
/// `renderide-shared::shared::MATERIAL_PROPERTY_UPDATE_HOST_ROW_BYTES`).
fn encode_material_update_stream(
    rows: &[(i32, renderide_shared::shared::MaterialPropertyUpdateType)],
) -> Vec<u8> {
    use renderide_shared::shared::MATERIAL_PROPERTY_UPDATE_HOST_ROW_BYTES as ROW_BYTES;
    let mut out = vec![0u8; rows.len() * ROW_BYTES];
    for (i, (property_id, opcode)) in rows.iter().enumerate() {
        let off = i * ROW_BYTES;
        out[off..off + 4].copy_from_slice(&property_id.to_le_bytes());
        out[off + 4] = *opcode as u8;
        // bytes [off+5..off+8] remain zero (padding)
    }
    out
}

fn wait_for_materials_update_batch_result(
    queues: &mut HostDualQueueIpc,
    lockstep: &mut LockstepDriver,
    update_batch_id: i32,
    timeout: Duration,
) -> Result<(), HarnessError> {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        let tick = lockstep.tick(queues);
        for msg in tick.other_messages {
            if let RendererCommand::MaterialsUpdateBatchResult(r) = msg
                && r.update_batch_id == update_batch_id
            {
                return Ok(());
            }
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    Err(HarnessError::AssetAckTimeout(
        timeout,
        "MaterialsUpdateBatchResult never arrived",
    ))
}

#[cfg(test)]
mod tests {
    use super::encode_material_update_stream;
    use renderide_shared::shared::{
        MATERIAL_PROPERTY_UPDATE_HOST_ROW_BYTES, MaterialPropertyUpdateType,
    };

    #[test]
    fn encodes_one_row_per_eight_bytes() {
        let bytes = encode_material_update_stream(&[
            (42, MaterialPropertyUpdateType::SelectTarget),
            (7, MaterialPropertyUpdateType::SetShader),
            (0, MaterialPropertyUpdateType::UpdateBatchEnd),
        ]);
        assert_eq!(bytes.len(), 3 * MATERIAL_PROPERTY_UPDATE_HOST_ROW_BYTES);
    }

    #[test]
    fn first_row_carries_select_target_with_property_id() {
        let bytes =
            encode_material_update_stream(&[(42, MaterialPropertyUpdateType::SelectTarget)]);
        let property_id = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let opcode = bytes[4];
        assert_eq!(property_id, 42);
        assert_eq!(opcode, MaterialPropertyUpdateType::SelectTarget as u8);
    }

    #[test]
    fn padding_bytes_remain_zero() {
        let bytes = encode_material_update_stream(&[(99, MaterialPropertyUpdateType::SetShader)]);
        assert_eq!(&bytes[5..8], &[0, 0, 0]);
    }

    #[test]
    fn set_shader_opcode_is_one() {
        let bytes = encode_material_update_stream(&[(0, MaterialPropertyUpdateType::SetShader)]);
        assert_eq!(bytes[4], 1);
    }

    #[test]
    fn update_batch_end_opcode_is_eleven() {
        let bytes =
            encode_material_update_stream(&[(0, MaterialPropertyUpdateType::UpdateBatchEnd)]);
        assert_eq!(bytes[4], 11);
    }
}
