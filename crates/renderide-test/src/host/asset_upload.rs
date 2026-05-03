//! Background-channel asset upload helpers.
//!
//! Writes mesh bytes via [`SharedMemoryWriter`], sends `MeshUploadData`/`ShaderUpload`/
//! `MaterialsUpdateBatch` on the Background queue, and pumps the lockstep loop while waiting
//! for the matching ack so the renderer's frame-start lockstep doesn't deadlock during the
//! upload.

use std::path::Path;
use std::time::{Duration, Instant};

use glam::IVec2;
use renderide_shared::buffer::SharedMemoryBufferDescriptor;
use renderide_shared::ipc::HostDualQueueIpc;
use renderide_shared::shared::{
    ColorProfile, MATERIAL_PROPERTY_UPDATE_HOST_ROW_BYTES, MaterialPropertyIdRequest,
    MaterialPropertyUpdateType, MaterialsUpdateBatch, RendererCommand, SetTexture2DData,
    SetTexture2DFormat, SetTexture2DProperties, ShaderUpload, TextureFilterMode, TextureFormat,
    TextureUpdateResultType, TextureUploadHint, TextureWrapMode,
};
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

/// Owns the SHM writers backing a [`MaterialsUpdateBatch`] so every region the batch
/// references stays mapped for as long as the renderer might re-read it. [`Drop`] tears down
/// the mappings.
pub(super) struct BoundMaterial {
    /// Live writers keeping the per-stream SHM regions alive (main row stream + side
    /// buffers). Order is irrelevant; they are simply held to keep file mappings open.
    _writers: Vec<SharedMemoryWriter>,
}

/// Single op in a material-update batch. Each variant becomes one row in the main stream
/// (8 bytes, see [`MATERIAL_PROPERTY_UPDATE_HOST_ROW_BYTES`]) and may also push a value into
/// one of the side buffers (`int_buffers`, `float_buffers`, `float4_buffers`).
#[derive(Clone, Copy, Debug)]
pub(super) enum MaterialUpdateOp {
    /// Direct-to-row opcode: row carries `material_asset_id` in `property_id`.
    SelectTarget { material_asset_id: i32 },
    /// Direct-to-row opcode: row carries the shader asset id in `property_id`.
    SetShader { shader_asset_id: i32 },
    /// Row carries the property id; the packed texture handle (e.g. `(kind << 29) | asset_id`
    /// for [`crate::host::asset_upload::pack_texture2d_handle`]) is appended to the int side
    /// buffer.
    SetTexture {
        property_id: i32,
        packed_handle: i32,
    },
    /// Row carries the property id; the four floats are appended to the float4 side buffer.
    SetFloat4 { property_id: i32, value: [f32; 4] },
    /// Stream terminator. The renderer expects the batch parser to see this opcode at the end
    /// of every per-target run.
    UpdateBatchEnd,
}

/// Per-call inputs for [`apply_material_batch`].
pub(super) struct MaterialBatchRequest<'a> {
    /// Shared-memory prefix matching `RendererInitData.shared_memory_prefix`.
    pub shared_memory_prefix: &'a str,
    /// Per-session backing directory passed to [`SharedMemoryWriterConfig::dir_override`].
    pub backing_dir: &'a Path,
    /// Base SHM buffer id; the row stream takes this id, the int side buffer takes
    /// `base_buffer_id + 1`, the float4 side buffer takes `base_buffer_id + 2`.
    pub base_buffer_id: i32,
    /// `update_batch_id` echoed back in the `MaterialsUpdateBatchResult` ack.
    pub update_batch_id: i32,
    /// Number of `SelectTarget` opcodes that route to materials (vs. property blocks). The
    /// renderer's parser uses this to decide whether each `SelectTarget` row points at a
    /// material asset id or a property-block asset id; for our flow every target is a
    /// material so this matches the count of `SelectTarget` ops in `ops`.
    pub material_update_count: i32,
    /// The ordered list of opcodes to encode and apply.
    pub ops: &'a [MaterialUpdateOp],
    /// Deadline for receiving `MaterialsUpdateBatchResult`.
    pub timeout: Duration,
}

/// Encodes `ops` into the appropriate SHM regions and sends a `MaterialsUpdateBatch` on the
/// background queue, blocking on `MaterialsUpdateBatchResult` while pumping the lockstep loop.
///
/// The batch may carry up to three SHM regions: the main row stream, an int side buffer (used
/// by `SetTexture` opcodes), and a float4 side buffer (used by `SetFloat4` opcodes). Each is
/// only allocated when the corresponding op type is present.
pub(super) fn apply_material_batch(
    queues: &mut HostDualQueueIpc,
    lockstep: &mut LockstepDriver,
    request: MaterialBatchRequest<'_>,
) -> Result<BoundMaterial, HarnessError> {
    let encoded = encode_material_batch(request.ops);

    let mut writers: Vec<SharedMemoryWriter> = Vec::new();

    let row_stream_writer = open_writer(
        request.shared_memory_prefix,
        request.backing_dir,
        request.base_buffer_id,
        &encoded.row_stream,
        "material_updates",
    )?;
    let row_stream_descriptor =
        row_stream_writer.descriptor_for(0, encoded.row_stream.len() as i32);
    writers.push(row_stream_writer);

    let mut int_buffers: Vec<SharedMemoryBufferDescriptor> = Vec::new();
    if !encoded.int_bytes.is_empty() {
        let w = open_writer(
            request.shared_memory_prefix,
            request.backing_dir,
            request.base_buffer_id + 1,
            &encoded.int_bytes,
            "material_updates_int",
        )?;
        int_buffers.push(w.descriptor_for(0, encoded.int_bytes.len() as i32));
        writers.push(w);
    }

    let mut float4_buffers: Vec<SharedMemoryBufferDescriptor> = Vec::new();
    if !encoded.float4_bytes.is_empty() {
        let w = open_writer(
            request.shared_memory_prefix,
            request.backing_dir,
            request.base_buffer_id + 2,
            &encoded.float4_bytes,
            "material_updates_float4",
        )?;
        float4_buffers.push(w.descriptor_for(0, encoded.float4_bytes.len() as i32));
        writers.push(w);
    }

    let batch = MaterialsUpdateBatch {
        update_batch_id: request.update_batch_id,
        material_updates: vec![row_stream_descriptor],
        material_update_count: request.material_update_count,
        int_buffers,
        float_buffers: Vec::new(),
        float4_buffers,
        matrix_buffers: Vec::new(),
        instance_changed_buffer: Default::default(),
    };

    if !queues.send_background(RendererCommand::MaterialsUpdateBatch(batch)) {
        return Err(HarnessError::QueueOptions(
            "send_background(MaterialsUpdateBatch) returned false (queue full?)".to_string(),
        ));
    }
    logger::info!(
        "AssetUpload: sent MaterialsUpdateBatch(batch={batch_id}, ops={n_ops})",
        batch_id = request.update_batch_id,
        n_ops = request.ops.len(),
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

    Ok(BoundMaterial { _writers: writers })
}

/// Packs an asset-id Texture2D handle the same way the renderer's
/// [`crate::shared::ShaderUpload`]-side `IdPacker<TextureAssetType>` decoder expects. The
/// Texture2D kind tag is `0`, so the packed value is just `asset_id` cast to `i32` — but
/// keep the helper for clarity and to make future Texture3D / Cubemap handles obvious.
pub(super) fn pack_texture2d_handle(asset_id: i32) -> i32 {
    // `Texture2D` enum tag = 0; the unpack code uses `necessary_bits(6) = 3` high bits. With
    // a zero tag and 29 low bits, the packed handle is just the asset id.
    asset_id
}

/// Bytes produced by [`encode_material_batch`] for each of the three SHM regions a
/// [`MaterialsUpdateBatch`] may carry.
struct EncodedMaterialBatch {
    row_stream: Vec<u8>,
    int_bytes: Vec<u8>,
    float4_bytes: Vec<u8>,
}

fn encode_material_batch(ops: &[MaterialUpdateOp]) -> EncodedMaterialBatch {
    const ROW_BYTES: usize = MATERIAL_PROPERTY_UPDATE_HOST_ROW_BYTES;
    let mut row_stream = vec![0u8; ops.len() * ROW_BYTES];
    let mut int_bytes: Vec<u8> = Vec::new();
    let mut float4_bytes: Vec<u8> = Vec::new();
    for (i, op) in ops.iter().enumerate() {
        let off = i * ROW_BYTES;
        let (property_id, opcode) = match *op {
            MaterialUpdateOp::SelectTarget { material_asset_id } => {
                (material_asset_id, MaterialPropertyUpdateType::SelectTarget)
            }
            MaterialUpdateOp::SetShader { shader_asset_id } => {
                (shader_asset_id, MaterialPropertyUpdateType::SetShader)
            }
            MaterialUpdateOp::SetTexture {
                property_id,
                packed_handle,
            } => {
                int_bytes.extend_from_slice(&packed_handle.to_le_bytes());
                (property_id, MaterialPropertyUpdateType::SetTexture)
            }
            MaterialUpdateOp::SetFloat4 { property_id, value } => {
                for v in value {
                    float4_bytes.extend_from_slice(&v.to_le_bytes());
                }
                (property_id, MaterialPropertyUpdateType::SetFloat4)
            }
            MaterialUpdateOp::UpdateBatchEnd => (0, MaterialPropertyUpdateType::UpdateBatchEnd),
        };
        row_stream[off..off + 4].copy_from_slice(&property_id.to_le_bytes());
        row_stream[off + 4] = opcode as u8;
        // bytes [off+5..off+8] remain zero (padding)
    }
    EncodedMaterialBatch {
        row_stream,
        int_bytes,
        float4_bytes,
    }
}

fn open_writer(
    prefix: &str,
    backing_dir: &Path,
    buffer_id: i32,
    bytes: &[u8],
    label: &str,
) -> Result<SharedMemoryWriter, HarnessError> {
    let cfg = SharedMemoryWriterConfig {
        prefix: prefix.to_string(),
        destroy_on_drop: true,
        dir_override: Some(backing_dir.to_path_buf()),
    };
    let mut writer = SharedMemoryWriter::open(cfg, buffer_id, bytes.len()).map_err(|e| {
        HarnessError::QueueOptions(format!(
            "SharedMemoryWriter::open({label} buffer={buffer_id}, cap={}): {e}",
            bytes.len(),
        ))
    })?;
    writer
        .write_at(0, bytes)
        .map_err(|e| HarnessError::QueueOptions(format!("write {label} bytes: {e}")))?;
    writer.flush();
    Ok(writer)
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

/// Per-call inputs for [`request_property_ids`].
pub(super) struct PropertyIdLookup<'a> {
    /// Echoed back by the renderer in `MaterialPropertyIdResult.request_id`.
    pub request_id: i32,
    /// Property names to intern (e.g. `["_Tex", "_Tex_ST"]`).
    pub names: &'a [&'a str],
    /// Deadline for receiving `MaterialPropertyIdResult`.
    pub timeout: Duration,
}

/// Sends a [`MaterialPropertyIdRequest`] and blocks on the matching
/// [`renderide_shared::shared::MaterialPropertyIdResult`] while pumping the lockstep loop.
/// Returns the per-name `i32` ids in the same order as `request.names`.
pub(super) fn request_property_ids(
    queues: &mut HostDualQueueIpc,
    lockstep: &mut LockstepDriver,
    request: PropertyIdLookup<'_>,
) -> Result<Vec<i32>, HarnessError> {
    let req = MaterialPropertyIdRequest {
        request_id: request.request_id,
        property_names: request
            .names
            .iter()
            .map(|n| Some((*n).to_string()))
            .collect(),
    };
    if !queues.send_background(RendererCommand::MaterialPropertyIdRequest(req)) {
        return Err(HarnessError::QueueOptions(
            "send_background(MaterialPropertyIdRequest) returned false (queue full?)".to_string(),
        ));
    }
    logger::info!(
        "AssetUpload: sent MaterialPropertyIdRequest(request_id={req_id}, names={names:?})",
        req_id = request.request_id,
        names = request.names,
    );
    let deadline = Instant::now() + request.timeout;
    while Instant::now() < deadline {
        let tick = lockstep.tick(queues);
        for msg in tick.other_messages {
            if let RendererCommand::MaterialPropertyIdResult(r) = msg
                && r.request_id == request.request_id
            {
                logger::info!(
                    "AssetUpload: received MaterialPropertyIdResult(request_id={req_id}, ids={ids:?})",
                    req_id = request.request_id,
                    ids = r.property_ids
                );
                return Ok(r.property_ids);
            }
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    Err(HarnessError::AssetAckTimeout(
        request.timeout,
        "MaterialPropertyIdResult never arrived",
    ))
}

/// Per-call inputs for [`upload_texture2d_rgba8`].
pub(super) struct Texture2DUploadRequest<'a> {
    /// Shared-memory prefix matching `RendererInitData.shared_memory_prefix`.
    pub shared_memory_prefix: &'a str,
    /// Per-session backing directory passed to [`SharedMemoryWriterConfig::dir_override`].
    pub backing_dir: &'a Path,
    /// Buffer id assigned to the texture-data SHM region.
    pub buffer_id: i32,
    /// Renderer-side asset id echoed back in `SetTexture2DResult`.
    pub asset_id: i32,
    /// Texture width in pixels.
    pub width: u32,
    /// Texture height in pixels.
    pub height: u32,
    /// `width * height * 4` RGBA8 bytes (row-major, no padding).
    pub rgba_bytes: &'a [u8],
    /// sRGB vs Linear color profile. Non-color textures (height/normal/etc.) should pass
    /// `Linear`; albedo/diffuse should pass one of the SRGB variants so the renderer
    /// linearizes when sampling.
    pub color_profile: ColorProfile,
    /// Deadline for receiving the data-upload portion of `SetTexture2DResult`.
    pub timeout: Duration,
}

/// Owns the live SHM writer backing a Texture2D upload so the renderer's read mapping stays
/// valid for the rest of the session. [`Drop`] tears down the file mapping.
pub(super) struct UploadedTexture {
    _writer: SharedMemoryWriter,
}

/// Uploads an RGBA8 texture in three IPC messages (`SetTexture2DFormat`,
/// `SetTexture2DProperties`, `SetTexture2DData`) and blocks on the data-upload portion of
/// `SetTexture2DResult` (`TextureUpdateResultType::DATA_UPLOAD`) while pumping the lockstep
/// loop. Filter mode is bilinear, wrap mode is repeat — sane defaults for procedural
/// textures.
pub(super) fn upload_texture2d_rgba8(
    queues: &mut HostDualQueueIpc,
    lockstep: &mut LockstepDriver,
    request: Texture2DUploadRequest<'_>,
) -> Result<UploadedTexture, HarnessError> {
    let expected_bytes = (request.width as usize) * (request.height as usize) * 4;
    if request.rgba_bytes.len() != expected_bytes {
        return Err(HarnessError::QueueOptions(format!(
            "RGBA8 texture {width}x{height} expects {expected_bytes} bytes, got {got}",
            width = request.width,
            height = request.height,
            got = request.rgba_bytes.len(),
        )));
    }

    // Send Format + Properties; both are processed synchronously by the renderer.
    let format = SetTexture2DFormat {
        asset_id: request.asset_id,
        width: request.width as i32,
        height: request.height as i32,
        mipmap_count: 1,
        format: TextureFormat::RGBA32,
        profile: request.color_profile,
    };
    if !queues.send_background(RendererCommand::SetTexture2DFormat(format)) {
        return Err(HarnessError::QueueOptions(
            "send_background(SetTexture2DFormat) returned false (queue full?)".to_string(),
        ));
    }
    let properties = SetTexture2DProperties {
        asset_id: request.asset_id,
        filter_mode: TextureFilterMode::Bilinear,
        aniso_level: 1,
        wrap_u: TextureWrapMode::Repeat,
        wrap_v: TextureWrapMode::Repeat,
        mipmap_bias: 0.0,
        apply_immediatelly: true,
        high_priority: true,
    };
    if !queues.send_background(RendererCommand::SetTexture2DProperties(properties)) {
        return Err(HarnessError::QueueOptions(
            "send_background(SetTexture2DProperties) returned false (queue full?)".to_string(),
        ));
    }

    // Open SHM, copy pixels in, send Data referencing the mapped region.
    let writer = open_writer(
        request.shared_memory_prefix,
        request.backing_dir,
        request.buffer_id,
        request.rgba_bytes,
        "texture2d_data",
    )?;
    let descriptor = writer.descriptor_for(0, request.rgba_bytes.len() as i32);
    let data = SetTexture2DData {
        asset_id: request.asset_id,
        data: descriptor,
        start_mip_level: 0,
        mip_map_sizes: vec![IVec2 {
            x: request.width as i32,
            y: request.height as i32,
        }],
        mip_starts: vec![0],
        flip_y: false,
        hint: TextureUploadHint::default(),
        high_priority: true,
    };
    if !queues.send_background(RendererCommand::SetTexture2DData(data)) {
        return Err(HarnessError::QueueOptions(
            "send_background(SetTexture2DData) returned false (queue full?)".to_string(),
        ));
    }
    logger::info!(
        "AssetUpload: sent SetTexture2D{{Format,Properties,Data}}(asset_id={asset}, {w}x{h}, bytes={bytes})",
        asset = request.asset_id,
        w = request.width,
        h = request.height,
        bytes = request.rgba_bytes.len(),
    );

    wait_for_texture_data_upload_result(queues, lockstep, request.asset_id, request.timeout)?;
    logger::info!(
        "AssetUpload: received SetTexture2DResult(asset_id={asset}, DATA_UPLOAD)",
        asset = request.asset_id
    );

    Ok(UploadedTexture { _writer: writer })
}

fn wait_for_texture_data_upload_result(
    queues: &mut HostDualQueueIpc,
    lockstep: &mut LockstepDriver,
    asset_id: i32,
    timeout: Duration,
) -> Result<(), HarnessError> {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        let tick = lockstep.tick(queues);
        for msg in tick.other_messages {
            if let RendererCommand::SetTexture2DResult(r) = msg
                && r.asset_id == asset_id
                && (r.r#type.0 & TextureUpdateResultType::DATA_UPLOAD) != 0
            {
                return Ok(());
            }
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    Err(HarnessError::AssetAckTimeout(
        timeout,
        "SetTexture2DResult(DATA_UPLOAD) never arrived",
    ))
}

#[cfg(test)]
mod tests {
    use super::{MaterialUpdateOp, encode_material_batch, pack_texture2d_handle};
    use renderide_shared::shared::{
        MATERIAL_PROPERTY_UPDATE_HOST_ROW_BYTES, MaterialPropertyUpdateType,
    };

    #[test]
    fn encodes_one_row_per_eight_bytes() {
        let encoded = encode_material_batch(&[
            MaterialUpdateOp::SelectTarget {
                material_asset_id: 42,
            },
            MaterialUpdateOp::SetShader { shader_asset_id: 7 },
            MaterialUpdateOp::UpdateBatchEnd,
        ]);
        assert_eq!(
            encoded.row_stream.len(),
            3 * MATERIAL_PROPERTY_UPDATE_HOST_ROW_BYTES
        );
        assert!(encoded.int_bytes.is_empty());
        assert!(encoded.float4_bytes.is_empty());
    }

    #[test]
    fn select_target_row_carries_material_asset_id() {
        let encoded = encode_material_batch(&[MaterialUpdateOp::SelectTarget {
            material_asset_id: 42,
        }]);
        let property_id = i32::from_le_bytes([
            encoded.row_stream[0],
            encoded.row_stream[1],
            encoded.row_stream[2],
            encoded.row_stream[3],
        ]);
        assert_eq!(property_id, 42);
        assert_eq!(
            encoded.row_stream[4],
            MaterialPropertyUpdateType::SelectTarget as u8
        );
    }

    #[test]
    fn set_shader_opcode_is_one() {
        let encoded = encode_material_batch(&[MaterialUpdateOp::SetShader { shader_asset_id: 0 }]);
        assert_eq!(encoded.row_stream[4], 1);
    }

    #[test]
    fn update_batch_end_opcode_is_eleven() {
        let encoded = encode_material_batch(&[MaterialUpdateOp::UpdateBatchEnd]);
        assert_eq!(encoded.row_stream[4], 11);
    }

    #[test]
    fn padding_bytes_remain_zero() {
        let encoded = encode_material_batch(&[MaterialUpdateOp::SetShader { shader_asset_id: 0 }]);
        assert_eq!(&encoded.row_stream[5..8], &[0, 0, 0]);
    }

    #[test]
    fn set_texture_appends_packed_handle_to_int_buffer() {
        let encoded = encode_material_batch(&[MaterialUpdateOp::SetTexture {
            property_id: 5,
            packed_handle: 0x00AB_CD01,
        }]);
        assert_eq!(encoded.int_bytes.len(), 4);
        let packed = i32::from_le_bytes([
            encoded.int_bytes[0],
            encoded.int_bytes[1],
            encoded.int_bytes[2],
            encoded.int_bytes[3],
        ]);
        assert_eq!(packed, 0x00AB_CD01);
        let property_id = i32::from_le_bytes([
            encoded.row_stream[0],
            encoded.row_stream[1],
            encoded.row_stream[2],
            encoded.row_stream[3],
        ]);
        assert_eq!(property_id, 5);
        assert_eq!(
            encoded.row_stream[4],
            MaterialPropertyUpdateType::SetTexture as u8
        );
    }

    #[test]
    fn set_float4_appends_four_floats_to_float4_buffer() {
        let encoded = encode_material_batch(&[MaterialUpdateOp::SetFloat4 {
            property_id: 9,
            value: [1.0, 2.0, 3.0, 4.0],
        }]);
        assert_eq!(encoded.float4_bytes.len(), 16);
        let floats: Vec<f32> = encoded
            .float4_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(floats, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn pack_texture2d_handle_uses_zero_kind_tag() {
        // Texture2D enum tag = 0; the unpack code expects the tag in the high 3 bits and the
        // asset id in the low 29. With a zero tag the packed handle is just the asset id.
        assert_eq!(pack_texture2d_handle(0), 0);
        assert_eq!(pack_texture2d_handle(42), 42);
        assert_eq!(pack_texture2d_handle(1234), 1234);
    }
}
