//! Material property lookup and material-update batch encoding.

use std::path::Path;
use std::time::Duration;

use renderide_shared::SharedMemoryWriter;
use renderide_shared::buffer::SharedMemoryBufferDescriptor;
use renderide_shared::ipc::HostDualQueueIpc;
use renderide_shared::shared::{
    MATERIAL_PROPERTY_UPDATE_HOST_ROW_BYTES, MaterialPropertyIdRequest, MaterialPropertyUpdateType,
    MaterialsUpdateBatch, RendererCommand,
};

use crate::error::HarnessError;

use super::super::command_wait::wait_for_command;
use super::super::lockstep::LockstepDriver;
use super::shared::open_writer;

/// Owns the SHM writers backing a [`MaterialsUpdateBatch`] so every region the batch references
/// stays mapped for as long as the renderer might re-read it.
pub(in crate::host) struct BoundMaterial {
    /// Live writers keeping the per-stream SHM regions alive.
    _writers: Vec<SharedMemoryWriter>,
}

/// Single op in a material-update batch.
#[derive(Clone, Copy, Debug)]
pub(in crate::host) enum MaterialUpdateOp {
    /// Direct-to-row opcode: row carries `material_asset_id` in `property_id`.
    SelectTarget {
        /// Material asset id selected by this row.
        material_asset_id: i32,
    },
    /// Direct-to-row opcode: row carries the shader asset id in `property_id`.
    SetShader {
        /// Shader asset id selected by this row.
        shader_asset_id: i32,
    },
    /// Row carries the property id; the packed texture handle is appended to the int buffer.
    SetTexture {
        /// Interned material property id.
        property_id: i32,
        /// Packed texture handle written to the int side buffer.
        packed_handle: i32,
    },
    /// Row carries the property id; one float is appended to the float side buffer.
    SetFloat {
        /// Interned material property id.
        property_id: i32,
        /// Scalar property value.
        value: f32,
    },
    /// Row carries the property id; the four floats are appended to the float4 side buffer.
    SetFloat4 {
        /// Interned material property id.
        property_id: i32,
        /// Float4 property value.
        value: [f32; 4],
    },
    /// Stream terminator for the current target run.
    UpdateBatchEnd,
}

/// Per-call inputs for [`apply_material_batch`].
pub(in crate::host) struct MaterialBatchRequest<'a> {
    /// Shared-memory prefix matching `RendererInitData.shared_memory_prefix`.
    pub shared_memory_prefix: &'a str,
    /// Per-session backing directory passed to `SharedMemoryWriterConfig::dir_override`.
    pub backing_dir: &'a Path,
    /// Base SHM buffer id for row, int, float, and float4 streams.
    pub base_buffer_id: i32,
    /// `update_batch_id` echoed back in the `MaterialsUpdateBatchResult` ack.
    pub update_batch_id: i32,
    /// Number of `SelectTarget` opcodes that route to materials.
    pub material_update_count: i32,
    /// The ordered list of opcodes to encode and apply.
    pub ops: &'a [MaterialUpdateOp],
    /// Deadline for receiving `MaterialsUpdateBatchResult`.
    pub timeout: Duration,
}

/// Encodes `ops`, sends a `MaterialsUpdateBatch`, and waits for its acknowledgement.
pub(in crate::host) fn apply_material_batch(
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
        let writer = open_writer(
            request.shared_memory_prefix,
            request.backing_dir,
            request.base_buffer_id + 1,
            &encoded.int_bytes,
            "material_updates_int",
        )?;
        int_buffers.push(writer.descriptor_for(0, encoded.int_bytes.len() as i32));
        writers.push(writer);
    }

    let mut float_buffers: Vec<SharedMemoryBufferDescriptor> = Vec::new();
    if !encoded.float_bytes.is_empty() {
        let writer = open_writer(
            request.shared_memory_prefix,
            request.backing_dir,
            request.base_buffer_id + 2,
            &encoded.float_bytes,
            "material_updates_float",
        )?;
        float_buffers.push(writer.descriptor_for(0, encoded.float_bytes.len() as i32));
        writers.push(writer);
    }

    let mut float4_buffers: Vec<SharedMemoryBufferDescriptor> = Vec::new();
    if !encoded.float4_bytes.is_empty() {
        let writer = open_writer(
            request.shared_memory_prefix,
            request.backing_dir,
            request.base_buffer_id + 3,
            &encoded.float4_bytes,
            "material_updates_float4",
        )?;
        float4_buffers.push(writer.descriptor_for(0, encoded.float4_bytes.len() as i32));
        writers.push(writer);
    }

    let batch = MaterialsUpdateBatch {
        update_batch_id: request.update_batch_id,
        material_updates: vec![row_stream_descriptor],
        material_update_count: request.material_update_count,
        int_buffers,
        float_buffers,
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

/// Packs an asset-id Texture2D handle in the format the renderer's texture decoder expects.
pub(in crate::host) fn pack_texture2d_handle(asset_id: i32) -> i32 {
    asset_id
}

/// Bytes produced by [`encode_material_batch`] for each SHM region.
struct EncodedMaterialBatch {
    /// Main material-update row stream.
    row_stream: Vec<u8>,
    /// Side buffer for integer payloads.
    int_bytes: Vec<u8>,
    /// Side buffer for scalar float payloads.
    float_bytes: Vec<u8>,
    /// Side buffer for float4 payloads.
    float4_bytes: Vec<u8>,
}

/// Encodes material-update operations into renderer wire rows and side buffers.
fn encode_material_batch(ops: &[MaterialUpdateOp]) -> EncodedMaterialBatch {
    const ROW_BYTES: usize = MATERIAL_PROPERTY_UPDATE_HOST_ROW_BYTES;
    let mut row_stream = vec![0u8; ops.len() * ROW_BYTES];
    let mut int_bytes: Vec<u8> = Vec::new();
    let mut float_bytes: Vec<u8> = Vec::new();
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
            MaterialUpdateOp::SetFloat { property_id, value } => {
                float_bytes.extend_from_slice(&value.to_le_bytes());
                (property_id, MaterialPropertyUpdateType::SetFloat)
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
    }
    EncodedMaterialBatch {
        row_stream,
        int_bytes,
        float_bytes,
        float4_bytes,
    }
}

/// Waits for the matching `MaterialsUpdateBatchResult`.
fn wait_for_materials_update_batch_result(
    queues: &mut HostDualQueueIpc,
    lockstep: &mut LockstepDriver,
    update_batch_id: i32,
    timeout: Duration,
) -> Result<(), HarnessError> {
    wait_for_command(
        queues,
        lockstep,
        timeout,
        |wait| HarnessError::AssetAckTimeout(wait, "MaterialsUpdateBatchResult never arrived"),
        |msg| {
            if let RendererCommand::MaterialsUpdateBatchResult(result) = msg
                && result.update_batch_id == update_batch_id
            {
                Some(())
            } else {
                None
            }
        },
    )
}

/// Per-call inputs for [`request_property_ids`].
pub(in crate::host) struct PropertyIdLookup<'a> {
    /// Echoed back by the renderer in `MaterialPropertyIdResult.request_id`.
    pub request_id: i32,
    /// Property names to intern.
    pub names: &'a [&'a str],
    /// Deadline for receiving `MaterialPropertyIdResult`.
    pub timeout: Duration,
}

/// Sends a `MaterialPropertyIdRequest` and returns the ids in `request.names` order.
pub(in crate::host) fn request_property_ids(
    queues: &mut HostDualQueueIpc,
    lockstep: &mut LockstepDriver,
    request: PropertyIdLookup<'_>,
) -> Result<Vec<i32>, HarnessError> {
    let req = MaterialPropertyIdRequest {
        request_id: request.request_id,
        property_names: request
            .names
            .iter()
            .map(|name| Some((*name).to_string()))
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

    let property_ids = wait_for_material_property_id_result(
        queues,
        lockstep,
        request.request_id,
        request.timeout,
    )?;
    logger::info!(
        "AssetUpload: received MaterialPropertyIdResult(request_id={req_id}, ids={property_ids:?})",
        req_id = request.request_id,
    );
    Ok(property_ids)
}

/// Waits for the matching `MaterialPropertyIdResult`.
fn wait_for_material_property_id_result(
    queues: &mut HostDualQueueIpc,
    lockstep: &mut LockstepDriver,
    request_id: i32,
    timeout: Duration,
) -> Result<Vec<i32>, HarnessError> {
    wait_for_command(
        queues,
        lockstep,
        timeout,
        |wait| HarnessError::AssetAckTimeout(wait, "MaterialPropertyIdResult never arrived"),
        |msg| match msg {
            RendererCommand::MaterialPropertyIdResult(result)
                if result.request_id == request_id =>
            {
                Some(result.property_ids)
            }
            _ => None,
        },
    )
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
        assert!(encoded.float_bytes.is_empty());
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
    fn set_float_appends_one_float_to_float_buffer() {
        let encoded = encode_material_batch(&[MaterialUpdateOp::SetFloat {
            property_id: 8,
            value: 0.625,
        }]);
        assert_eq!(encoded.float_bytes.len(), 4);
        let value = f32::from_le_bytes([
            encoded.float_bytes[0],
            encoded.float_bytes[1],
            encoded.float_bytes[2],
            encoded.float_bytes[3],
        ]);
        assert_eq!(value, 0.625);
        let property_id = i32::from_le_bytes([
            encoded.row_stream[0],
            encoded.row_stream[1],
            encoded.row_stream[2],
            encoded.row_stream[3],
        ]);
        assert_eq!(property_id, 8);
        assert_eq!(
            encoded.row_stream[4],
            MaterialPropertyUpdateType::SetFloat as u8
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
        assert_eq!(pack_texture2d_handle(0), 0);
        assert_eq!(pack_texture2d_handle(42), 42);
        assert_eq!(pack_texture2d_handle(1234), 1234);
    }
}
