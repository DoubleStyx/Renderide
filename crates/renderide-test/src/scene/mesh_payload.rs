//! Converts the procedural [`super::sphere::SphereMesh`] into the byte payload + descriptor
//! head fields required by `MeshUploadData`.
//!
//! The renderer-side parser expects the layout produced by
//! [`renderide_shared::wire_writer::mesh_layout::write_mesh_payload`], so we delegate the actual
//! byte interleaving to that helper and only add the high-level mesh metadata (vertex count,
//! attribute descriptors, submesh, bounds) on top.

use renderide_shared::buffer::SharedMemoryBufferDescriptor;
use renderide_shared::shared::{
    BlendshapeBufferDescriptor, IndexBufferFormat, MeshUploadData, MeshUploadHint,
    MeshUploadHintFlag, RenderBoundingBox, SubmeshBufferDescriptor, SubmeshTopology,
    VertexAttributeDescriptor,
};
use renderide_shared::wire_writer::mesh_layout::{
    self, normal_float3_attr, position_float3_attr, write_mesh_payload, InterleavedAttribute,
    MeshLayoutInput, MeshPayload,
};

use super::sphere::SphereMesh;

/// Combines the encoded SHM byte payload with an unfilled [`MeshUploadData`] head ready to receive
/// a `SharedMemoryBufferDescriptor` once the host writes the bytes.
#[derive(Clone, Debug)]
pub struct SphereMeshUpload {
    /// Bytes to write into the host shared-memory buffer at the offset chosen by the harness.
    pub payload: MeshPayload,
    /// Number of vertices in the encoded mesh (mirrors `MeshUploadData.vertex_count`).
    pub vertex_count: i32,
    /// Index format used when packing indices into [`Self::payload`]'s tail.
    pub index_buffer_format: IndexBufferFormat,
    /// Vertex attribute layout matching [`Self::payload`]'s interleaved stride.
    pub vertex_attributes: Vec<VertexAttributeDescriptor>,
    /// Single submesh covering the full index range.
    pub submeshes: Vec<SubmeshBufferDescriptor>,
    /// Conservative axis-aligned bounds (slightly larger than the unit sphere to avoid
    /// false-negative frustum culling).
    pub bounds: RenderBoundingBox,
}

/// Errors produced when packing the sphere mesh.
#[derive(Debug, thiserror::Error)]
pub enum SphereMeshUploadError {
    /// The wire-writer rejected the inputs (mismatched lengths, etc.).
    #[error("encode mesh payload: {0}")]
    Encode(#[from] mesh_layout::MeshLayoutError),
}

/// Errors produced when assembling the final [`MeshUploadData`] from a packed sphere upload.
#[derive(Debug, thiserror::Error)]
pub enum SphereMeshDescriptorError {
    /// Sphere had more vertices than fit in `i32` (impossible in practice; defensive).
    #[error("vertex count overflow: {0}")]
    VertexCountOverflow(usize),
}

/// Encodes a [`SphereMesh`] to a [`SphereMeshUpload`].
///
/// The result is independent of the asset id and the SHM descriptor so the same upload payload
/// can be reused across runs (the harness picks the asset id and writes the bytes into its own
/// shared-memory buffer).
pub fn pack_sphere_mesh_upload(
    mesh: &SphereMesh,
) -> Result<SphereMeshUpload, SphereMeshUploadError> {
    let vertex_count = mesh.vertices.len() as i32;
    let positions: Vec<u8> = mesh
        .vertices
        .iter()
        .flat_map(|v| v.position.iter().flat_map(|c| c.to_le_bytes()))
        .collect();
    let normals: Vec<u8> = mesh
        .vertices
        .iter()
        .flat_map(|v| v.normal.iter().flat_map(|c| c.to_le_bytes()))
        .collect();

    let (index_buffer_format, index_bytes, index_count) =
        if mesh.vertices.len() <= u16::MAX as usize {
            let mut bytes = Vec::with_capacity(mesh.indices.len() * 2);
            for i in &mesh.indices {
                bytes.extend_from_slice(&(*i as u16).to_le_bytes());
            }
            (IndexBufferFormat::UInt16, bytes, mesh.indices.len() as i32)
        } else {
            let mut bytes = Vec::with_capacity(mesh.indices.len() * 4);
            for i in &mesh.indices {
                bytes.extend_from_slice(&i.to_le_bytes());
            }
            (IndexBufferFormat::UInt32, bytes, mesh.indices.len() as i32)
        };

    let vertex_attributes = vec![position_float3_attr(), normal_float3_attr()];
    let payload = write_mesh_payload(&MeshLayoutInput {
        vertex_count,
        vertex_attributes: vertex_attributes.clone(),
        sources: vec![
            InterleavedAttribute { bytes: &positions },
            InterleavedAttribute { bytes: &normals },
        ],
        indices: &index_bytes,
        index_buffer_format,
    })?;

    let submeshes = vec![SubmeshBufferDescriptor {
        topology: SubmeshTopology::Triangles,
        index_start: 0,
        index_count,
        bounds: unit_sphere_bounds(),
    }];

    Ok(SphereMeshUpload {
        payload,
        vertex_count,
        index_buffer_format,
        vertex_attributes,
        submeshes,
        bounds: unit_sphere_bounds(),
    })
}

/// Builds a fully populated [`MeshUploadData`] referencing `buffer_descriptor` and `asset_id`.
pub fn make_mesh_upload_data(
    upload: &SphereMeshUpload,
    asset_id: i32,
    buffer_descriptor: SharedMemoryBufferDescriptor,
) -> Result<MeshUploadData, SphereMeshDescriptorError> {
    if upload.vertex_count < 0 {
        return Err(SphereMeshDescriptorError::VertexCountOverflow(
            upload.vertex_count.unsigned_abs() as usize,
        ));
    }
    Ok(MeshUploadData {
        high_priority: false,
        buffer: buffer_descriptor,
        vertex_count: upload.vertex_count,
        bone_weight_count: 0,
        bone_count: 0,
        index_buffer_format: upload.index_buffer_format,
        vertex_attributes: upload.vertex_attributes.clone(),
        submeshes: upload.submeshes.clone(),
        blendshape_buffers: Vec::<BlendshapeBufferDescriptor>::new(),
        upload_hint: MeshUploadHint {
            flags: MeshUploadHintFlag(0),
        },
        bounds: upload.bounds,
        asset_id,
    })
}

/// Conservative axis-aligned bounds for the unit sphere (slight outward bias keeps frustum culling
/// from false-negatives at oblique angles while staying non-degenerate).
pub fn unit_sphere_bounds() -> RenderBoundingBox {
    RenderBoundingBox {
        center: glam::Vec3::ZERO,
        extents: glam::Vec3::splat(1.05),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_sphere_with_uint16_indices() {
        let mesh = SphereMesh::generate(8, 12);
        let upload = pack_sphere_mesh_upload(&mesh).expect("pack");
        assert_eq!(upload.vertex_count as usize, mesh.vertices.len());
        assert_eq!(upload.index_buffer_format, IndexBufferFormat::UInt16);
        assert_eq!(upload.payload.vertex_stride_bytes, 12 + 12);
        // vertices (24B/vertex) + indices (2B each) + bone_counts (vertex_count zeroed bytes)
        assert_eq!(
            upload.payload.bytes.len(),
            upload.vertex_count as usize * 24
                + mesh.indices.len() * 2
                + upload.vertex_count as usize
        );
        assert_eq!(upload.submeshes.len(), 1);
        assert_eq!(upload.submeshes[0].index_count, mesh.indices.len() as i32);
    }

    #[test]
    fn make_mesh_upload_data_has_expected_fields() {
        let mesh = SphereMesh::generate(4, 6);
        let upload = pack_sphere_mesh_upload(&mesh).expect("pack");
        let descriptor = SharedMemoryBufferDescriptor {
            buffer_id: 99,
            buffer_capacity: upload.payload.bytes.len() as i32,
            offset: 0,
            length: upload.payload.bytes.len() as i32,
        };
        let upload_data = make_mesh_upload_data(&upload, 42, descriptor).expect("make upload data");
        assert_eq!(upload_data.asset_id, 42);
        assert_eq!(upload_data.vertex_count, upload.vertex_count);
        assert_eq!(upload_data.bone_count, 0);
        assert_eq!(upload_data.bone_weight_count, 0);
        assert_eq!(upload_data.index_buffer_format, IndexBufferFormat::UInt16);
        assert_eq!(upload_data.vertex_attributes.len(), 2);
        assert_eq!(upload_data.submeshes.len(), 1);
        assert!(upload_data.blendshape_buffers.is_empty());
        assert!(!upload_data.high_priority);
        assert_eq!(upload_data.buffer.buffer_id, 99);
    }
}
