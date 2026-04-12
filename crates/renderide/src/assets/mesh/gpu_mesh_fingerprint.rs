//! Stable fingerprints for mesh layout and upload inputs (no vertex/index payload hashing).

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::shared::MeshUploadData;

use super::layout::MeshBufferLayout;

/// Stable hash of host layout metadata and buffer byte sizes (for tracing and cache keys).
///
/// Does not hash the vertex/index payload bytes.
pub fn mesh_layout_fingerprint(data: &MeshUploadData, layout: &MeshBufferLayout) -> u64 {
    let mut h = DefaultHasher::new();
    data.asset_id.hash(&mut h);
    data.vertex_count.hash(&mut h);
    data.bone_count.hash(&mut h);
    data.bone_weight_count.hash(&mut h);
    (data.index_buffer_format as i32).hash(&mut h);
    data.vertex_attributes.len().hash(&mut h);
    for a in &data.vertex_attributes {
        (a.attribute as i32).hash(&mut h);
        (a.format as i32).hash(&mut h);
        a.dimensions.hash(&mut h);
    }
    data.submeshes.len().hash(&mut h);
    for s in &data.submeshes {
        (s.topology as i32).hash(&mut h);
        s.index_start.hash(&mut h);
        s.index_count.hash(&mut h);
    }
    data.blendshape_buffers.len().hash(&mut h);
    data.upload_hint.flags.0.hash(&mut h);
    layout.vertex_size.hash(&mut h);
    layout.index_buffer_length.hash(&mut h);
    layout.total_buffer_length.hash(&mut h);
    h.finish()
}

/// Fingerprint of inputs that determine [`super::layout::compute_mesh_buffer_layout`] (no raw payload bytes).
pub fn mesh_upload_input_fingerprint(data: &MeshUploadData) -> u64 {
    let mut h = DefaultHasher::new();
    data.asset_id.hash(&mut h);
    data.vertex_count.hash(&mut h);
    data.bone_count.hash(&mut h);
    data.bone_weight_count.hash(&mut h);
    (data.index_buffer_format as i32).hash(&mut h);
    data.vertex_attributes.len().hash(&mut h);
    for a in &data.vertex_attributes {
        (a.attribute as i32).hash(&mut h);
        (a.format as i32).hash(&mut h);
        a.dimensions.hash(&mut h);
    }
    data.submeshes.len().hash(&mut h);
    for s in &data.submeshes {
        (s.topology as i32).hash(&mut h);
        s.index_start.hash(&mut h);
        s.index_count.hash(&mut h);
    }
    data.blendshape_buffers.len().hash(&mut h);
    for b in &data.blendshape_buffers {
        b.blendshape_index.hash(&mut h);
        b.data_flags.0.hash(&mut h);
        b.frame_weight.to_bits().hash(&mut h);
    }
    h.finish()
}
