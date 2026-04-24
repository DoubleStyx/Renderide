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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::BlendshapeBufferDescriptor;

    fn base_data() -> MeshUploadData {
        MeshUploadData {
            asset_id: 42,
            vertex_count: 8,
            ..Default::default()
        }
    }

    fn base_layout() -> MeshBufferLayout {
        MeshBufferLayout {
            vertex_size: 0,
            index_buffer_start: 0,
            index_buffer_length: 0,
            bone_counts_start: 0,
            bone_counts_length: 0,
            bone_weights_start: 0,
            bone_weights_length: 0,
            bind_poses_start: 0,
            bind_poses_length: 0,
            blendshape_data_start: 0,
            blendshape_data_length: 0,
            total_buffer_length: 0,
        }
    }

    #[test]
    fn layout_fingerprint_is_deterministic() {
        let d = base_data();
        let l = base_layout();
        assert_eq!(
            mesh_layout_fingerprint(&d, &l),
            mesh_layout_fingerprint(&d, &l)
        );
    }

    #[test]
    fn layout_fingerprint_changes_with_asset_id() {
        let a = base_data();
        let mut b = base_data();
        b.asset_id = a.asset_id + 1;
        let l = base_layout();
        assert_ne!(
            mesh_layout_fingerprint(&a, &l),
            mesh_layout_fingerprint(&b, &l)
        );
    }

    #[test]
    fn layout_fingerprint_changes_with_vertex_count() {
        let a = base_data();
        let mut b = base_data();
        b.vertex_count = a.vertex_count + 1;
        let l = base_layout();
        assert_ne!(
            mesh_layout_fingerprint(&a, &l),
            mesh_layout_fingerprint(&b, &l)
        );
    }

    #[test]
    fn input_fingerprint_is_deterministic() {
        let d = base_data();
        assert_eq!(
            mesh_upload_input_fingerprint(&d),
            mesh_upload_input_fingerprint(&d)
        );
    }

    #[test]
    fn input_fingerprint_changes_with_blendshape_frame_weight() {
        let mut a = base_data();
        a.blendshape_buffers.push(BlendshapeBufferDescriptor {
            blendshape_index: 0,
            frame_index: 0,
            frame_weight: 1.0,
            ..Default::default()
        });
        let mut b = a.clone();
        b.blendshape_buffers[0].frame_weight = 0.5;
        assert_ne!(
            mesh_upload_input_fingerprint(&a),
            mesh_upload_input_fingerprint(&b)
        );
    }
}
