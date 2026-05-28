use glam::Mat4;

use super::super::layout::{
    compute_index_count, compute_mesh_buffer_layout, compute_vertex_stride,
};
use crate::shared::{
    SubmeshBufferDescriptor, SubmeshTopology, VertexAttributeDescriptor, VertexAttributeFormat,
    VertexAttributeType,
};

#[test]
fn layout_no_bones_no_blend_matches_stride() {
    let sub = vec![SubmeshBufferDescriptor {
        topology: SubmeshTopology::default(),
        index_start: 0,
        index_count: 3,
        bounds: crate::shared::RenderBoundingBox::default(),
    }];
    let ic = compute_index_count(&sub);
    let l = compute_mesh_buffer_layout(32, 2, ic, 2, 0, 0, None).unwrap();
    assert_eq!(l.vertex_size, 64);
    assert_eq!(l.index_buffer_start, 64);
    assert_eq!(l.index_buffer_length, 6);
    assert_eq!(l.bone_counts_length, 2);
    assert_eq!(l.bone_weights_length, 0);
    assert_eq!(l.bind_poses_length, 0);
    assert_eq!(l.total_buffer_length, 64 + 6 + 2);
}

#[test]
fn layout_negative_bone_counts_clamped() {
    let sub = vec![SubmeshBufferDescriptor {
        topology: SubmeshTopology::default(),
        index_start: 0,
        index_count: 3,
        bounds: crate::shared::RenderBoundingBox::default(),
    }];
    let ic = compute_index_count(&sub);
    let l = compute_mesh_buffer_layout(32, 2, ic, 2, -1, -1, None).unwrap();
    assert_eq!(l.vertex_size, 64);
    assert_eq!(l.index_buffer_start, 64);
    assert_eq!(l.index_buffer_length, 6);
    assert_eq!(l.bone_counts_length, 2);
    assert_eq!(l.bone_weights_length, 0);
    assert_eq!(l.bind_poses_length, 0);
    assert_eq!(l.total_buffer_length, 64 + 6 + 2);
}

#[test]
fn vertex_stride_sum() {
    let attrs = [
        VertexAttributeDescriptor {
            attribute: VertexAttributeType::Position,
            format: VertexAttributeFormat::Float32,
            dimensions: 3,
        },
        VertexAttributeDescriptor {
            attribute: VertexAttributeType::Normal,
            format: VertexAttributeFormat::Float32,
            dimensions: 3,
        },
    ];
    assert_eq!(compute_vertex_stride(&attrs), 24);
}

/// Unity uploads inverse bind matrices; the renderer stores them as [`Mat4::from_cols_array_2d`]
/// without an extra `.inverse()` (see [`crate::assets::mesh::GpuMesh::skinning_bind_matrices`]).
#[test]
fn unity_bindpose_raw_matches_glam_columns_not_inverse() {
    let expected = Mat4::from_translation(glam::Vec3::new(1.0, 2.0, 3.0));
    let a = expected.to_cols_array();
    let raw: [[f32; 4]; 4] = [
        [a[0], a[1], a[2], a[3]],
        [a[4], a[5], a[6], a[7]],
        [a[8], a[9], a[10], a[11]],
        [a[12], a[13], a[14], a[15]],
    ];
    let stored = Mat4::from_cols_array_2d(&raw);
    assert!(stored.abs_diff_eq(expected, 1e-5));
    assert!(!stored.abs_diff_eq(expected.inverse(), 1e-2));
}
