//! End-to-end exercise of the sphere asset pipeline:
//! [`renderide_test::scene::sphere::generate_sphere`] ->
//! [`renderide_test::scene::mesh_payload::pack_sphere_mesh_upload`] ->
//! [`renderide_test::scene::mesh_payload::make_mesh_upload_data`].
//!
//! All three stages are pure CPU code; no GPU is touched.

use renderide_shared::buffer::SharedMemoryBufferDescriptor;
use renderide_shared::shared::IndexBufferFormat;
use renderide_test::scene::mesh_payload::{
    make_mesh_upload_data, pack_sphere_mesh_upload, unit_sphere_bounds,
};
use renderide_test::scene::sphere::generate_sphere;

#[test]
fn sphere_mesh_to_upload_data_pipeline_is_self_consistent() {
    let mesh = generate_sphere(16, 24);
    assert!(!mesh.vertices.is_empty());
    assert!(!mesh.indices.is_empty());

    let upload = pack_sphere_mesh_upload(&mesh).expect("pack");
    assert_eq!(upload.vertex_count as usize, mesh.vertices.len());
    // pos + normal + uv + color = 4 attributes
    assert_eq!(upload.vertex_attributes.len(), 4);
    assert_eq!(upload.submeshes.len(), 1);
    assert_eq!(
        upload.submeshes[0].index_count,
        mesh.indices.len() as i32,
        "submesh index_count should mirror the source mesh"
    );
    assert_eq!(upload.bounds.center, unit_sphere_bounds().center);
    assert_eq!(upload.bounds.extents, unit_sphere_bounds().extents);
    assert_eq!(upload.index_buffer_format, IndexBufferFormat::UInt16);

    let descriptor = SharedMemoryBufferDescriptor {
        buffer_id: 7,
        buffer_capacity: upload.payload.bytes.len() as i32,
        offset: 0,
        length: upload.payload.bytes.len() as i32,
    };
    let upload_data = make_mesh_upload_data(&upload, 42, descriptor).expect("upload data");
    assert_eq!(upload_data.asset_id, 42);
    assert_eq!(upload_data.vertex_count, upload.vertex_count);
    assert_eq!(upload_data.index_buffer_format, IndexBufferFormat::UInt16);
    assert_eq!(upload_data.buffer.buffer_id, 7);
    assert!(!upload_data.high_priority);
    assert!(upload_data.blendshape_buffers.is_empty());
}
