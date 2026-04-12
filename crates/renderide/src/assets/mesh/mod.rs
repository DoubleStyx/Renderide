//! Mesh layout (host `MeshBuffer` contract) and GPU upload.

mod gpu_mesh;
mod gpu_mesh_fingerprint;
mod gpu_mesh_validation;
mod layout;

pub use gpu_mesh::GpuMesh;
pub use gpu_mesh_fingerprint::{mesh_layout_fingerprint, mesh_upload_input_fingerprint};
pub use gpu_mesh_validation::{compute_and_validate_mesh_layout, try_upload_mesh_from_raw};
pub use layout::{
    attribute_offset_and_size, color_float4_stream_bytes, compute_index_count,
    compute_mesh_buffer_layout, compute_vertex_stride, extract_bind_poses,
    extract_blendshape_offsets, extract_float3_position_normal_as_vec4_streams,
    index_bytes_per_element, split_bone_weights_tail_for_gpu,
    synthetic_bone_data_for_blendshape_only, uv0_float2_stream_bytes, MeshBufferLayout,
    BLENDSHAPE_OFFSET_GPU_STRIDE, MAX_BUFFER_SIZE,
};
