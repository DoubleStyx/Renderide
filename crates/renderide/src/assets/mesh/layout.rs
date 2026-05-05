//! Mesh packed-buffer layout matching `Renderite.Shared.MeshBuffer.ComputeBufferLayout`.
//!
//! Regions: vertices -> indices -> bone_counts -> bone_weights -> bind_poses -> blendshape_data.

mod blendshape;
mod buffer_layout;
mod skinning;
mod streams;

pub use blendshape::{
    BLENDSHAPE_PACKED_VECTOR_DELTA_RANGE, BLENDSHAPE_PACKED_VECTOR_SPARSE_ENTRY_SIZE,
    BLENDSHAPE_PACKED_VECTOR_SPARSE_ENTRY_WORDS, BLENDSHAPE_POSITION_SPARSE_ENTRY_SIZE,
    BLENDSHAPE_POSITION_SPARSE_ENTRY_WORDS, BlendshapeFrameCoefficient, BlendshapeFrameRange,
    BlendshapeFrameSpan, BlendshapeGpuPack, blendshape_deform_is_active,
    extract_blendshape_offsets, select_blendshape_frame_coefficients,
};
pub use buffer_layout::{
    MAX_BUFFER_SIZE, MeshBufferLayout, compute_index_count, compute_mesh_buffer_layout,
    compute_vertex_stride, extract_bind_poses, index_bytes_per_element,
};
pub use skinning::split_bone_weights_tail_for_gpu;
#[cfg(test)]
pub use streams::vertex_float4_stream_bytes;
pub use streams::{
    attribute_offset_and_size, color_float4_stream_bytes,
    extract_float3_position_normal_as_vec4_streams, uv0_float2_stream_bytes,
    vertex_float2_stream_bytes,
};
