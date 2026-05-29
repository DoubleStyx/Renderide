//! Mesh layout (host `MeshBuffer` contract) and GPU upload.

mod gpu_mesh;
mod layout;
#[cfg(test)]
mod layout_tests;

pub use gpu_mesh::GpuMesh;
pub(crate) use gpu_mesh::{
    MeshBufferUploadSink, MeshDerivedStreamDemand, MeshDerivedStreamMask, MeshGpuUploadContext,
    PreparedDerivedStreams, compute_and_validate_mesh_layout, mesh_upload_input_fingerprint,
    prepare_derived_stream_bytes, try_upload_mesh_from_raw,
};
pub use layout::{
    BLENDSHAPE_PACKED_VECTOR_SPARSE_ENTRY_WORDS, BLENDSHAPE_POSITION_SPARSE_ENTRY_SIZE,
    BLENDSHAPE_POSITION_SPARSE_ENTRY_WORDS, BlendshapeFrameRange, BlendshapeFrameSpan,
    BlendshapeGpuPack, MeshBufferLayout, blendshape_deform_is_active,
    select_blendshape_frame_coefficients,
};
