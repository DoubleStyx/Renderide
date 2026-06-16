//! Blendshape sparse-buffer upload helpers.

use std::sync::Arc;

use crate::render_contract::{
    BLENDSHAPE_SPARSE_MIN_BUFFER_BYTES, blendshape_sparse_buffers_fit_device,
};
use crate::shared::MeshUploadData;

use super::super::super::layout::{
    BlendshapeFrameRange, BlendshapeFrameSpan, MeshBufferLayout, extract_blendshape_offsets,
};
use super::{MeshGpuUploadContext, try_create_buffer_init};

/// Sparse GPU buffers and CPU scatter ranges produced by `layout::extract_blendshape_offsets`.
pub(in crate::assets::mesh::gpu_mesh) struct BlendshapeBuffersUpload {
    /// Storage buffer of packed sparse channel deltas (padded when empty; see `BLENDSHAPE_SPARSE_MIN_BUFFER_BYTES`).
    pub sparse_buffer: Option<Arc<wgpu::Buffer>>,
    /// Copy of frame rows for CPU-side scatter dispatch.
    pub frame_ranges: Vec<BlendshapeFrameRange>,
    /// Per-shape spans into [`Self::frame_ranges`].
    pub shape_frame_spans: Vec<BlendshapeFrameSpan>,
    /// Logical blendshape slot count for weight indexing.
    pub num_blendshapes: u32,
    /// Whether any sparse row carries a nonzero position delta.
    pub has_position_deltas: bool,
    /// Whether any sparse row carries a nonzero normal delta.
    pub has_normal_deltas: bool,
    /// Whether any sparse row carries a nonzero tangent delta.
    pub has_tangent_deltas: bool,
}

/// Pads sparse CPU bytes to at least [`BLENDSHAPE_SPARSE_MIN_BUFFER_BYTES`] for `wgpu` buffers.
pub(in crate::assets::mesh::gpu_mesh) fn padded_sparse_bytes(sparse_deltas: &[u8]) -> Vec<u8> {
    let mut v = sparse_deltas.to_vec();
    let min = BLENDSHAPE_SPARSE_MIN_BUFFER_BYTES as usize;
    if v.len() < min {
        v.resize(min, 0);
    }
    v
}

/// Builds sparse / descriptor GPU buffers (or empty upload when blendshapes are disabled).
pub(in crate::assets::mesh::gpu_mesh) fn upload_blendshape_buffer(
    ctx: MeshGpuUploadContext<'_>,
    raw: &[u8],
    data: &MeshUploadData,
    layout: &MeshBufferLayout,
    use_blendshapes: bool,
    max_buf: u64,
) -> Option<BlendshapeBuffersUpload> {
    profiling::scope!("asset::mesh_upload_blendshape_buffers");
    if !use_blendshapes {
        return Some(BlendshapeBuffersUpload {
            sparse_buffer: None,
            frame_ranges: Vec::new(),
            shape_frame_spans: Vec::new(),
            num_blendshapes: 0,
            has_position_deltas: false,
            has_normal_deltas: false,
            has_tangent_deltas: false,
        });
    }
    let Some(pack) =
        extract_blendshape_offsets(raw, layout, &data.blendshape_buffers, data.vertex_count)
    else {
        return Some(BlendshapeBuffersUpload {
            sparse_buffer: None,
            frame_ranges: Vec::new(),
            shape_frame_spans: Vec::new(),
            num_blendshapes: 0,
            has_position_deltas: false,
            has_normal_deltas: false,
            has_tangent_deltas: false,
        });
    };

    let n_u32 = pack.num_blendshapes.max(0) as u32;
    if !blendshape_sparse_buffers_fit_device(
        &pack,
        max_buf,
        ctx.gpu_limits.wgpu.max_storage_buffer_binding_size,
    ) {
        logger::warn!(
            "mesh {}: blendshapes dropped (sparse bytes exceed buffer / binding limits)",
            data.asset_id
        );
        return Some(BlendshapeBuffersUpload {
            sparse_buffer: None,
            frame_ranges: Vec::new(),
            shape_frame_spans: Vec::new(),
            num_blendshapes: 0,
            has_position_deltas: false,
            has_normal_deltas: false,
            has_tangent_deltas: false,
        });
    }
    if pack.clamped_packed_deltas {
        logger::warn!(
            "mesh {}: blendshape normal/tangent deltas exceeded packed range and were clamped",
            data.asset_id
        );
    }

    let sparse_bytes = padded_sparse_bytes(&pack.sparse_deltas);

    let sparse_label = format!("mesh {} blendshape_sparse", data.asset_id);
    let sparse_buf = try_create_buffer_init(
        ctx,
        &wgpu::util::BufferInitDescriptor {
            label: Some(&sparse_label),
            contents: &sparse_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        },
    )?;
    crate::profiling::note_resource_churn!(Buffer, "assets::mesh_blendshape_sparse");

    Some(BlendshapeBuffersUpload {
        sparse_buffer: Some(Arc::new(sparse_buf)),
        frame_ranges: pack.frame_ranges,
        shape_frame_spans: pack.shape_frame_spans,
        num_blendshapes: n_u32,
        has_position_deltas: pack.has_position_deltas,
        has_normal_deltas: pack.has_normal_deltas,
        has_tangent_deltas: pack.has_tangent_deltas,
    })
}
