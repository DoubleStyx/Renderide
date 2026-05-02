//! Sparse blendshape buffer size checks and scatter dispatch chunking for [`wgpu::Limits`].

use crate::assets::mesh::{BLENDSHAPE_SPARSE_ENTRY_SIZE, BlendshapeGpuPack};

/// Minimum storage buffer size used when a mesh has blendshapes but zero sparse bytes (padding).
pub const BLENDSHAPE_SPARSE_MIN_BUFFER_BYTES: u64 = 16;

/// Returns `false` when sparse or frame-descriptor payloads cannot exist on the device or be bound
/// as a single storage read (typical WebGPU path).
pub fn blendshape_sparse_buffers_fit_device(
    pack: &BlendshapeGpuPack,
    max_buffer_size: u64,
    max_storage_buffer_binding_size: u64,
) -> bool {
    let sparse_len = pack.sparse_deltas.len().max(BLENDSHAPE_SPARSE_ENTRY_SIZE);
    let sparse_u64 = sparse_len as u64;
    let desc_len = pack.shape_descriptor_bytes.len();
    let desc_u64 = desc_len as u64;
    if sparse_u64 > max_buffer_size || desc_u64 > max_buffer_size {
        return false;
    }
    if sparse_u64 > max_storage_buffer_binding_size || desc_u64 > max_storage_buffer_binding_size {
        return false;
    }
    true
}

/// Plans `(sparse_base, sparse_count)` sub-ranges (global entry indices) so each dispatch stays
/// within `max_workgroups_per_dim x 64` threads (one thread per sparse entry).
pub fn plan_blendshape_scatter_chunks(
    first_entry: u32,
    entry_count: u32,
    max_workgroups_per_dim: u32,
) -> Vec<(u32, u32)> {
    if entry_count == 0 {
        return Vec::new();
    }
    let max_entries = max_workgroups_per_dim.saturating_mul(64);
    if max_entries == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut offset = 0u32;
    while offset < entry_count {
        let chunk = (entry_count - offset).min(max_entries);
        out.push((first_entry.saturating_add(offset), chunk));
        offset = offset.saturating_add(chunk);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::assets::mesh::{
        BLENDSHAPE_SHAPE_DESCRIPTOR_SIZE, BLENDSHAPE_SPARSE_ENTRY_SIZE, BlendshapeFrameRange,
        BlendshapeFrameSpan,
    };

    #[test]
    fn scatter_chunks_cover_all_entries() {
        let first = 10u32;
        let n = 500u32;
        let max_wg = 4u32;
        let chunks = plan_blendshape_scatter_chunks(first, n, max_wg);
        let sum: u32 = chunks.iter().map(|(_, c)| c).sum();
        assert_eq!(sum, n);
        assert_eq!(chunks.first().copied(), Some((10, 256)));
        assert_eq!(chunks.last().copied(), Some((266, 244)));
    }

    #[test]
    fn sparse_fit_accepts_tiny_pack() {
        let pack = BlendshapeGpuPack {
            sparse_deltas: vec![0u8; BLENDSHAPE_SPARSE_ENTRY_SIZE],
            shape_descriptor_bytes: vec![0u8; BLENDSHAPE_SHAPE_DESCRIPTOR_SIZE],
            frame_ranges: vec![BlendshapeFrameRange {
                shape_index: 0,
                frame_index: 0,
                frame_weight: 1.0,
                first_entry: 0,
                entry_count: 1,
            }],
            shape_frame_spans: vec![BlendshapeFrameSpan {
                first_frame: 0,
                frame_count: 1,
            }],
            num_blendshapes: 1,
        };
        assert!(blendshape_sparse_buffers_fit_device(&pack, 1024, 1024));
    }
}
