//! Layout validation and [`try_upload_mesh_from_raw`] entry point (uses [`super::gpu_mesh::GpuMesh`]).

use crate::gpu::GpuLimits;
use crate::shared::MeshUploadData;

use super::gpu_mesh::GpuMesh;
use super::gpu_mesh_fingerprint::mesh_layout_fingerprint;
use super::layout::{
    MeshBufferLayout, compute_index_count, compute_mesh_buffer_layout, compute_vertex_stride,
    index_bytes_per_element,
};

/// Computes [`MeshBufferLayout`] from [`MeshUploadData`] and validates bone region lengths.
pub fn compute_and_validate_mesh_layout(data: &MeshUploadData) -> Option<MeshBufferLayout> {
    profiling::scope!("asset::mesh_compute_layout");
    if data.buffer.length <= 0 {
        return None;
    }
    let vertex_stride = compute_vertex_stride(&data.vertex_attributes);
    if vertex_stride <= 0 {
        logger::error!("mesh {}: invalid vertex stride", data.asset_id);
        return None;
    }
    let index_count = compute_index_count(&data.submeshes);
    let index_bytes = index_bytes_per_element(data.index_buffer_format);
    let layout = match compute_mesh_buffer_layout(
        vertex_stride,
        data.vertex_count,
        index_count,
        index_bytes,
        data.bone_count,
        data.bone_weight_count,
        Some(&data.blendshape_buffers),
    ) {
        Ok(l) => l,
        Err(e) => {
            logger::error!("mesh {}: layout error: {}", data.asset_id, e);
            return None;
        }
    };

    let expected_bone_weights_len = (data.bone_weight_count.max(0) * 8) as usize;
    let expected_bind_poses_len = (data.bone_count.max(0) * 64) as usize;
    if layout.bone_weights_length != expected_bone_weights_len {
        logger::error!("mesh {}: bone_weights layout mismatch", data.asset_id);
        return None;
    }
    if layout.bind_poses_length != expected_bind_poses_len {
        logger::error!("mesh {}: bind poses layout mismatch", data.asset_id);
        return None;
    }
    Some(layout)
}

/// Builds layout and uploads; returns [`GpuMesh`] if validation and GPU creation succeed.
///
/// When `queue` and `existing` refer to the resident [`GpuMesh`] for `data.asset_id`, and topology
/// matches, **reuses** existing `wgpu::Buffer` allocations and uses [`wgpu::Queue::write_buffer`]
/// instead of allocating new buffers with `create_buffer_init`.
pub fn try_upload_mesh_from_raw(
    device: &wgpu::Device,
    gpu_limits: &GpuLimits,
    queue: Option<&wgpu::Queue>,
    raw: &[u8],
    data: &MeshUploadData,
    existing: Option<GpuMesh>,
    layout: &MeshBufferLayout,
) -> Option<GpuMesh> {
    profiling::scope!("asset::mesh_upload_raw");
    if raw.len() < layout.total_buffer_length {
        logger::error!(
            "mesh {}: raw too short (need {}, got {})",
            data.asset_id,
            layout.total_buffer_length,
            raw.len()
        );
        return None;
    }

    let trace_enabled = logger::enabled(logger::LogLevel::Trace);
    // Compute the layout fingerprint only when the trace log will actually consume it. The
    // fingerprint hashes every vertex attribute and submesh, so recomputing it on every mesh
    // upload at non-trace log levels was pure waste.
    let layout_fp = if trace_enabled {
        mesh_layout_fingerprint(data, layout)
    } else {
        0
    };
    let hint = data.upload_hint.flags;

    if let (Some(queue), Some(existing)) = (queue, existing) {
        profiling::scope!("asset::mesh_in_place_compatibility");
        if existing.compatible_for_in_place_update(data, layout, raw) {
            profiling::scope!("asset::mesh_in_place_upload");
            if let Some(mesh) = existing.write_in_place(queue, raw, data, layout, hint) {
                if trace_enabled {
                    logger::trace!(
                        "mesh {}: in-place upload (layout_fp={:#x})",
                        data.asset_id,
                        layout_fp
                    );
                }
                return Some(mesh);
            }
        }
    }

    if trace_enabled {
        logger::trace!(
            "mesh {}: full GPU buffer upload (layout_fp={:#x})",
            data.asset_id,
            layout_fp
        );
    }
    GpuMesh::upload(device, gpu_limits, raw, data, layout)
}

#[cfg(test)]
mod tests {
    use crate::shared::MeshUploadHintFlag;

    use super::super::gpu_mesh_hints::{
        mesh_upload_hint_any_selective, mesh_upload_hint_touches_vertex_streams,
    };

    #[test]
    fn mesh_upload_hint_any_selective_false_when_empty() {
        assert!(!mesh_upload_hint_any_selective(MeshUploadHintFlag(0)));
    }

    #[test]
    fn mesh_upload_hint_any_selective_true_for_geometry() {
        assert!(mesh_upload_hint_any_selective(MeshUploadHintFlag(
            MeshUploadHintFlag::GEOMETRY
        )));
    }

    #[test]
    fn mesh_upload_hint_touches_vertex_streams_for_positions() {
        assert!(mesh_upload_hint_touches_vertex_streams(MeshUploadHintFlag(
            MeshUploadHintFlag::POSITIONS
        )));
        assert!(!mesh_upload_hint_touches_vertex_streams(
            MeshUploadHintFlag(MeshUploadHintFlag::GEOMETRY)
        ));
    }
}
