//! Free helpers for index format, submesh ranges, selective upload hints, and in-place stream checks.

use crate::materials::RasterPrimitiveTopology;
use crate::shared::{
    BlendshapeBufferDescriptor, IndexBufferFormat, MeshUploadData, MeshUploadHintFlag,
    SubmeshBufferDescriptor, VertexAttributeDescriptor, VertexAttributeFormat, VertexAttributeType,
};

use super::gpu_mesh::GpuMesh;

pub(super) fn wgpu_index_format(f: IndexBufferFormat) -> wgpu::IndexFormat {
    match f {
        IndexBufferFormat::UInt16 => wgpu::IndexFormat::Uint16,
        IndexBufferFormat::UInt32 => wgpu::IndexFormat::Uint32,
    }
}

pub(super) fn validated_submesh_ranges(
    submeshes: &[SubmeshBufferDescriptor],
    index_count_u32: u32,
) -> Vec<(u32, u32)> {
    if submeshes.is_empty() {
        if index_count_u32 > 0 {
            return vec![(0, index_count_u32)];
        }
        return Vec::new();
    }
    let valid: Vec<(u32, u32)> = submeshes
        .iter()
        .filter(|s| valid_submesh_range(s, index_count_u32))
        .map(|s| (s.index_start as u32, s.index_count as u32))
        .collect();
    if valid.is_empty() && index_count_u32 > 0 {
        vec![(0, index_count_u32)]
    } else {
        valid
    }
}

/// Per-submesh primitive topologies, row-aligned with [`validated_submesh_ranges`].
///
/// Mirrors the same filter and fallback policy as `validated_submesh_ranges` so the two arrays
/// can be indexed by the same submesh index. When the host sends no submeshes (or every submesh
/// fails validation but `index_count_u32 > 0`), the synthesized full-range entry defaults to
/// [`RasterPrimitiveTopology::TriangleList`].
pub(super) fn validated_submesh_topologies(
    submeshes: &[SubmeshBufferDescriptor],
    index_count_u32: u32,
) -> Vec<RasterPrimitiveTopology> {
    if submeshes.is_empty() {
        if index_count_u32 > 0 {
            return vec![RasterPrimitiveTopology::default()];
        }
        return Vec::new();
    }
    let valid: Vec<RasterPrimitiveTopology> = submeshes
        .iter()
        .filter(|s| valid_submesh_range(s, index_count_u32))
        .map(|s| RasterPrimitiveTopology::from(s.topology))
        .collect();
    if valid.is_empty() && index_count_u32 > 0 {
        vec![RasterPrimitiveTopology::default()]
    } else {
        valid
    }
}

fn valid_submesh_range(s: &SubmeshBufferDescriptor, index_count_u32: u32) -> bool {
    s.index_count > 0
        && (i64::from(s.index_start) + i64::from(s.index_count)) <= i64::from(index_count_u32)
}

pub(super) fn derived_streams_compatible_for_in_place(
    gpu: &GpuMesh,
    vertex_slice: &[u8],
    data: &MeshUploadData,
    vc_usize: usize,
    vertex_stride_us: usize,
) -> bool {
    if vc_usize == 0 || vertex_stride_us == 0 {
        return gpu.positions_buffer.is_none()
            && gpu.normals_buffer.is_none()
            && gpu.uv0_buffer.is_none()
            && gpu.color_buffer.is_none()
            && gpu.tangent_buffer.is_none()
            && gpu.uv1_buffer.is_none()
            && gpu.uv2_buffer.is_none()
            && gpu.uv3_buffer.is_none();
    }
    let Some(needed_vertex_bytes) = vc_usize.checked_mul(vertex_stride_us) else {
        return false;
    };
    if vertex_slice.len() < needed_vertex_bytes {
        return false;
    }

    let pos_norm_bytes = has_supported_position_stream(&data.vertex_attributes)
        .then(|| (vc_usize as u64).saturating_mul(16));
    match (&gpu.positions_buffer, &gpu.normals_buffer, pos_norm_bytes) {
        (Some(pb), Some(nb), Some(bytes)) => {
            if pb.size() != bytes || nb.size() != bytes {
                return false;
            }
        }
        (None, None, None) => {}
        _ => return false,
    }

    let uv_bytes = (vc_usize as u64).saturating_mul(8);
    let vec4_bytes = (vc_usize as u64).saturating_mul(16);
    if !required_stream_size_matches(gpu.uv0_buffer.as_deref(), uv_bytes) {
        return false;
    }
    if !required_stream_size_matches(gpu.color_buffer.as_deref(), vec4_bytes) {
        return false;
    }
    if !optional_stream_size_matches(gpu.tangent_buffer.as_deref(), Some(vec4_bytes)) {
        return false;
    }
    for buffer in [&gpu.uv1_buffer, &gpu.uv2_buffer, &gpu.uv3_buffer] {
        if !optional_stream_size_matches(buffer.as_deref(), Some(uv_bytes)) {
            return false;
        }
    }
    true
}

fn has_supported_position_stream(attrs: &[VertexAttributeDescriptor]) -> bool {
    attrs.iter().any(|attr| {
        (attr.attribute as i16) == (VertexAttributeType::Position as i16)
            && attr.format == VertexAttributeFormat::Float32
            && attr.dimensions >= 3
    })
}

fn required_stream_size_matches(buffer: Option<&wgpu::Buffer>, expected: u64) -> bool {
    buffer.is_some_and(|buffer| buffer.size() == expected)
}

fn optional_stream_size_matches(buffer: Option<&wgpu::Buffer>, expected: Option<u64>) -> bool {
    match (buffer, expected) {
        (Some(buffer), Some(expected)) => buffer.size() == expected,
        (None, None) => true,
        (None, Some(_)) => true,
        (Some(_), None) => false,
    }
}

/// True when the host requests any selective (non-full-replace) upload region.
pub(crate) fn mesh_upload_hint_any_selective(h: MeshUploadHintFlag) -> bool {
    h.vertex_layout()
        || h.positions()
        || h.normals()
        || h.tangents()
        || h.colors()
        || h.uv0s()
        || h.uv1s()
        || h.uv2s()
        || h.uv3s()
        || h.uv4s()
        || h.uv5s()
        || h.uv6s()
        || h.uv7s()
        || h.geometry()
        || h.submesh_layout()
        || h.bone_weights()
        || h.bind_poses()
        || h.blendshapes()
}

/// True when the hint touches vertex attribute streams (positions, UVs, etc.).
pub(crate) fn mesh_upload_hint_touches_vertex_streams(h: MeshUploadHintFlag) -> bool {
    h.vertex_layout()
        || h.positions()
        || h.normals()
        || h.tangents()
        || h.colors()
        || h.uv0s()
        || h.uv1s()
        || h.uv2s()
        || h.uv3s()
        || h.uv4s()
        || h.uv5s()
        || h.uv6s()
        || h.uv7s()
}

pub(super) fn blendshape_descriptor_count(descs: &[BlendshapeBufferDescriptor]) -> u32 {
    descs
        .iter()
        .map(|d| d.blendshape_index.max(0) as u32 + 1)
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::SubmeshTopology;

    fn submesh(start: i32, count: i32) -> SubmeshBufferDescriptor {
        SubmeshBufferDescriptor {
            topology: SubmeshTopology::default(),
            index_start: start,
            index_count: count,
            ..Default::default()
        }
    }

    fn submesh_with_topology(
        start: i32,
        count: i32,
        topology: SubmeshTopology,
    ) -> SubmeshBufferDescriptor {
        SubmeshBufferDescriptor {
            topology,
            index_start: start,
            index_count: count,
            ..Default::default()
        }
    }

    #[test]
    fn empty_submeshes_with_zero_indices_yields_empty() {
        assert!(validated_submesh_ranges(&[], 0).is_empty());
    }

    #[test]
    fn empty_submeshes_with_indices_yields_single_full_range() {
        assert_eq!(validated_submesh_ranges(&[], 12), vec![(0, 12)]);
    }

    #[test]
    fn valid_submeshes_are_passed_through() {
        let s = [submesh(0, 6), submesh(6, 6)];
        assert_eq!(validated_submesh_ranges(&s, 12), vec![(0, 6), (6, 6)]);
    }

    #[test]
    fn zero_count_submeshes_are_filtered_out() {
        let s = [submesh(0, 0), submesh(0, 6)];
        assert_eq!(validated_submesh_ranges(&s, 6), vec![(0, 6)]);
    }

    #[test]
    fn out_of_range_submeshes_are_filtered_out() {
        let s = [submesh(0, 6), submesh(6, 10)];
        assert_eq!(validated_submesh_ranges(&s, 12), vec![(0, 6)]);
    }

    #[test]
    fn all_invalid_submeshes_fall_back_to_full_range() {
        let s = [submesh(100, 6)];
        assert_eq!(validated_submesh_ranges(&s, 12), vec![(0, 12)]);
    }

    #[test]
    fn all_invalid_with_no_indices_yields_empty() {
        let s = [submesh(100, 6)];
        assert!(validated_submesh_ranges(&s, 0).is_empty());
    }

    #[test]
    fn blendshape_count_empty_is_zero() {
        assert_eq!(blendshape_descriptor_count(&[]), 0);
    }

    #[test]
    fn blendshape_count_is_max_index_plus_one() {
        let d = [
            BlendshapeBufferDescriptor {
                blendshape_index: 0,
                ..Default::default()
            },
            BlendshapeBufferDescriptor {
                blendshape_index: 4,
                ..Default::default()
            },
            BlendshapeBufferDescriptor {
                blendshape_index: 2,
                ..Default::default()
            },
        ];
        assert_eq!(blendshape_descriptor_count(&d), 5);
    }

    #[test]
    fn submesh_topologies_empty_with_zero_indices_yields_empty() {
        assert!(validated_submesh_topologies(&[], 0).is_empty());
    }

    #[test]
    fn submesh_topologies_empty_with_indices_falls_back_to_triangle_list() {
        assert_eq!(
            validated_submesh_topologies(&[], 12),
            vec![RasterPrimitiveTopology::TriangleList],
        );
    }

    #[test]
    fn submesh_topologies_round_trip_mixed_topologies() {
        let s = [
            submesh_with_topology(0, 6, SubmeshTopology::Points),
            submesh_with_topology(6, 6, SubmeshTopology::Triangles),
        ];
        assert_eq!(
            validated_submesh_topologies(&s, 12),
            vec![
                RasterPrimitiveTopology::PointList,
                RasterPrimitiveTopology::TriangleList,
            ],
        );
    }

    #[test]
    fn submesh_topologies_filter_aligns_with_ranges() {
        let s = [
            submesh_with_topology(0, 0, SubmeshTopology::Points),
            submesh_with_topology(0, 6, SubmeshTopology::Triangles),
        ];
        assert_eq!(validated_submesh_ranges(&s, 6), vec![(0, 6)]);
        assert_eq!(
            validated_submesh_topologies(&s, 6),
            vec![RasterPrimitiveTopology::TriangleList],
        );
    }

    #[test]
    fn submesh_topologies_all_invalid_falls_back_to_triangle_list() {
        let s = [submesh_with_topology(100, 6, SubmeshTopology::Points)];
        assert_eq!(
            validated_submesh_topologies(&s, 12),
            vec![RasterPrimitiveTopology::TriangleList],
        );
    }

    #[test]
    fn blendshape_count_treats_negative_indices_as_zero() {
        let d = [BlendshapeBufferDescriptor {
            blendshape_index: -3,
            ..Default::default()
        }];
        assert_eq!(blendshape_descriptor_count(&d), 1);
    }
}
