//! Free helpers for index format, submesh ranges, selective upload hints, and in-place stream checks.

use crate::shared::{
    BlendshapeBufferDescriptor, IndexBufferFormat, MeshUploadData, MeshUploadHintFlag,
    SubmeshBufferDescriptor, VertexAttributeType,
};

use super::gpu_mesh::GpuMesh;
use super::layout::{
    color_float4_stream_bytes, extract_float3_position_normal_as_vec4_streams,
    uv0_float2_stream_bytes, vertex_float2_stream_bytes, vertex_float4_stream_bytes,
};

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
        .filter(|s| {
            s.index_count > 0
                && (s.index_start as i64 + s.index_count as i64) <= index_count_u32 as i64
        })
        .map(|s| (s.index_start as u32, s.index_count as u32))
        .collect();
    if valid.is_empty() && index_count_u32 > 0 {
        vec![(0, index_count_u32)]
    } else {
        valid
    }
}

pub(super) fn derived_streams_compatible_for_in_place(
    gpu: &GpuMesh,
    vertex_slice: &[u8],
    data: &MeshUploadData,
    vc_usize: usize,
    vertex_stride_us: usize,
) -> bool {
    let pos_norm = extract_float3_position_normal_as_vec4_streams(
        vertex_slice,
        vc_usize,
        vertex_stride_us,
        &data.vertex_attributes,
    );
    match (
        &gpu.positions_buffer,
        &gpu.normals_buffer,
        pos_norm.as_ref(),
    ) {
        (Some(pb), Some(nb), Some((pvec, nvec))) => {
            if pb.size() != pvec.len() as u64 || nb.size() != nvec.len() as u64 {
                return false;
            }
        }
        (None, None, None) => {}
        _ => return false,
    }

    let uv_new = uv0_float2_stream_bytes(
        vertex_slice,
        vc_usize,
        vertex_stride_us,
        &data.vertex_attributes,
    );
    match (&gpu.uv0_buffer, uv_new.as_ref()) {
        (Some(b), Some(uv)) => {
            if b.size() != uv.len() as u64 {
                return false;
            }
        }
        (None, None) => {}
        _ => return false,
    }

    let color_new = color_float4_stream_bytes(
        vertex_slice,
        vc_usize,
        vertex_stride_us,
        &data.vertex_attributes,
    );
    match (&gpu.color_buffer, color_new.as_ref()) {
        (Some(b), Some(c)) => {
            if b.size() != c.len() as u64 {
                return false;
            }
        }
        (None, None) => {}
        _ => return false,
    }

    let tangent_new = vertex_float4_stream_bytes(
        vertex_slice,
        vc_usize,
        vertex_stride_us,
        &data.vertex_attributes,
        VertexAttributeType::Tangent,
        [1.0, 1.0, 1.0, 1.0],
    );
    if let Some(b) = &gpu.tangent_buffer {
        let Some(t) = tangent_new.as_ref() else {
            return false;
        };
        if b.size() != t.len() as u64 {
            return false;
        }
    }

    for (buffer, target) in [
        (&gpu.uv1_buffer, VertexAttributeType::UV1),
        (&gpu.uv2_buffer, VertexAttributeType::UV2),
        (&gpu.uv3_buffer, VertexAttributeType::UV3),
    ] {
        let uv_new = vertex_float2_stream_bytes(
            vertex_slice,
            vc_usize,
            vertex_stride_us,
            &data.vertex_attributes,
            target,
        );
        if let Some(b) = buffer {
            let Some(uv) = uv_new.as_ref() else {
                return false;
            };
            if b.size() != uv.len() as u64 {
                return false;
            }
        }
    }

    true
}

/// True when the host requests any selective (non–full-replace) upload region.
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
    fn blendshape_count_treats_negative_indices_as_zero() {
        let d = [BlendshapeBufferDescriptor {
            blendshape_index: -3,
            ..Default::default()
        }];
        assert_eq!(blendshape_descriptor_count(&d), 1);
    }
}
