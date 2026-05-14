//! Unit tests for [`super::layout`] (host mesh buffer layout and stream extraction).

use glam::Mat4;

use super::layout::{
    BLENDSHAPE_PACKED_VECTOR_DELTA_RANGE, BLENDSHAPE_PACKED_VECTOR_SPARSE_ENTRY_SIZE,
    BLENDSHAPE_POSITION_SPARSE_ENTRY_SIZE, BlendshapeFrameRange, BlendshapeFrameSpan,
    WIDE_UV_VERTEX_STRIDE_BYTES, blendshape_deform_is_active, color_float4_stream_bytes,
    compute_index_count, compute_mesh_buffer_layout, compute_vertex_stride,
    extract_blendshape_offsets, extract_float3_position_normal_as_vec4_streams,
    index_bytes_per_element, raw_float4_stream_bytes, select_blendshape_frame_coefficients,
    split_bone_weights_tail_for_gpu, uv0_float2_stream_bytes, vertex_float2_stream_bytes,
    vertex_float4_stream_bytes, wide_uv_stream_bytes,
};
use crate::shared::{
    BlendshapeBufferDescriptor, BlendshapeDataFlags, IndexBufferFormat, SubmeshBufferDescriptor,
    SubmeshTopology, VertexAttributeDescriptor, VertexAttributeFormat, VertexAttributeType,
};

fn position_frame_range(
    shape_index: u32,
    frame_index: i32,
    frame_weight: f32,
) -> BlendshapeFrameRange {
    position_frame_range_at(shape_index, frame_index, frame_weight, 0)
}

fn position_frame_range_at(
    shape_index: u32,
    frame_index: i32,
    frame_weight: f32,
    first_word: u32,
) -> BlendshapeFrameRange {
    BlendshapeFrameRange {
        shape_index,
        frame_index,
        frame_weight,
        position_first_word: first_word,
        position_count: 1,
        normal_first_word: first_word + 4,
        normal_count: 0,
        tangent_first_word: first_word + 4,
        tangent_count: 0,
    }
}

fn unpack_snorm16_delta(bits: u32) -> f32 {
    let raw = bits & 0xffff;
    let signed = if raw & 0x8000 != 0 {
        raw as i32 - 65536
    } else {
        raw as i32
    };
    (signed as f32 / 32767.0).max(-1.0) * BLENDSHAPE_PACKED_VECTOR_DELTA_RANGE
}

fn packed_delta_x(bytes: &[u8], first_word: u32) -> f32 {
    let offset = first_word as usize * size_of::<u32>() + 4;
    let xy = u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("xy"));
    unpack_snorm16_delta(xy)
}

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

#[test]
fn position_stream_synthesizes_normals_when_normal_missing() {
    let attrs = [VertexAttributeDescriptor {
        attribute: VertexAttributeType::Position,
        format: VertexAttributeFormat::Float32,
        dimensions: 3,
    }];
    let mut raw = Vec::new();
    for value in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
        raw.extend_from_slice(&value.to_le_bytes());
    }

    let (pos, nrm) =
        extract_float3_position_normal_as_vec4_streams(&raw, 2, 12, &attrs).expect("streams");
    let pos0: [f32; 4] = bytemuck::pod_read_unaligned(&pos[..16]);
    let pos1: [f32; 4] = bytemuck::pod_read_unaligned(&pos[16..32]);
    let nrm0: [f32; 4] = bytemuck::pod_read_unaligned(&nrm[..16]);
    let nrm1: [f32; 4] = bytemuck::pod_read_unaligned(&nrm[16..32]);

    assert_eq!(pos0, [1.0, 2.0, 3.0, 1.0]);
    assert_eq!(pos1, [4.0, 5.0, 6.0, 1.0]);
    assert_eq!(nrm0, [0.0, 0.0, 1.0, 0.0]);
    assert_eq!(nrm1, [0.0, 0.0, 1.0, 0.0]);
}

#[test]
fn position_normal_stream_decodes_half_positions_and_signed_normals() {
    let attrs = [
        VertexAttributeDescriptor {
            attribute: VertexAttributeType::Position,
            format: VertexAttributeFormat::Half16,
            dimensions: 3,
        },
        VertexAttributeDescriptor {
            attribute: VertexAttributeType::Normal,
            format: VertexAttributeFormat::SInt8,
            dimensions: 3,
        },
    ];
    let mut raw = Vec::new();
    for value in [0x3c00u16, 0x4000, 0x4200] {
        raw.extend_from_slice(&value.to_le_bytes());
    }
    raw.extend_from_slice(&[0u8, 127u8, 0u8]);

    let (pos, nrm) =
        extract_float3_position_normal_as_vec4_streams(&raw, 1, 9, &attrs).expect("streams");
    let pos0: [f32; 4] = bytemuck::pod_read_unaligned(&pos[..16]);
    let nrm0: [f32; 4] = bytemuck::pod_read_unaligned(&nrm[..16]);

    assert_eq!(pos0, [1.0, 2.0, 3.0, 1.0]);
    assert_eq!(nrm0, [0.0, 1.0, 0.0, 0.0]);
}

#[test]
fn raw_normal_payload_stream_preserves_unorm_payload_values() {
    let attrs = [VertexAttributeDescriptor {
        attribute: VertexAttributeType::Normal,
        format: VertexAttributeFormat::UNorm8,
        dimensions: 3,
    }];
    let raw = [0u8, 128u8, 255u8];
    let out = raw_float4_stream_bytes(&raw, 1, 3, &attrs, VertexAttributeType::Normal, [0.0; 4])
        .expect("raw normal payload stream");
    let payload: [f32; 4] = bytemuck::pod_read_unaligned(&out[..16]);

    assert_eq!(payload[0], 0.0);
    assert!((payload[1] - (128.0 / 255.0)).abs() < 1e-6);
    assert_eq!(payload[2], 1.0);
    assert_eq!(payload[3], 0.0);
}

#[test]
fn split_bone_weights_four_influences_roundtrip() {
    let mut tail = Vec::new();
    for v in 0..2u8 {
        for k in 0..4u8 {
            let w = 0.25 + f32::from(v) * 0.01 + f32::from(k) * 0.01;
            let j = i32::from(k) + i32::from(v) * 10;
            tail.extend_from_slice(&w.to_le_bytes());
            tail.extend_from_slice(&j.to_le_bytes());
        }
    }
    let bone_counts = [4u8, 4u8];
    let (idx, wt) = split_bone_weights_tail_for_gpu(&bone_counts, &tail, 2).expect("split");
    let w0 = f32::from_le_bytes(wt[0..4].try_into().unwrap());
    let i0 = u32::from_le_bytes(idx[0..4].try_into().unwrap());
    assert!((w0 - (0.28 / 1.06)).abs() < 1e-5);
    assert_eq!(i0, 3);

    // Vertex 1 is sorted by descending weight, so k=3 is first.
    let w1_0 = f32::from_le_bytes(wt[16..20].try_into().unwrap());
    let i1_0 = u32::from_le_bytes(idx[16..20].try_into().unwrap());
    assert!((w1_0 - (0.29 / 1.10)).abs() < 1e-5);
    assert_eq!(i1_0, 13);
}

#[test]
fn split_bone_weights_negative_index_zeroes_weight() {
    let mut tail = Vec::new();
    tail.extend_from_slice(&0.5f32.to_le_bytes());
    tail.extend_from_slice(&(-1i32).to_le_bytes());
    let bone_counts = [1u8];
    let (idx, wt) = split_bone_weights_tail_for_gpu(&bone_counts, &tail, 1).expect("split");
    let w0 = f32::from_le_bytes(wt[0..4].try_into().unwrap());
    let i0 = u32::from_le_bytes(idx[0..4].try_into().unwrap());
    assert!((w0 - 0.0).abs() < 1e-5);
    assert_eq!(i0, 0u32);
}

#[test]
fn split_bone_weights_preserves_variable_counts_and_keeps_strongest_four() {
    let mut tail = Vec::new();
    for (w, j) in [
        (0.2f32, 2i32),
        (0.4, 4),
        (0.1, 1),
        (0.5, 5),
        (0.3, 3),
        (0.6, 6),
        (0.05, 7),
    ] {
        tail.extend_from_slice(&w.to_le_bytes());
        tail.extend_from_slice(&j.to_le_bytes());
    }
    let bone_counts = [2u8, 0u8, 5u8];
    let (idx, wt) = split_bone_weights_tail_for_gpu(&bone_counts, &tail, 3).expect("split");

    let v0_w0 = f32::from_le_bytes(wt[0..4].try_into().unwrap());
    let v0_i0 = u32::from_le_bytes(idx[0..4].try_into().unwrap());
    let v1_w0 = f32::from_le_bytes(wt[16..20].try_into().unwrap());
    let v2_i0 = u32::from_le_bytes(idx[32..36].try_into().unwrap());
    let v2_i3 = u32::from_le_bytes(idx[44..48].try_into().unwrap());

    assert!((v0_w0 - (0.4 / 0.6)).abs() < 1e-5);
    assert_eq!(v0_i0, 4);
    assert_eq!(v1_w0, 0.0);
    assert_eq!(v2_i0, 6);
    assert_eq!(v2_i3, 1);
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

#[test]
fn uv0_float2_zeros_when_missing() {
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
    let stride = 24usize;
    let verts = 2usize;
    let raw = vec![0u8; verts * stride];
    let out = uv0_float2_stream_bytes(&raw, verts, stride, &attrs).expect("uv stream");
    assert_eq!(out.len(), verts * 8);
    assert!(out.iter().all(|&b| b == 0));
}

#[test]
fn uv0_float2_preserves_unsigned_integer_values() {
    let attrs = [
        VertexAttributeDescriptor {
            attribute: VertexAttributeType::Position,
            format: VertexAttributeFormat::Float32,
            dimensions: 3,
        },
        VertexAttributeDescriptor {
            attribute: VertexAttributeType::UV0,
            format: VertexAttributeFormat::UInt8,
            dimensions: 2,
        },
    ];
    let mut raw = Vec::new();
    raw.extend_from_slice(&0.0f32.to_le_bytes());
    raw.extend_from_slice(&0.0f32.to_le_bytes());
    raw.extend_from_slice(&0.0f32.to_le_bytes());
    raw.extend_from_slice(&[3, 250]);

    let out = uv0_float2_stream_bytes(&raw, 1, 14, &attrs).expect("uv stream");
    let uv: [f32; 2] = bytemuck::pod_read_unaligned(&out[..8]);
    assert_eq!(uv, [3.0, 250.0]);
}

#[test]
fn vertex_float2_extracts_uv1_stream() {
    let attrs = [
        VertexAttributeDescriptor {
            attribute: VertexAttributeType::Position,
            format: VertexAttributeFormat::Float32,
            dimensions: 3,
        },
        VertexAttributeDescriptor {
            attribute: VertexAttributeType::UV1,
            format: VertexAttributeFormat::Float32,
            dimensions: 2,
        },
    ];
    let mut raw = Vec::new();
    raw.extend_from_slice(&0.0f32.to_le_bytes());
    raw.extend_from_slice(&0.0f32.to_le_bytes());
    raw.extend_from_slice(&0.0f32.to_le_bytes());
    raw.extend_from_slice(&1.25f32.to_le_bytes());
    raw.extend_from_slice(&2.5f32.to_le_bytes());

    let out = vertex_float2_stream_bytes(&raw, 1, 20, &attrs, VertexAttributeType::UV1)
        .expect("uv1 stream");
    let uv: [f32; 2] = bytemuck::pod_read_unaligned(&out[..8]);
    assert_eq!(uv, [1.25, 2.5]);
}

#[test]
fn vertex_float2_extracts_uv2_stream() {
    let attrs = [
        VertexAttributeDescriptor {
            attribute: VertexAttributeType::Position,
            format: VertexAttributeFormat::Float32,
            dimensions: 3,
        },
        VertexAttributeDescriptor {
            attribute: VertexAttributeType::UV2,
            format: VertexAttributeFormat::Float32,
            dimensions: 2,
        },
    ];
    let mut raw = Vec::new();
    raw.extend_from_slice(&0.0f32.to_le_bytes());
    raw.extend_from_slice(&0.0f32.to_le_bytes());
    raw.extend_from_slice(&0.0f32.to_le_bytes());
    raw.extend_from_slice(&3.75f32.to_le_bytes());
    raw.extend_from_slice(&4.5f32.to_le_bytes());

    let out = vertex_float2_stream_bytes(&raw, 1, 20, &attrs, VertexAttributeType::UV2)
        .expect("uv2 stream");
    let uv: [f32; 2] = bytemuck::pod_read_unaligned(&out[..8]);
    assert_eq!(uv, [3.75, 4.5]);
}

#[test]
fn vertex_float2_extracts_uv3_stream() {
    let attrs = [
        VertexAttributeDescriptor {
            attribute: VertexAttributeType::Position,
            format: VertexAttributeFormat::Float32,
            dimensions: 3,
        },
        VertexAttributeDescriptor {
            attribute: VertexAttributeType::UV3,
            format: VertexAttributeFormat::Float32,
            dimensions: 2,
        },
    ];
    let mut raw = Vec::new();
    raw.extend_from_slice(&0.0f32.to_le_bytes());
    raw.extend_from_slice(&0.0f32.to_le_bytes());
    raw.extend_from_slice(&0.0f32.to_le_bytes());
    raw.extend_from_slice(&5.25f32.to_le_bytes());
    raw.extend_from_slice(&6.125f32.to_le_bytes());

    let out = vertex_float2_stream_bytes(&raw, 1, 20, &attrs, VertexAttributeType::UV3)
        .expect("uv3 stream");
    let uv: [f32; 2] = bytemuck::pod_read_unaligned(&out[..8]);
    assert_eq!(uv, [5.25, 6.125]);
}

#[test]
fn wide_uv_stream_packs_all_channels_as_vec4_rows() {
    let attrs = [
        VertexAttributeDescriptor {
            attribute: VertexAttributeType::UV0,
            format: VertexAttributeFormat::Float32,
            dimensions: 4,
        },
        VertexAttributeDescriptor {
            attribute: VertexAttributeType::UV7,
            format: VertexAttributeFormat::Float32,
            dimensions: 3,
        },
    ];
    let mut raw = Vec::new();
    for value in [1.0f32, 2.0, 3.0, 4.0, 7.0, 8.0, 9.0] {
        raw.extend_from_slice(&value.to_le_bytes());
    }
    for value in [10.0f32, 20.0, 30.0, 40.0, 70.0, 80.0, 90.0] {
        raw.extend_from_slice(&value.to_le_bytes());
    }

    let out = wide_uv_stream_bytes(&raw, 2, 28, &attrs).expect("wide uv stream");
    assert_eq!(out.len(), 2 * WIDE_UV_VERTEX_STRIDE_BYTES);

    let uv0: [f32; 4] = bytemuck::pod_read_unaligned(&out[..16]);
    let missing_uv4_offset = 4 * 16;
    let missing_uv4: [f32; 4] =
        bytemuck::pod_read_unaligned(&out[missing_uv4_offset..missing_uv4_offset + 16]);
    let uv7_offset = 7 * 16;
    let uv7: [f32; 4] = bytemuck::pod_read_unaligned(&out[uv7_offset..uv7_offset + 16]);
    let second_uv0_offset = WIDE_UV_VERTEX_STRIDE_BYTES;
    let second_uv0: [f32; 4] =
        bytemuck::pod_read_unaligned(&out[second_uv0_offset..second_uv0_offset + 16]);

    assert_eq!(uv0, [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(missing_uv4, [0.0, 0.0, 0.0, 0.0]);
    assert_eq!(uv7, [7.0, 8.0, 9.0, 0.0]);
    assert_eq!(second_uv0, [10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn vertex_float4_defaults_when_tangent_missing() {
    let attrs = [VertexAttributeDescriptor {
        attribute: VertexAttributeType::Position,
        format: VertexAttributeFormat::Float32,
        dimensions: 3,
    }];
    let raw = vec![0u8; 12];
    let out = vertex_float4_stream_bytes(
        &raw,
        1,
        12,
        &attrs,
        VertexAttributeType::Tangent,
        [1.0, 0.0, 0.0, 1.0],
    )
    .expect("tangent stream");
    let tangent: [f32; 4] = bytemuck::pod_read_unaligned(&out[..16]);
    assert_eq!(tangent, [1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn color_stream_defaults_to_opaque_white_when_missing() {
    let attrs = [VertexAttributeDescriptor {
        attribute: VertexAttributeType::Position,
        format: VertexAttributeFormat::Float32,
        dimensions: 3,
    }];
    let raw = vec![0u8; 12];
    let out = color_float4_stream_bytes(&raw, 1, 12, &attrs).expect("color stream");
    let rgba: [f32; 4] = bytemuck::pod_read_unaligned(&out[..16]);
    assert_eq!(rgba, [1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn color_stream_decodes_unorm8_rgba() {
    let attrs = [VertexAttributeDescriptor {
        attribute: VertexAttributeType::Color,
        format: VertexAttributeFormat::UNorm8,
        dimensions: 4,
    }];
    let raw = vec![255u8, 128u8, 0u8, 64u8];
    let out = color_float4_stream_bytes(&raw, 1, 4, &attrs).expect("color stream");
    let rgba: [f32; 4] = bytemuck::pod_read_unaligned(&out[..16]);
    assert!((rgba[0] - 1.0).abs() < 1e-6);
    assert!((rgba[1] - (128.0 / 255.0)).abs() < 1e-6);
    assert!((rgba[2] - 0.0).abs() < 1e-6);
    assert!((rgba[3] - (64.0 / 255.0)).abs() < 1e-6);
}

#[test]
fn color_stream_decodes_uint8_rgba_as_normalized_color() {
    let attrs = [VertexAttributeDescriptor {
        attribute: VertexAttributeType::Color,
        format: VertexAttributeFormat::UInt8,
        dimensions: 4,
    }];
    let raw = vec![0u8, 64u8, 128u8, 255u8];
    let out = color_float4_stream_bytes(&raw, 1, 4, &attrs).expect("color stream");
    let rgba: [f32; 4] = bytemuck::pod_read_unaligned(&out[..16]);
    assert!((rgba[0] - 0.0).abs() < 1e-6);
    assert!((rgba[1] - (64.0 / 255.0)).abs() < 1e-6);
    assert!((rgba[2] - (128.0 / 255.0)).abs() < 1e-6);
    assert!((rgba[3] - 1.0).abs() < 1e-6);
}

#[test]
fn extract_blendshape_sparse_keeps_nonzero_position_rows_only() {
    let vertex_count = 2i32;
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
    let stride = compute_vertex_stride(&attrs);
    let sub = [SubmeshBufferDescriptor {
        topology: SubmeshTopology::default(),
        index_start: 0,
        index_count: 3,
        bounds: crate::shared::RenderBoundingBox::default(),
    }];
    let ic = compute_index_count(&sub);
    let blend = [BlendshapeBufferDescriptor {
        blendshape_index: 0,
        frame_index: 0,
        frame_weight: 1.0,
        data_flags: BlendshapeDataFlags(BlendshapeDataFlags::POSITIONS),
    }];
    let layout = compute_mesh_buffer_layout(
        stride,
        vertex_count,
        ic,
        index_bytes_per_element(IndexBufferFormat::UInt16),
        0,
        0,
        Some(&blend),
    )
    .expect("layout");
    let mut full = vec![0u8; layout.total_buffer_length];
    let off = layout.blendshape_data_start;
    let ax = 1.0f32.to_le_bytes();
    full[off..off + 4].copy_from_slice(&ax);
    let pack = extract_blendshape_offsets(&full, &layout, &blend, vertex_count).expect("pack");
    assert_eq!(pack.num_blendshapes, 1);
    assert_eq!(
        pack.shape_frame_spans[0],
        BlendshapeFrameSpan {
            first_frame: 0,
            frame_count: 1,
        }
    );
    assert_eq!(pack.frame_ranges[0].position_first_word, 0);
    assert_eq!(pack.frame_ranges[0].position_count, 1);
    assert_eq!(pack.frame_ranges[0].normal_count, 0);
    assert_eq!(pack.frame_ranges[0].tangent_count, 0);
    assert_eq!(
        pack.sparse_deltas.len(),
        BLENDSHAPE_POSITION_SPARSE_ENTRY_SIZE
    );
    assert!(pack.has_position_deltas);
    assert!(!pack.has_normal_deltas);
    assert!(!pack.has_tangent_deltas);
}

#[test]
fn extract_blendshape_sparse_keeps_normal_and_tangent_channels_separate() {
    let vertex_count = 1i32;
    let attrs = [VertexAttributeDescriptor {
        attribute: VertexAttributeType::Position,
        format: VertexAttributeFormat::Float32,
        dimensions: 3,
    }];
    let stride = compute_vertex_stride(&attrs);
    let blend = [BlendshapeBufferDescriptor {
        blendshape_index: 0,
        frame_index: 0,
        frame_weight: 1.0,
        data_flags: BlendshapeDataFlags(
            BlendshapeDataFlags::NORMALS | BlendshapeDataFlags::TANGETS,
        ),
    }];
    let layout =
        compute_mesh_buffer_layout(stride, vertex_count, 0, 2, 0, 0, Some(&blend)).expect("layout");
    let mut full = vec![0u8; layout.total_buffer_length];
    let off = layout.blendshape_data_start;
    full[off..off + 4].copy_from_slice(&0.25f32.to_le_bytes());
    full[off + 12..off + 16].copy_from_slice(&0.5f32.to_le_bytes());

    let pack = extract_blendshape_offsets(&full, &layout, &blend, vertex_count).expect("pack");

    assert_eq!(pack.frame_ranges[0].position_count, 0);
    assert_eq!(pack.frame_ranges[0].normal_first_word, 0);
    assert_eq!(pack.frame_ranges[0].normal_count, 1);
    assert_eq!(pack.frame_ranges[0].tangent_first_word, 3);
    assert_eq!(pack.frame_ranges[0].tangent_count, 1);
    assert_eq!(
        pack.sparse_deltas.len(),
        BLENDSHAPE_PACKED_VECTOR_SPARSE_ENTRY_SIZE * 2
    );
    assert!(!pack.has_position_deltas);
    assert!(pack.has_normal_deltas);
    assert!(pack.has_tangent_deltas);
    assert!((packed_delta_x(&pack.sparse_deltas, 0) - 0.25).abs() < 0.0001);
    assert!((packed_delta_x(&pack.sparse_deltas, 3) - 0.5).abs() < 0.0001);
}

#[test]
fn extract_blendshape_sparse_does_not_union_channels_on_different_vertices() {
    let vertex_count = 2i32;
    let attrs = [VertexAttributeDescriptor {
        attribute: VertexAttributeType::Position,
        format: VertexAttributeFormat::Float32,
        dimensions: 3,
    }];
    let stride = compute_vertex_stride(&attrs);
    let blend = [BlendshapeBufferDescriptor {
        blendshape_index: 0,
        frame_index: 0,
        frame_weight: 1.0,
        data_flags: BlendshapeDataFlags(
            BlendshapeDataFlags::POSITIONS | BlendshapeDataFlags::NORMALS,
        ),
    }];
    let layout =
        compute_mesh_buffer_layout(stride, vertex_count, 0, 2, 0, 0, Some(&blend)).expect("layout");
    let mut full = vec![0u8; layout.total_buffer_length];
    let off = layout.blendshape_data_start;
    full[off..off + 4].copy_from_slice(&1.0f32.to_le_bytes());
    let normals_off = off + vertex_count as usize * 12;
    full[normals_off + 12..normals_off + 16].copy_from_slice(&0.5f32.to_le_bytes());

    let pack = extract_blendshape_offsets(&full, &layout, &blend, vertex_count).expect("pack");

    assert_eq!(pack.frame_ranges[0].position_count, 1);
    assert_eq!(pack.frame_ranges[0].normal_count, 1);
    assert_eq!(pack.frame_ranges[0].tangent_count, 0);
    assert_eq!(
        pack.sparse_deltas.len(),
        BLENDSHAPE_POSITION_SPARSE_ENTRY_SIZE + BLENDSHAPE_PACKED_VECTOR_SPARSE_ENTRY_SIZE
    );
}

#[test]
fn extract_blendshape_sparse_all_channels_on_one_vertex_is_no_larger_than_old_union_row() {
    let vertex_count = 1i32;
    let attrs = [VertexAttributeDescriptor {
        attribute: VertexAttributeType::Position,
        format: VertexAttributeFormat::Float32,
        dimensions: 3,
    }];
    let stride = compute_vertex_stride(&attrs);
    let blend = [BlendshapeBufferDescriptor {
        blendshape_index: 0,
        frame_index: 0,
        frame_weight: 1.0,
        data_flags: BlendshapeDataFlags(
            BlendshapeDataFlags::POSITIONS
                | BlendshapeDataFlags::NORMALS
                | BlendshapeDataFlags::TANGETS,
        ),
    }];
    let layout =
        compute_mesh_buffer_layout(stride, vertex_count, 0, 2, 0, 0, Some(&blend)).expect("layout");
    let mut full = vec![0u8; layout.total_buffer_length];
    let off = layout.blendshape_data_start;
    full[off..off + 4].copy_from_slice(&1.0f32.to_le_bytes());
    full[off + 12..off + 16].copy_from_slice(&0.25f32.to_le_bytes());
    full[off + 24..off + 28].copy_from_slice(&0.5f32.to_le_bytes());

    let pack = extract_blendshape_offsets(&full, &layout, &blend, vertex_count).expect("pack");

    assert_eq!(pack.frame_ranges[0].position_count, 1);
    assert_eq!(pack.frame_ranges[0].normal_count, 1);
    assert_eq!(pack.frame_ranges[0].tangent_count, 1);
    assert_eq!(
        pack.sparse_deltas.len(),
        BLENDSHAPE_POSITION_SPARSE_ENTRY_SIZE + BLENDSHAPE_PACKED_VECTOR_SPARSE_ENTRY_SIZE * 2
    );
}

#[test]
fn extract_blendshape_packed_deltas_clamp_to_supported_range() {
    let vertex_count = 1i32;
    let attrs = [VertexAttributeDescriptor {
        attribute: VertexAttributeType::Position,
        format: VertexAttributeFormat::Float32,
        dimensions: 3,
    }];
    let stride = compute_vertex_stride(&attrs);
    let blend = [BlendshapeBufferDescriptor {
        blendshape_index: 0,
        frame_index: 0,
        frame_weight: 1.0,
        data_flags: BlendshapeDataFlags(BlendshapeDataFlags::NORMALS),
    }];
    let layout =
        compute_mesh_buffer_layout(stride, vertex_count, 0, 2, 0, 0, Some(&blend)).expect("layout");
    let mut full = vec![0u8; layout.total_buffer_length];
    let off = layout.blendshape_data_start;
    full[off..off + 4].copy_from_slice(&3.0f32.to_le_bytes());

    let pack = extract_blendshape_offsets(&full, &layout, &blend, vertex_count).expect("pack");

    assert!(pack.clamped_packed_deltas);
    assert!((packed_delta_x(&pack.sparse_deltas, 0) - 2.0).abs() < 0.0001);
}

#[test]
fn extract_blendshape_frame_ranges_follow_blendshape_indices_not_descriptor_order() {
    let vertex_count = 1i32;
    let attrs = [VertexAttributeDescriptor {
        attribute: VertexAttributeType::Position,
        format: VertexAttributeFormat::Float32,
        dimensions: 3,
    }];
    let stride = compute_vertex_stride(&attrs);
    let blend = [
        BlendshapeBufferDescriptor {
            blendshape_index: 2,
            frame_index: 0,
            frame_weight: 1.0,
            data_flags: BlendshapeDataFlags(BlendshapeDataFlags::POSITIONS),
        },
        BlendshapeBufferDescriptor {
            blendshape_index: 0,
            frame_index: 0,
            frame_weight: 1.0,
            data_flags: BlendshapeDataFlags(BlendshapeDataFlags::POSITIONS),
        },
    ];
    let layout =
        compute_mesh_buffer_layout(stride, vertex_count, 0, 2, 0, 0, Some(&blend)).expect("layout");
    let mut full = vec![0u8; layout.total_buffer_length];
    let first_descriptor_offset = layout.blendshape_data_start;
    let second_descriptor_offset = first_descriptor_offset + 12;
    full[first_descriptor_offset..first_descriptor_offset + 4]
        .copy_from_slice(&20.0f32.to_le_bytes());
    full[second_descriptor_offset..second_descriptor_offset + 4]
        .copy_from_slice(&10.0f32.to_le_bytes());

    let pack = extract_blendshape_offsets(&full, &layout, &blend, vertex_count).expect("pack");

    assert_eq!(pack.num_blendshapes, 3);
    assert_eq!(
        pack.shape_frame_spans,
        vec![
            BlendshapeFrameSpan {
                first_frame: 0,
                frame_count: 1,
            },
            BlendshapeFrameSpan {
                first_frame: 1,
                frame_count: 0,
            },
            BlendshapeFrameSpan {
                first_frame: 1,
                frame_count: 1,
            },
        ]
    );
    assert_eq!(pack.frame_ranges[0].shape_index, 0);
    assert_eq!(pack.frame_ranges[0].position_first_word, 0);
    assert_eq!(pack.frame_ranges[1].shape_index, 2);
    assert_eq!(pack.frame_ranges[1].position_first_word, 4);
    let first_dx = f32::from_le_bytes(pack.sparse_deltas[4..8].try_into().expect("dx"));
    let second_offset = BLENDSHAPE_POSITION_SPARSE_ENTRY_SIZE + 4;
    let second_dx = f32::from_le_bytes(
        pack.sparse_deltas[second_offset..second_offset + 4]
            .try_into()
            .expect("dx"),
    );
    assert_eq!(first_dx, 10.0);
    assert_eq!(second_dx, 20.0);
}

#[test]
fn extract_blendshape_two_frames_are_not_collapsed() {
    let vertex_count = 1i32;
    let attrs = [VertexAttributeDescriptor {
        attribute: VertexAttributeType::Position,
        format: VertexAttributeFormat::Float32,
        dimensions: 3,
    }];
    let stride = compute_vertex_stride(&attrs);
    let blend = [
        BlendshapeBufferDescriptor {
            blendshape_index: 0,
            frame_index: 0,
            frame_weight: 0.0,
            data_flags: BlendshapeDataFlags(BlendshapeDataFlags::POSITIONS),
        },
        BlendshapeBufferDescriptor {
            blendshape_index: 0,
            frame_index: 1,
            frame_weight: 100.0,
            data_flags: BlendshapeDataFlags(BlendshapeDataFlags::POSITIONS),
        },
    ];
    let layout =
        compute_mesh_buffer_layout(stride, vertex_count, 0, 2, 0, 0, Some(&blend)).expect("layout");
    let mut full = vec![0u8; layout.total_buffer_length];
    let first_descriptor_offset = layout.blendshape_data_start;
    let second_descriptor_offset = first_descriptor_offset + 12;
    full[first_descriptor_offset..first_descriptor_offset + 4]
        .copy_from_slice(&1.0f32.to_le_bytes());
    full[second_descriptor_offset..second_descriptor_offset + 4]
        .copy_from_slice(&2.0f32.to_le_bytes());

    let pack = extract_blendshape_offsets(&full, &layout, &blend, vertex_count).expect("pack");

    assert_eq!(pack.shape_frame_spans[0].frame_count, 2);
    assert_eq!(pack.frame_ranges[0].frame_weight, 0.0);
    assert_eq!(pack.frame_ranges[0].position_first_word, 0);
    assert_eq!(pack.frame_ranges[1].frame_weight, 100.0);
    assert_eq!(pack.frame_ranges[1].position_first_word, 4);
}

#[test]
fn extract_blendshape_duplicate_same_frame_is_skipped_deterministically() {
    let vertex_count = 1i32;
    let attrs = [VertexAttributeDescriptor {
        attribute: VertexAttributeType::Position,
        format: VertexAttributeFormat::Float32,
        dimensions: 3,
    }];
    let stride = compute_vertex_stride(&attrs);
    let blend = [
        BlendshapeBufferDescriptor {
            blendshape_index: 0,
            frame_index: 7,
            frame_weight: 100.0,
            data_flags: BlendshapeDataFlags(BlendshapeDataFlags::POSITIONS),
        },
        BlendshapeBufferDescriptor {
            blendshape_index: 0,
            frame_index: 7,
            frame_weight: 100.0,
            data_flags: BlendshapeDataFlags(BlendshapeDataFlags::POSITIONS),
        },
    ];
    let layout =
        compute_mesh_buffer_layout(stride, vertex_count, 0, 2, 0, 0, Some(&blend)).expect("layout");
    let mut full = vec![0u8; layout.total_buffer_length];
    full[layout.blendshape_data_start..layout.blendshape_data_start + 4]
        .copy_from_slice(&1.0f32.to_le_bytes());
    full[layout.blendshape_data_start + 12..layout.blendshape_data_start + 16]
        .copy_from_slice(&2.0f32.to_le_bytes());

    let pack = extract_blendshape_offsets(&full, &layout, &blend, vertex_count).expect("pack");

    assert_eq!(pack.shape_frame_spans[0].frame_count, 1);
    assert_eq!(pack.frame_ranges[0].position_count, 1);
    let dx = f32::from_le_bytes(pack.sparse_deltas[4..8].try_into().expect("dx"));
    assert_eq!(dx, 1.0);
}

#[test]
fn blendshape_coefficients_single_frame_scale_by_frame_weight() {
    let one_weight_frame = [position_frame_range(0, 0, 1.0)];
    let spans = [BlendshapeFrameSpan {
        first_frame: 0,
        frame_count: 1,
    }];

    let one_weight = select_blendshape_frame_coefficients(0, 0.25, &spans, &one_weight_frame);

    assert_eq!(one_weight[0].expect("coefficient").effective_weight, 0.25);

    let hundred_weight_frame = [position_frame_range(0, 0, 100.0)];

    let selected = select_blendshape_frame_coefficients(0, 25.0, &spans, &hundred_weight_frame);

    assert_eq!(selected[0].expect("coefficient").frame_range_index, 0);
    assert_eq!(selected[0].expect("coefficient").effective_weight, 0.25);
    assert!(selected[1].is_none());
}

#[test]
fn blendshape_coefficients_two_frames_interpolate_and_extrapolate() {
    let frames = [
        position_frame_range_at(0, 0, 0.0, 0),
        position_frame_range_at(0, 1, 100.0, 4),
    ];
    let spans = [BlendshapeFrameSpan {
        first_frame: 0,
        frame_count: 2,
    }];

    let mid = select_blendshape_frame_coefficients(0, 50.0, &spans, &frames);
    assert_eq!(mid[0].expect("lo").effective_weight, 0.5);
    assert_eq!(mid[1].expect("hi").effective_weight, 0.5);

    let over = select_blendshape_frame_coefficients(0, 150.0, &spans, &frames);
    assert_eq!(over[0].expect("lo").effective_weight, -0.5);
    assert_eq!(over[1].expect("hi").effective_weight, 1.5);

    let negative = select_blendshape_frame_coefficients(0, -25.0, &spans, &frames);
    assert_eq!(negative[0].expect("lo").effective_weight, 1.25);
    assert_eq!(negative[1].expect("hi").effective_weight, -0.25);
}

#[test]
fn blendshape_active_predicate_uses_frame_coefficients() {
    let frames = [position_frame_range(0, 0, 100.0)];
    let spans = [BlendshapeFrameSpan {
        first_frame: 0,
        frame_count: 1,
    }];

    assert!(!blendshape_deform_is_active(1, &spans, &frames, &[0.0]));
    assert!(!blendshape_deform_is_active(
        1,
        &spans,
        &frames,
        &[f32::NAN]
    ));
    assert!(!blendshape_deform_is_active(1, &spans, &frames, &[]));
    assert!(blendshape_deform_is_active(1, &spans, &frames, &[25.0]));
}
