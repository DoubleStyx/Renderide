use super::super::layout::{
    WIDE_UV_HIGH_VERTEX_STRIDE_BYTES, WIDE_UV_LOW_VERTEX_STRIDE_BYTES, color_float4_stream_bytes,
    extract_float3_position_normal_as_vec4_streams, raw_float4_stream_bytes,
    uv0_float2_stream_bytes, vertex_float2_stream_bytes, vertex_float4_stream_bytes,
    wide_high_uv_stream_bytes, wide_low_uv_stream_bytes,
};
use super::{
    push_f32, read_f32x2_stream, read_f32x4_stream, read_wide_high_uv_stream,
    read_wide_low_uv_stream,
};
use crate::shared::{VertexAttributeDescriptor, VertexAttributeFormat, VertexAttributeType};

fn float_attr(attribute: VertexAttributeType, dimensions: i32) -> VertexAttributeDescriptor {
    VertexAttributeDescriptor {
        attribute,
        format: VertexAttributeFormat::Float32,
        dimensions,
    }
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
fn large_position_normal_stream_matches_expected() {
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
    let verts = 1_100usize;
    let stride = 24usize;
    let mut raw = Vec::with_capacity(verts * stride);
    for i in 0..verts {
        let base = i as f32;
        push_f32(&mut raw, base);
        push_f32(&mut raw, base + 0.25);
        push_f32(&mut raw, base + 0.5);
        push_f32(&mut raw, 0.0);
        push_f32(&mut raw, 1.0);
        push_f32(&mut raw, 0.0);
    }

    let (pos, nrm) = extract_float3_position_normal_as_vec4_streams(&raw, verts, stride, &attrs)
        .expect("streams");

    for vertex in [0usize, 513, 1_099] {
        let base = vertex as f32;
        assert_eq!(
            read_f32x4_stream(&pos, vertex),
            [base, base + 0.25, base + 0.5, 1.0]
        );
        assert_eq!(read_f32x4_stream(&nrm, vertex), [0.0, 1.0, 0.0, 0.0]);
    }
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
fn vertex_float2_prefers_requested_uv_over_fallback() {
    let attrs = [
        float_attr(VertexAttributeType::UV0, 2),
        float_attr(VertexAttributeType::UV1, 2),
        float_attr(VertexAttributeType::UV2, 2),
    ];
    let mut raw = Vec::new();
    for value in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
        push_f32(&mut raw, value);
    }

    let out = vertex_float2_stream_bytes(&raw, 1, 24, &attrs, VertexAttributeType::UV2)
        .expect("uv2 stream");
    assert_eq!(read_f32x2_stream(&out, 0), [5.0, 6.0]);
}

#[test]
fn vertex_float2_falls_back_from_uv2_to_uv1_then_uv0() {
    let attrs = [
        float_attr(VertexAttributeType::UV0, 2),
        float_attr(VertexAttributeType::UV1, 2),
    ];
    let mut raw = Vec::new();
    for value in [1.0f32, 2.0, 3.0, 4.0] {
        push_f32(&mut raw, value);
    }

    let out = vertex_float2_stream_bytes(&raw, 1, 16, &attrs, VertexAttributeType::UV2)
        .expect("uv2 stream");
    assert_eq!(read_f32x2_stream(&out, 0), [3.0, 4.0]);

    let attrs = [float_attr(VertexAttributeType::UV0, 2)];
    let mut raw = Vec::new();
    for value in [7.0f32, 8.0] {
        push_f32(&mut raw, value);
    }

    let out = vertex_float2_stream_bytes(&raw, 1, 8, &attrs, VertexAttributeType::UV2)
        .expect("uv2 stream");
    assert_eq!(read_f32x2_stream(&out, 0), [7.0, 8.0]);
}

#[test]
fn vertex_float2_falls_back_from_uv3_to_highest_available_lower_uv() {
    let attrs = [
        float_attr(VertexAttributeType::UV0, 2),
        float_attr(VertexAttributeType::UV2, 2),
    ];
    let mut raw = Vec::new();
    for value in [1.0f32, 2.0, 5.0, 6.0] {
        push_f32(&mut raw, value);
    }

    let out = vertex_float2_stream_bytes(&raw, 1, 16, &attrs, VertexAttributeType::UV3)
        .expect("uv3 stream");
    assert_eq!(read_f32x2_stream(&out, 0), [5.0, 6.0]);
}

#[test]
fn vertex_float2_skips_unsupported_uv_fallback_attributes() {
    let attrs = [
        float_attr(VertexAttributeType::UV2, 1),
        float_attr(VertexAttributeType::UV1, 2),
        float_attr(VertexAttributeType::UV0, 2),
    ];
    let mut raw = Vec::new();
    for value in [9.0f32, 3.0, 4.0, 1.0, 2.0] {
        push_f32(&mut raw, value);
    }

    let out = vertex_float2_stream_bytes(&raw, 1, 20, &attrs, VertexAttributeType::UV2)
        .expect("uv2 stream");
    assert_eq!(read_f32x2_stream(&out, 0), [3.0, 4.0]);
}

#[test]
fn vertex_float2_zeros_when_no_uv_fallback_is_available() {
    let attrs = [float_attr(VertexAttributeType::Position, 3)];
    let raw = vec![0u8; 12];

    let out = vertex_float2_stream_bytes(&raw, 1, 12, &attrs, VertexAttributeType::UV3)
        .expect("uv3 stream");
    assert_eq!(out.len(), 8);
    assert!(out.iter().all(|&b| b == 0));
}

#[test]
fn wide_uv_pages_pack_channels_as_vec4_rows() {
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

    let low = wide_low_uv_stream_bytes(&raw, 2, 28, &attrs).expect("wide low uv stream");
    let high = wide_high_uv_stream_bytes(&raw, 2, 28, &attrs).expect("wide high uv stream");
    assert_eq!(low.len(), 2 * WIDE_UV_LOW_VERTEX_STRIDE_BYTES);
    assert_eq!(high.len(), 2 * WIDE_UV_HIGH_VERTEX_STRIDE_BYTES);

    let uv0 = read_wide_low_uv_stream(&low, 0, 0);
    let missing_uv4 = read_wide_high_uv_stream(&high, 0, 4);
    let uv7 = read_wide_high_uv_stream(&high, 0, 7);
    let second_uv0 = read_wide_low_uv_stream(&low, 1, 0);

    assert_eq!(uv0, [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(missing_uv4, [0.0, 0.0, 0.0, 0.0]);
    assert_eq!(uv7, [7.0, 8.0, 9.0, 0.0]);
    assert_eq!(second_uv0, [10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn large_wide_uv_pages_match_expected() {
    let attrs = [
        VertexAttributeDescriptor {
            attribute: VertexAttributeType::UV0,
            format: VertexAttributeFormat::Float32,
            dimensions: 4,
        },
        VertexAttributeDescriptor {
            attribute: VertexAttributeType::UV5,
            format: VertexAttributeFormat::Float32,
            dimensions: 3,
        },
        VertexAttributeDescriptor {
            attribute: VertexAttributeType::UV7,
            format: VertexAttributeFormat::Float32,
            dimensions: 2,
        },
    ];
    let verts = 1_100usize;
    let stride = 36usize;
    let mut raw = Vec::with_capacity(verts * stride);
    for i in 0..verts {
        let base = i as f32;
        push_f32(&mut raw, base);
        push_f32(&mut raw, base + 1.0);
        push_f32(&mut raw, base + 2.0);
        push_f32(&mut raw, base + 3.0);
        push_f32(&mut raw, base + 50.0);
        push_f32(&mut raw, base + 51.0);
        push_f32(&mut raw, base + 52.0);
        push_f32(&mut raw, base + 70.0);
        push_f32(&mut raw, base + 71.0);
    }

    let low = wide_low_uv_stream_bytes(&raw, verts, stride, &attrs).expect("wide low uv stream");
    let high = wide_high_uv_stream_bytes(&raw, verts, stride, &attrs).expect("wide high uv stream");
    assert_eq!(low.len(), verts * WIDE_UV_LOW_VERTEX_STRIDE_BYTES);
    assert_eq!(high.len(), verts * WIDE_UV_HIGH_VERTEX_STRIDE_BYTES);

    for vertex in [0usize, 777, 1_099] {
        let base = vertex as f32;
        assert_eq!(
            read_wide_low_uv_stream(&low, vertex, 0),
            [base, base + 1.0, base + 2.0, base + 3.0]
        );
        assert_eq!(read_wide_low_uv_stream(&low, vertex, 3), [0.0; 4]);
        assert_eq!(
            read_wide_high_uv_stream(&high, vertex, 5),
            [base + 50.0, base + 51.0, base + 52.0, 0.0]
        );
        assert_eq!(
            read_wide_high_uv_stream(&high, vertex, 7),
            [base + 70.0, base + 71.0, 0.0, 0.0]
        );
    }
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
fn large_uv_color_and_raw_float4_streams_match_expected() {
    let attrs = [
        VertexAttributeDescriptor {
            attribute: VertexAttributeType::UV0,
            format: VertexAttributeFormat::Float32,
            dimensions: 2,
        },
        VertexAttributeDescriptor {
            attribute: VertexAttributeType::Color,
            format: VertexAttributeFormat::UInt8,
            dimensions: 4,
        },
        VertexAttributeDescriptor {
            attribute: VertexAttributeType::Tangent,
            format: VertexAttributeFormat::Float32,
            dimensions: 4,
        },
    ];
    let verts = 1_100usize;
    let stride = 28usize;
    let mut raw = Vec::with_capacity(verts * stride);
    for i in 0..verts {
        let base = i as f32;
        push_f32(&mut raw, base * 0.5);
        push_f32(&mut raw, base + 2.0);
        raw.extend_from_slice(&[
            (i % 256) as u8,
            ((i + 32) % 256) as u8,
            ((i + 64) % 256) as u8,
            255,
        ]);
        push_f32(&mut raw, base + 10.0);
        push_f32(&mut raw, base + 11.0);
        push_f32(&mut raw, base + 12.0);
        push_f32(&mut raw, base + 13.0);
    }

    let uv = uv0_float2_stream_bytes(&raw, verts, stride, &attrs).expect("uv stream");
    let color = color_float4_stream_bytes(&raw, verts, stride, &attrs).expect("color stream");
    let tangent = raw_float4_stream_bytes(
        &raw,
        verts,
        stride,
        &attrs,
        VertexAttributeType::Tangent,
        [0.0; 4],
    )
    .expect("tangent stream");

    for vertex in [0usize, 777, 1_099] {
        let base = vertex as f32;
        assert_eq!(read_f32x2_stream(&uv, vertex), [base * 0.5, base + 2.0]);
        assert_eq!(
            read_f32x4_stream(&tangent, vertex),
            [base + 10.0, base + 11.0, base + 12.0, base + 13.0]
        );
        let rgba = read_f32x4_stream(&color, vertex);
        assert!((rgba[0] - ((vertex % 256) as f32 / 255.0)).abs() < 1e-6);
        assert!((rgba[1] - (((vertex + 32) % 256) as f32 / 255.0)).abs() < 1e-6);
        assert!((rgba[2] - (((vertex + 64) % 256) as f32 / 255.0)).abs() < 1e-6);
        assert_eq!(rgba[3], 1.0);
    }
}
