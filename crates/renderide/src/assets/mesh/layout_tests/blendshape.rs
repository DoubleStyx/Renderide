use super::super::layout::{
    BLENDSHAPE_PACKED_VECTOR_DELTA_RANGE, BLENDSHAPE_PACKED_VECTOR_SPARSE_ENTRY_SIZE,
    BLENDSHAPE_POSITION_SPARSE_ENTRY_SIZE, BlendshapeFrameRange, BlendshapeFrameSpan,
    blendshape_deform_is_active, compute_index_count, compute_mesh_buffer_layout,
    compute_vertex_stride, extract_blendshape_offsets, index_bytes_per_element,
    select_blendshape_frame_coefficients,
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
