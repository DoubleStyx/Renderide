use super::super::layout::split_bone_weights_tail_for_gpu;

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn read_f32(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
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
    let streams = split_bone_weights_tail_for_gpu(&bone_counts, &tail, 2, 20).expect("split");
    let w0 = read_f32(&streams.bone_weights_vec4, 0);
    let i0 = read_u32(&streams.bone_indices_vec4, 0);
    assert!((w0 - (0.28 / 1.06)).abs() < 1e-5);
    assert_eq!(i0, 3);

    // Vertex 1 is sorted by descending weight, so k=3 is first.
    let w1_0 = read_f32(&streams.bone_weights_vec4, 16);
    let i1_0 = read_u32(&streams.bone_indices_vec4, 16);
    assert!((w1_0 - (0.29 / 1.10)).abs() < 1e-5);
    assert_eq!(i1_0, 13);
}

#[test]
fn split_bone_weights_negative_index_zeroes_weight() {
    let mut tail = Vec::new();
    tail.extend_from_slice(&0.5f32.to_le_bytes());
    tail.extend_from_slice(&(-1i32).to_le_bytes());
    let bone_counts = [1u8];
    let streams = split_bone_weights_tail_for_gpu(&bone_counts, &tail, 1, 1).expect("split");
    let w0 = read_f32(&streams.bone_weights_vec4, 0);
    let i0 = read_u32(&streams.bone_indices_vec4, 0);
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
    let streams = split_bone_weights_tail_for_gpu(&bone_counts, &tail, 3, 8).expect("split");

    let v0_w0 = read_f32(&streams.bone_weights_vec4, 0);
    let v0_i0 = read_u32(&streams.bone_indices_vec4, 0);
    let v1_w0 = read_f32(&streams.bone_weights_vec4, 16);
    let v2_i0 = read_u32(&streams.bone_indices_vec4, 32);
    let v2_i3 = read_u32(&streams.bone_indices_vec4, 44);

    assert!((v0_w0 - (0.4 / 0.6)).abs() < 1e-5);
    assert_eq!(v0_i0, 4);
    assert_eq!(v1_w0, 0.0);
    assert_eq!(v2_i0, 6);
    assert_eq!(v2_i3, 1);
}

#[test]
fn split_bone_weights_preserves_unlimited_influence_stream() {
    let mut tail = Vec::new();
    for (w, j) in [(0.2f32, 2i32), (0.0, 9), (0.4, 4), (0.1, -1), (0.6, 6)] {
        tail.extend_from_slice(&w.to_le_bytes());
        tail.extend_from_slice(&j.to_le_bytes());
    }
    let bone_counts = [2u8, 3u8];
    let streams = split_bone_weights_tail_for_gpu(&bone_counts, &tail, 2, 7).expect("split");

    assert_eq!(read_u32(&streams.influence_offsets, 0), 0);
    assert_eq!(read_u32(&streams.influence_offsets, 4), 1);
    assert_eq!(read_u32(&streams.influence_offsets, 8), 3);
    assert_eq!(read_u32(&streams.influences, 0), 2);
    assert!((read_f32(&streams.influences, 4) - 0.2).abs() < 1e-5);
    assert_eq!(read_u32(&streams.influences, 8), 4);
    assert!((read_f32(&streams.influences, 12) - 0.4).abs() < 1e-5);
    assert_eq!(read_u32(&streams.influences, 16), 6);
    assert!((read_f32(&streams.influences, 20) - 0.6).abs() < 1e-5);
}

#[test]
fn split_bone_weights_skips_invalid_influences() {
    let mut tail = Vec::new();
    for (w, j) in [
        (0.5f32, 1i32),
        (0.3, 3),
        (f32::INFINITY, 2),
        (-0.2, 0),
        (0.0, 2),
    ] {
        tail.extend_from_slice(&w.to_le_bytes());
        tail.extend_from_slice(&j.to_le_bytes());
    }
    let bone_counts = [5u8];
    let streams = split_bone_weights_tail_for_gpu(&bone_counts, &tail, 1, 3).expect("split");

    assert_eq!(read_u32(&streams.influence_offsets, 0), 0);
    assert_eq!(read_u32(&streams.influence_offsets, 4), 1);
    assert_eq!(read_u32(&streams.influences, 0), 1);
    assert!((read_f32(&streams.influences, 4) - 0.5).abs() < 1e-5);
    assert!((read_f32(&streams.bone_weights_vec4, 0) - 1.0).abs() < 1e-5);
    assert_eq!(read_u32(&streams.bone_indices_vec4, 0), 1);
}

#[test]
fn split_bone_weights_rejects_mismatched_declared_tail_length() {
    let mut one_influence = Vec::new();
    one_influence.extend_from_slice(&0.5f32.to_le_bytes());
    one_influence.extend_from_slice(&1i32.to_le_bytes());

    assert!(split_bone_weights_tail_for_gpu(&[2], &one_influence, 1, 2).is_none());

    let mut two_influences = one_influence.clone();
    two_influences.extend_from_slice(&0.25f32.to_le_bytes());
    two_influences.extend_from_slice(&0i32.to_le_bytes());

    assert!(split_bone_weights_tail_for_gpu(&[1], &two_influences, 1, 2).is_none());
}
