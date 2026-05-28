use super::super::layout::split_bone_weights_tail_for_gpu;

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
