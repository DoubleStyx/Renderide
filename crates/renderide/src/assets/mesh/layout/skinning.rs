//! Skinning buffer extraction helpers.

/// Splits the mesh tail `bone_weights` region into GPU storage buffers for the skinning shader:
/// `array<vec4<u32>>` joint indices and `array<vec4<f32>>` weights per vertex.
///
/// Supports either **4 influences** (`32 * vertex_count` bytes as `(f32 weight, i32 index)` tuples)
/// or **1 influence** (`8 * vertex_count` bytes).
pub fn split_bone_weights_tail_for_gpu(
    bone_weights_tail: &[u8],
    vertex_count: usize,
) -> Option<(Vec<u8>, Vec<u8>)> {
    if vertex_count == 0 {
        return None;
    }
    let four_inf = vertex_count * 32;
    let one_inf = vertex_count * 8;
    let span = if bone_weights_tail.len() >= four_inf {
        4usize
    } else if bone_weights_tail.len() >= one_inf {
        1usize
    } else {
        return None;
    };

    let mut idx_bytes = vec![0u8; vertex_count * 16];
    let mut wt_bytes = vec![0u8; vertex_count * 16];

    for v in 0..vertex_count {
        for k in 0..4 {
            let (w, j) = if k < span {
                let off = v * (span * 8) + k * 8;
                if off + 8 > bone_weights_tail.len() {
                    return None;
                }
                let w_raw = f32::from_le_bytes(bone_weights_tail[off..off + 4].try_into().ok()?);
                let j = i32::from_le_bytes(bone_weights_tail[off + 4..off + 8].try_into().ok()?);
                // Match legacy skinned VB build: unmapped bones must not contribute (index 0 only if weight > 0).
                if j < 0 {
                    (0.0f32, 0u32)
                } else {
                    (w_raw, j as u32)
                }
            } else {
                (0.0f32, 0u32)
            };
            let wb = v * 16 + k * 4;
            wt_bytes[wb..wb + 4].copy_from_slice(&w.to_le_bytes());
            idx_bytes[wb..wb + 4].copy_from_slice(&j.to_le_bytes());
        }
    }
    Some((idx_bytes, wt_bytes))
}
