//! Skinning buffer extraction helpers.

/// GPU-ready bone influence streams built from the host compact variable-length stream.
pub(in crate::assets::mesh) struct BoneWeightGpuStreams {
    /// Per-vertex top-four bone indices as `vec4<u32>`.
    pub bone_indices_vec4: Vec<u8>,
    /// Per-vertex normalized top-four bone weights as `vec4<f32>`.
    pub bone_weights_vec4: Vec<u8>,
    /// Per-vertex prefix offsets into [`Self::influences`], plus one trailing sentinel.
    pub influence_offsets: Vec<u8>,
    /// Raw valid influences as `(u32 bone_index, f32 weight)` storage entries.
    pub influences: Vec<u8>,
}

/// Splits the mesh tail `bone_counts` and `bone_weights` regions into GPU storage buffers for the
/// skinning shader.
///
/// The host stores a compact variable-length stream: one byte count per vertex followed by that many
/// `(f32 weight, i32 index)` tuples. The renderer preserves the four strongest in-range influences
/// for fixed-width skinning modes and also preserves every in-range influence for unlimited
/// skinning.
pub(in crate::assets::mesh) fn split_bone_weights_tail_for_gpu(
    bone_counts: &[u8],
    bone_weights_tail: &[u8],
    vertex_count: usize,
    bone_count: usize,
) -> Option<BoneWeightGpuStreams> {
    if vertex_count == 0 {
        return None;
    }
    if bone_counts.len() < vertex_count {
        return None;
    }

    let mut idx_bytes = vec![0u8; vertex_count * 16];
    let mut wt_bytes = vec![0u8; vertex_count * 16];
    let mut influence_offsets = Vec::with_capacity((vertex_count + 1) * 4);
    let mut influences_bytes = Vec::new();
    let mut tail_offset = 0usize;
    let mut influence_count = 0u32;
    influence_offsets.extend_from_slice(&influence_count.to_le_bytes());

    for (v, vertex_bone_count) in bone_counts.iter().copied().enumerate().take(vertex_count) {
        let mut influences = [BoneInfluence::ZERO; 4];
        for _ in 0..usize::from(vertex_bone_count) {
            let end = tail_offset.checked_add(8)?;
            let src = bone_weights_tail.get(tail_offset..end)?;
            tail_offset = end;
            let weight = f32::from_le_bytes(src[0..4].try_into().ok()?);
            let index = i32::from_le_bytes(src[4..8].try_into().ok()?);
            if valid_bone_influence(weight, index, bone_count) {
                let influence = BoneInfluence {
                    weight,
                    index: index as u32,
                };
                insert_influence(&mut influences, influence);
                influences_bytes.extend_from_slice(&influence.index.to_le_bytes());
                influences_bytes.extend_from_slice(&influence.weight.to_le_bytes());
                influence_count = influence_count.checked_add(1)?;
            }
        }
        influence_offsets.extend_from_slice(&influence_count.to_le_bytes());
        let weight_sum = influences
            .iter()
            .fold(0.0f32, |sum, influence| sum + influence.weight);
        for (k, influence) in influences.iter().enumerate() {
            let w = if weight_sum > 1.0e-6 {
                influence.weight / weight_sum
            } else {
                0.0
            };
            let wb = v * 16 + k * 4;
            wt_bytes[wb..wb + 4].copy_from_slice(&w.to_le_bytes());
            idx_bytes[wb..wb + 4].copy_from_slice(&influence.index.to_le_bytes());
        }
    }
    if tail_offset != bone_weights_tail.len() {
        return None;
    }
    Some(BoneWeightGpuStreams {
        bone_indices_vec4: idx_bytes,
        bone_weights_vec4: wt_bytes,
        influence_offsets,
        influences: influences_bytes,
    })
}

fn valid_bone_influence(weight: f32, index: i32, bone_count: usize) -> bool {
    weight.is_finite() && weight > 0.0 && index >= 0 && (index as usize) < bone_count
}

#[derive(Clone, Copy)]
struct BoneInfluence {
    weight: f32,
    index: u32,
}

impl BoneInfluence {
    const ZERO: Self = Self {
        weight: 0.0,
        index: 0,
    };
}

fn insert_influence(influences: &mut [BoneInfluence; 4], candidate: BoneInfluence) {
    let mut insert_at = influences.len();
    for (i, current) in influences.iter().enumerate() {
        if candidate.weight > current.weight {
            insert_at = i;
            break;
        }
    }
    if insert_at == influences.len() {
        return;
    }
    for i in (insert_at + 1..influences.len()).rev() {
        influences[i] = influences[i - 1];
    }
    influences[insert_at] = candidate;
}
