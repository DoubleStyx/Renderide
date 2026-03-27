//! Build interleaved skinned vertex buffers from mesh bone weights and tangents.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::assets::{self, MeshAsset};
use crate::shared::{VertexAttributeFormat, VertexAttributeType};

use super::decode::read_vec3;
use super::types::{VertexPosNormal, VertexSkinned};

/// Layout-compatible with Renderite.Shared.BoneWeight (weight at offset 0, boneIndex at offset 4).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BoneWeightPod {
    weight: f32,
    bone_index: i32,
}

/// Default tangent when mesh has no tangent attribute.
const DEFAULT_TANGENT: [f32; 3] = [1.0, 0.0, 0.0];

/// Builds a GPU vertex buffer of [`VertexSkinned`] from base vertices and bone streams.
pub(super) fn build_skinned_vertices(
    device: &wgpu::Device,
    mesh: &MeshAsset,
    vertex_stride: usize,
    base_vertices: &[VertexPosNormal],
) -> Option<wgpu::Buffer> {
    let bone_counts = mesh.bone_counts.as_ref()?;
    let bone_weights = mesh.bone_weights.as_ref()?;
    if bone_counts.len() != base_vertices.len() {
        return None;
    }
    let tangent_info =
        assets::attribute_offset_size_format(&mesh.vertex_attributes, VertexAttributeType::tangent);
    let (tangent_off, tangent_size, tangent_format) =
        tangent_info.unwrap_or((0, 0, VertexAttributeFormat::float32));

    let vc = base_vertices.len();
    let mut skinned = Vec::with_capacity(vc);
    let mut weight_offset = 0;
    for (i, v) in base_vertices.iter().enumerate() {
        let tangent = if tangent_size > 0 {
            let base = i * vertex_stride;
            read_vec3(&mesh.vertex_data, base, tangent_off, tangent_format)
                .unwrap_or(DEFAULT_TANGENT)
        } else {
            DEFAULT_TANGENT
        };

        let n_raw = bone_counts.get(i).copied().unwrap_or(0) as usize;
        let n = n_raw.min(4);
        let mut indices = [0i32; 4];
        let mut weights = [0.0f32; 4];
        for j in 0..n {
            if weight_offset + 8 <= bone_weights.len() {
                let w: BoneWeightPod =
                    bytemuck::pod_read_unaligned(&bone_weights[weight_offset..weight_offset + 8]);
                if w.bone_index < 0 {
                    // Invalid/unmapped bone: zero the weight so it has no effect.
                    indices[j] = 0;
                    weights[j] = 0.0;
                } else {
                    indices[j] = w.bone_index.clamp(0, 255);
                    weights[j] = w.weight;
                }
                weight_offset += 8;
            }
        }
        // Consume any excess entries (beyond the 4-influence GPU limit) to keep
        // weight_offset aligned for subsequent vertices.
        for _ in n..n_raw {
            if weight_offset + 8 <= bone_weights.len() {
                weight_offset += 8;
            }
        }
        skinned.push(VertexSkinned {
            position: v.position,
            normal: v.normal,
            tangent,
            bone_indices: indices,
            bone_weights: weights,
        });
    }
    Some(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh vertex buffer (skinned)"),
            contents: bytemuck::cast_slice(&skinned),
            usage: wgpu::BufferUsages::VERTEX,
        }),
    )
}
