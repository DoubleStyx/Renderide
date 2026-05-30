//! Resident GPU byte accounting for uploaded mesh buffers.

use std::sync::Arc;

use super::DerivedStreams;
use super::bone_skin::BoneSkinUpload;

/// Sums the sizes of optional GPU buffers.
pub(in crate::assets::mesh::gpu_mesh) fn sum_optional_buffer_bytes(
    buffers: &[Option<&Arc<wgpu::Buffer>>],
) -> u64 {
    buffers
        .iter()
        .filter_map(|o| o.as_ref().map(|b| b.size()))
        .sum()
}

/// Sums VRAM for all optional mesh buffers plus fixed vertex/index sizes.
pub(in crate::assets::mesh::gpu_mesh) fn resident_bytes_for_mesh_upload(
    core_vb: &wgpu::Buffer,
    core_ib: &wgpu::Buffer,
    derived: &DerivedStreams,
    bone_skin: &BoneSkinUpload,
    blend_sparse: Option<&Arc<wgpu::Buffer>>,
) -> u64 {
    let mut n = core_vb.size() + core_ib.size();
    n += sum_optional_buffer_bytes(&[
        bone_skin.bone_counts_buffer.as_ref(),
        bone_skin.bone_indices_buffer.as_ref(),
        bone_skin.bone_weights_vec4_buffer.as_ref(),
        bone_skin.bone_influence_offsets_buffer.as_ref(),
        bone_skin.bone_influences_buffer.as_ref(),
        bone_skin.bind_poses_buffer.as_ref(),
        derived.positions_buffer.as_ref(),
        derived.normals_buffer.as_ref(),
        derived.uv0_buffer.as_ref(),
        derived.color_buffer.as_ref(),
        derived.tangent_buffer.as_ref(),
        derived.raw_tangent_buffer.as_ref(),
        derived.uv1_buffer.as_ref(),
        derived.uv2_buffer.as_ref(),
        derived.uv3_buffer.as_ref(),
        derived.wide_uv_buffer.as_ref(),
    ]);
    if let Some(b) = blend_sparse {
        n += b.size();
    }
    n
}
