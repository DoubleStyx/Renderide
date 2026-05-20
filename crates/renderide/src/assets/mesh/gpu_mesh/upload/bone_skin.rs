//! Bone-weight, bind-pose, and skinning upload helpers.

use std::sync::Arc;

use glam::Mat4;

use crate::shared::MeshUploadData;

use super::super::super::layout::{
    MeshBufferLayout, extract_bind_poses, split_bone_weights_tail_for_gpu,
};
use super::{MeshGpuUploadContext, try_create_buffer_init};

/// Aggregated bone/skin GPU state and skinning matrices.
pub(in crate::assets::mesh::gpu_mesh) struct BoneSkinUpload {
    /// Per-vertex bone-count storage buffer.
    pub bone_counts_buffer: Option<Arc<wgpu::Buffer>>,
    /// Per-vertex bone-index storage buffer.
    pub bone_indices_buffer: Option<Arc<wgpu::Buffer>>,
    /// Per-vertex bone-weight storage buffer.
    pub bone_weights_vec4_buffer: Option<Arc<wgpu::Buffer>>,
    /// Bind-pose storage buffer.
    pub bind_poses_buffer: Option<Arc<wgpu::Buffer>>,
    /// CPU skinning bind matrices derived from bind poses.
    pub skinning_bind_matrices: Vec<Mat4>,
}

fn upload_skeleton_bone_buffers(
    ctx: MeshGpuUploadContext<'_>,
    raw: &[u8],
    data: &MeshUploadData,
    layout: &MeshBufferLayout,
    vc_usize: usize,
) -> Option<BoneSkinUpload> {
    profiling::scope!("asset::mesh_upload_skeleton_buffers");
    let bp_raw = &raw[layout.bind_poses_start..layout.bind_poses_start + layout.bind_poses_length];
    let bind_poses_arr = extract_bind_poses(bp_raw, data.bone_count as usize)?;
    let bp_bytes: Vec<u8> = bind_poses_arr
        .iter()
        .flat_map(|m| bytemuck::bytes_of(m).iter().copied())
        .collect();
    let skinning: Vec<Mat4> = bind_poses_arr
        .iter()
        .map(Mat4::from_cols_array_2d)
        .collect();

    let bc = &raw[layout.bone_counts_start..layout.bone_counts_start + layout.bone_counts_length];
    let bw =
        &raw[layout.bone_weights_start..layout.bone_weights_start + layout.bone_weights_length];
    let (bi_buf, bw_buf) = if let Some((ib, wb)) = split_bone_weights_tail_for_gpu(bc, bw, vc_usize)
    {
        let bi = try_create_buffer_init(
            ctx,
            &wgpu::util::BufferInitDescriptor {
                label: Some(&format!("mesh {} bone_indices", data.asset_id)),
                contents: &ib,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            },
        )?;
        crate::profiling::note_resource_churn!(Buffer, "assets::mesh_bone_indices");
        let bwt = try_create_buffer_init(
            ctx,
            &wgpu::util::BufferInitDescriptor {
                label: Some(&format!("mesh {} bone_weights_vec4", data.asset_id)),
                contents: &wb,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            },
        )?;
        crate::profiling::note_resource_churn!(Buffer, "assets::mesh_bone_weights_vec4");
        (Some(Arc::new(bi)), Some(Arc::new(bwt)))
    } else {
        logger::warn!(
            "mesh {}: bone weight tail could not be repacked for GPU skinning",
            data.asset_id
        );
        (None, None)
    };

    let bc_buf = try_create_buffer_init(
        ctx,
        &wgpu::util::BufferInitDescriptor {
            label: Some(&format!("mesh {} bone_counts", data.asset_id)),
            contents: bc,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        },
    )?;
    crate::profiling::note_resource_churn!(Buffer, "assets::mesh_bone_counts");
    let bp_buf = try_create_buffer_init(
        ctx,
        &wgpu::util::BufferInitDescriptor {
            label: Some(&format!("mesh {} bind_poses", data.asset_id)),
            contents: &bp_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        },
    )?;
    crate::profiling::note_resource_churn!(Buffer, "assets::mesh_bind_poses");
    Some(BoneSkinUpload {
        bone_counts_buffer: Some(Arc::new(bc_buf)),
        bone_indices_buffer: bi_buf,
        bone_weights_vec4_buffer: bw_buf,
        bind_poses_buffer: Some(Arc::new(bp_buf)),
        skinning_bind_matrices: skinning,
    })
}

/// Bone indices/weights, bind poses, and skinning matrices for real skeleton paths.
///
/// Returns [`None`] when the real-skeleton bind-pose slice is invalid ([`extract_bind_poses`]).
pub(in crate::assets::mesh::gpu_mesh) fn upload_bone_and_skin_buffers(
    ctx: MeshGpuUploadContext<'_>,
    raw: &[u8],
    data: &MeshUploadData,
    layout: &MeshBufferLayout,
    vc_usize: usize,
) -> Option<BoneSkinUpload> {
    profiling::scope!("asset::mesh_upload_bone_skin_buffers");
    if data.bone_count > 0 {
        upload_skeleton_bone_buffers(ctx, raw, data, layout, vc_usize)
    } else {
        Some(BoneSkinUpload {
            bone_counts_buffer: None,
            bone_indices_buffer: None,
            bone_weights_vec4_buffer: None,
            bind_poses_buffer: None,
            skinning_bind_matrices: Vec::new(),
        })
    }
}
