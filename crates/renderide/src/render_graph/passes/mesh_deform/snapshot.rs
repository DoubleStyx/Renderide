//! GPU mesh handles and predicates for deform dispatch eligibility.

use std::sync::Arc;

use glam::Mat4;

use crate::assets::mesh::GpuMesh;

/// GPU buffer handles + metadata copied from [`crate::assets::mesh::GpuMesh`] so we can hold
/// deform state without borrowing the mesh pool across preprocess/scratch access.
///
/// Deformed positions/normals and blend temp live in [`crate::backend::mesh_deform::GpuSkinCache`]
/// per instance, not on the mesh.
pub(super) struct MeshDeformSnapshot {
    pub(super) vertex_count: u32,
    pub(super) num_blendshapes: u32,
    pub(super) has_skeleton: bool,
    pub(super) positions_buffer: Option<Arc<wgpu::Buffer>>,
    pub(super) normals_buffer: Option<Arc<wgpu::Buffer>>,
    pub(super) blendshape_sparse_buffer: Option<Arc<wgpu::Buffer>>,
    /// Per-shape `(first_entry, entry_count)` into [`Self::blendshape_sparse_buffer`] (scatter dispatch).
    pub(super) blendshape_sparse_ranges: Vec<(u32, u32)>,
    pub(super) bone_indices_buffer: Option<Arc<wgpu::Buffer>>,
    pub(super) bone_weights_vec4_buffer: Option<Arc<wgpu::Buffer>>,
    pub(super) skinning_bind_matrices: Vec<Mat4>,
}

impl MeshDeformSnapshot {
    /// Copies GPU resources from `m`. When `clone_skinning_bind_matrices` is `false`, bind matrices
    /// are omitted (blendshape-only path never reads them).
    pub(super) fn from_mesh(m: &GpuMesh, clone_skinning_bind_matrices: bool) -> Self {
        Self {
            vertex_count: m.vertex_count,
            num_blendshapes: m.num_blendshapes,
            has_skeleton: m.has_skeleton,
            positions_buffer: m.positions_buffer.clone(),
            normals_buffer: m.normals_buffer.clone(),
            blendshape_sparse_buffer: m.blendshape_sparse_buffer.clone(),
            blendshape_sparse_ranges: m.blendshape_sparse_ranges.clone(),
            bone_indices_buffer: m.bone_indices_buffer.clone(),
            bone_weights_vec4_buffer: m.bone_weights_vec4_buffer.clone(),
            skinning_bind_matrices: if clone_skinning_bind_matrices {
                m.skinning_bind_matrices.clone()
            } else {
                Vec::new()
            },
        }
    }
}

/// Returns whether blendshape compute should run for `m` (parity with deform encode).
pub(super) fn deform_needs_blend_mesh(m: &GpuMesh) -> bool {
    m.num_blendshapes > 0
        && m.blendshape_sparse_buffer.is_some()
        && m.blendshape_sparse_ranges.len() == m.num_blendshapes as usize
}

/// Returns whether skinning compute should run for `m` (parity with deform encode).
pub(super) fn deform_needs_skin_mesh(m: &GpuMesh, bone_transform_indices: Option<&[i32]>) -> bool {
    bone_transform_indices.is_some()
        && m.has_skeleton
        && m.normals_buffer.is_some()
        && m.bone_indices_buffer.is_some()
        && m.bone_weights_vec4_buffer.is_some()
        && !m.skinning_bind_matrices.is_empty()
}

/// Returns `true` when deform encoding would run blend and/or skin dispatches (not an early no-op).
pub(super) fn gpu_mesh_needs_deform_dispatch(
    m: &GpuMesh,
    bone_transform_indices: Option<&[i32]>,
) -> bool {
    if m.positions_buffer.is_none() || m.vertex_count == 0 {
        return false;
    }
    deform_needs_blend_mesh(m) || deform_needs_skin_mesh(m, bone_transform_indices)
}

/// Snapshot variant of [`deform_needs_blend_mesh`].
pub(super) fn deform_needs_blend_snapshot(mesh: &MeshDeformSnapshot) -> bool {
    mesh.num_blendshapes > 0
        && mesh.blendshape_sparse_buffer.is_some()
        && mesh.blendshape_sparse_ranges.len() == mesh.num_blendshapes as usize
}

/// Snapshot variant of [`deform_needs_skin_mesh`].
pub(super) fn deform_needs_skin_snapshot(
    mesh: &MeshDeformSnapshot,
    bone_transform_indices: Option<&[i32]>,
) -> bool {
    bone_transform_indices.is_some()
        && mesh.has_skeleton
        && mesh.normals_buffer.is_some()
        && mesh.bone_indices_buffer.is_some()
        && mesh.bone_weights_vec4_buffer.is_some()
        && !mesh.skinning_bind_matrices.is_empty()
}
