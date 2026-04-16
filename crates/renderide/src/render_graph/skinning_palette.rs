//! CPU bone palette matching [`super::passes::mesh_deform`] skinning dispatch for culling parity.

use glam::Mat4;

use crate::scene::{RenderSpaceId, SceneCoordinator};
use crate::shared::RenderingContext;

/// Inputs for [`build_skinning_palette`].
pub struct SkinningPaletteParams<'a> {
    /// Scene graph and transforms for bone and SMR nodes.
    pub scene: &'a SceneCoordinator,
    /// Render space containing the skinned mesh.
    pub space_id: RenderSpaceId,
    /// Bind-pose inverse bind matrices from the mesh asset.
    pub skinning_bind_matrices: &'a [Mat4],
    /// Whether the mesh declares a skeleton rig.
    pub has_skeleton: bool,
    /// Per-bone transform indices (host order), or `-1` for bind-only.
    pub bone_transform_indices: &'a [i32],
    /// Skinned mesh renderer node id (`-1` when not applicable).
    pub smr_node_id: i32,
    /// Which rendering context (e.g. main vs mirror) to resolve transforms in.
    pub render_context: RenderingContext,
    /// Head/output matrix for VR / secondary views.
    pub head_output_transform: Mat4,
}

/// Builds the same `world_bone * skinning_bind_matrices[i]` palette as the skinning compute pass.
pub fn build_skinning_palette(params: SkinningPaletteParams<'_>) -> Option<Vec<Mat4>> {
    let SkinningPaletteParams {
        scene,
        space_id,
        skinning_bind_matrices,
        has_skeleton,
        bone_transform_indices,
        smr_node_id,
        render_context,
        head_output_transform,
    } = params;

    let bone_count = skinning_bind_matrices.len();
    if bone_count == 0 || !has_skeleton {
        return None;
    }

    let smr_world = (smr_node_id >= 0)
        .then(|| {
            scene.world_matrix_for_render_context(
                space_id,
                smr_node_id as usize,
                render_context,
                head_output_transform,
            )
        })
        .flatten()
        .unwrap_or(Mat4::IDENTITY);

    let mut out = Vec::with_capacity(bone_count);
    for (bi, bind_mat) in skinning_bind_matrices.iter().enumerate() {
        let tid = bone_transform_indices.get(bi).copied().unwrap_or(-1);
        let pal = if tid < 0 {
            smr_world
        } else {
            match scene.world_matrix_for_render_context(
                space_id,
                tid as usize,
                render_context,
                head_output_transform,
            ) {
                Some(world) => world * bind_mat,
                None => smr_world,
            }
        };
        out.push(pal);
    }
    Some(out)
}
