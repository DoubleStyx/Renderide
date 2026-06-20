//! World-matrix and front-face resolution shared by the scene-walk and prepared-renderable
//! collection paths.

use glam::Mat4;

use crate::materials::RasterFrontFace;
#[cfg(test)]
use crate::scene::SkinnedMeshRenderer;
use crate::scene::{RenderSpaceId, SceneTransformRead};

use super::DrawCollectionInputs;

/// Resolves the draw's world matrix when the selected vertex stream is still local-space.
#[inline]
pub(super) fn world_matrix_for_local_vertex_stream(
    ctx: &DrawCollectionInputs<'_>,
    space_id: RenderSpaceId,
    node_id: i32,
    is_overlay: bool,
) -> Option<Mat4> {
    if node_id < 0 {
        return None;
    }
    if is_overlay {
        return SceneTransformRead::overlay_layer_model_matrix_for_context(
            ctx.scene_assets.scene,
            space_id,
            node_id as usize,
            ctx.view.render_context,
        )
        .or_else(|| {
            SceneTransformRead::world_matrix_for_context(
                ctx.scene_assets.scene,
                space_id,
                node_id as usize,
                ctx.view.render_context,
            )
        });
    }
    SceneTransformRead::world_matrix_for_render_context(
        ctx.scene_assets.scene,
        space_id,
        node_id as usize,
        ctx.view.render_context,
        ctx.view.head_output_transform,
    )
}

/// Resolves the raster front face for the model matrix used by the forward vertex shader.
#[inline]
pub(super) fn front_face_for_world_matrix(world_matrix: Option<Mat4>) -> RasterFrontFace {
    world_matrix
        .map(RasterFrontFace::from_model_matrix)
        .unwrap_or_default()
}

/// Resolves root-transform parity for skinned world-space vertex streams.
#[inline]
#[cfg(test)]
pub(super) fn skinned_front_face_world_matrix(
    ctx: &DrawCollectionInputs<'_>,
    space_id: RenderSpaceId,
    node_id: i32,
    skinned: Option<&SkinnedMeshRenderer>,
) -> Option<Mat4> {
    let root_node = skinned
        .and_then(|renderer| renderer.root_bone_transform_id)
        .filter(|&id| id >= 0)
        .unwrap_or(node_id);
    if root_node < 0 {
        return None;
    }
    SceneTransformRead::world_matrix_for_render_context(
        ctx.scene_assets.scene,
        space_id,
        root_node as usize,
        ctx.view.render_context,
        ctx.view.head_output_transform,
    )
}

/// Selects the determinant source used for front-face winding.
#[inline]
pub(super) fn front_face_for_draw_matrices(
    world_space_deformed: bool,
    rigid_world_matrix: Option<Mat4>,
    deformed_front_face_world_matrix: Option<Mat4>,
) -> RasterFrontFace {
    if world_space_deformed {
        return front_face_for_world_matrix(
            deformed_front_face_world_matrix.or(rigid_world_matrix),
        );
    }
    front_face_for_world_matrix(rigid_world_matrix)
}
