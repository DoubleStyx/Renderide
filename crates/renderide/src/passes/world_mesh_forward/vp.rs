//! Per-draw view–projection and model matrices for world mesh forward shading.
//!
//! See module docs on [`super::WorldMeshForwardOpaquePass`] for VR vs overlay rules.

use glam::Mat4;

use crate::camera::view_matrix_for_world_mesh_render_space;
use crate::materials::RasterPipelineKind;
use crate::render_graph::{HostCameraFrame, WorldMeshDrawItem};
use crate::scene::SceneCoordinator;
use crate::shared::RenderingContext;

/// Chooses perspective vs orthographic projection for a draw (overlay vs world).
#[inline]
pub(crate) fn projection_for_world_mesh_draw(
    is_overlay: bool,
    overlay_proj: Option<Mat4>,
    world_proj: Mat4,
) -> Mat4 {
    if is_overlay {
        overlay_proj.unwrap_or(world_proj)
    } else {
        world_proj
    }
}

/// Per-draw matrices and stream metadata consumed by the forward mesh vertex shader.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PerDrawVpMatrices {
    /// Projection-view matrix for the left eye, or the only eye in mono.
    pub view_proj_left: Mat4,
    /// Projection-view matrix for the right eye; duplicates [`Self::view_proj_left`] in mono.
    pub view_proj_right: Mat4,
    /// Object model matrix used by shaders that receive local-space positions.
    pub model: Mat4,
    /// Whether the bound `@location(0)` position stream is already in world space.
    pub position_stream_world_space: bool,
}

impl PerDrawVpMatrices {
    /// Identity fallback used when the draw's render space no longer exists.
    fn identity() -> Self {
        Self {
            view_proj_left: Mat4::IDENTITY,
            view_proj_right: Mat4::IDENTITY,
            model: Mat4::IDENTITY,
            position_stream_world_space: false,
        }
    }

    /// Builds the per-draw packet from selected view matrices and resolved model selection.
    fn new(view_proj_left: Mat4, view_proj_right: Mat4, model: PerDrawModelSelection) -> Self {
        Self {
            view_proj_left: view_proj_left * model.view_proj_correction,
            view_proj_right: view_proj_right * model.view_proj_correction,
            model: model.model,
            position_stream_world_space: model.position_stream_world_space,
        }
    }
}

/// Model matrix and vertex-stream metadata selected for one draw.
#[derive(Clone, Copy, Debug, PartialEq)]
struct PerDrawModelSelection {
    /// Model matrix visible to the vertex shader.
    model: Mat4,
    /// Matrix right-multiplied into view-projection before packing.
    view_proj_correction: Mat4,
    /// Whether `@location(0)` positions have already been transformed into world space.
    position_stream_world_space: bool,
}

/// Returns a finite inverse for clipping world-space-deformed null streams.
fn inverse_or_identity(model: Mat4) -> Mat4 {
    let det = model.determinant();
    if !det.is_finite() || det.abs() < 1e-20 {
        Mat4::IDENTITY
    } else {
        model.inverse()
    }
}

/// Selects the model matrix visible to shaders for the bound position stream.
fn select_model_for_vertex_stream(
    item: &WorldMeshDrawItem,
    resolved_model: Mat4,
) -> PerDrawModelSelection {
    if item.world_space_deformed {
        if matches!(item.batch_key.pipeline, RasterPipelineKind::Null) {
            PerDrawModelSelection {
                model: resolved_model,
                view_proj_correction: inverse_or_identity(resolved_model),
                position_stream_world_space: true,
            }
        } else {
            PerDrawModelSelection {
                model: Mat4::IDENTITY,
                view_proj_correction: Mat4::IDENTITY,
                position_stream_world_space: false,
            }
        }
    } else {
        PerDrawModelSelection {
            model: resolved_model,
            view_proj_correction: Mat4::IDENTITY,
            position_stream_world_space: false,
        }
    }
}

/// Resolves the scene model matrix for a draw, using the cached collection-time matrix when present.
fn resolved_model_matrix(
    scene: &SceneCoordinator,
    item: &WorldMeshDrawItem,
    hc: HostCameraFrame,
    render_context: RenderingContext,
) -> Mat4 {
    item.rigid_world_matrix.unwrap_or_else(|| {
        scene
            .world_matrix_for_render_context(
                item.space_id,
                item.node_id as usize,
                render_context,
                hc.head_output_transform,
            )
            .unwrap_or(Mat4::IDENTITY)
    })
}

/// Resolves the model selection for `item`, avoiding scene matrix work when the shader needs identity.
fn resolve_model_selection(
    scene: &SceneCoordinator,
    item: &WorldMeshDrawItem,
    hc: HostCameraFrame,
    render_context: RenderingContext,
) -> PerDrawModelSelection {
    if item.world_space_deformed && !matches!(item.batch_key.pipeline, RasterPipelineKind::Null) {
        return PerDrawModelSelection {
            model: Mat4::IDENTITY,
            view_proj_correction: Mat4::IDENTITY,
            position_stream_world_space: false,
        };
    }
    select_model_for_vertex_stream(item, resolved_model_matrix(scene, item, hc, render_context))
}

/// Computes per-draw view-projection, model, and position-stream metadata for one sorted draw.
pub(crate) fn compute_per_draw_vp_matrices(
    scene: &SceneCoordinator,
    item: &WorldMeshDrawItem,
    hc: HostCameraFrame,
    render_context: RenderingContext,
    world_proj: Mat4,
    overlay_proj: Option<Mat4>,
) -> PerDrawVpMatrices {
    let Some(space) = scene.space(item.space_id) else {
        return PerDrawVpMatrices::identity();
    };
    let view = hc
        .explicit_world_to_view
        .unwrap_or_else(|| view_matrix_for_world_mesh_render_space(scene, space));
    let model = || resolve_model_selection(scene, item, hc, render_context);
    let vr_stereo_view = Mat4::IDENTITY;
    if let (true, Some(stereo)) = (hc.vr_active, hc.stereo) {
        let (sl, sr) = stereo.view_proj;
        if item.is_overlay {
            let op = projection_for_world_mesh_draw(true, overlay_proj, world_proj);
            let base_vp = op * view;
            PerDrawVpMatrices::new(base_vp, base_vp, model())
        } else {
            PerDrawVpMatrices::new(sl * vr_stereo_view, sr * vr_stereo_view, model())
        }
    } else {
        let proj = projection_for_world_mesh_draw(item.is_overlay, overlay_proj, world_proj);
        let base_vp = proj * view;
        PerDrawVpMatrices::new(base_vp, base_vp, model())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use glam::{Mat4, Vec3};

    use crate::materials::RasterPipelineKind;
    use crate::render_graph::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};

    use super::{projection_for_world_mesh_draw, select_model_for_vertex_stream};

    fn draw_item(skinned: bool) -> crate::render_graph::WorldMeshDrawItem {
        dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id: 0,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: false,
        })
    }

    #[test]
    fn projection_overlay_prefers_explicit_ortho_when_present() {
        let world = Mat4::IDENTITY;
        let overlay = Mat4::from_translation(Vec3::new(3.0, 0.0, 0.0));
        assert_eq!(
            projection_for_world_mesh_draw(true, Some(overlay), world),
            overlay
        );
    }

    #[test]
    fn projection_world_ignores_overlay_matrix() {
        let world = Mat4::from_scale(Vec3::splat(2.0));
        let overlay = Mat4::from_translation(Vec3::new(3.0, 0.0, 0.0));
        assert_eq!(
            projection_for_world_mesh_draw(false, Some(overlay), world),
            world
        );
    }

    #[test]
    fn null_world_space_deformed_keeps_model_and_marks_world_position_stream() {
        let item = draw_item(true);
        let model = Mat4::from_translation(Vec3::new(3.0, 4.0, 5.0))
            * Mat4::from_scale(Vec3::new(2.0, 3.0, 4.0));

        let selection = select_model_for_vertex_stream(&item, model);

        assert_eq!(selection.model, model);
        assert!(selection.position_stream_world_space);
        assert!(
            ((selection.view_proj_correction * model) - Mat4::IDENTITY)
                .to_cols_array()
                .into_iter()
                .all(|v| v.abs() < 1e-4)
        );
    }

    #[test]
    fn non_null_world_space_deformed_keeps_identity_model_path() {
        let mut item = draw_item(true);
        item.batch_key.pipeline = RasterPipelineKind::EmbeddedStem(Arc::from("unlit_default"));
        let model = Mat4::from_translation(Vec3::new(3.0, 4.0, 5.0));

        let selection = select_model_for_vertex_stream(&item, model);

        assert_eq!(selection.model, Mat4::IDENTITY);
        assert!(!selection.position_stream_world_space);
        assert_eq!(selection.view_proj_correction, Mat4::IDENTITY);
    }

    #[test]
    fn rigid_null_draw_uses_resolved_model_without_world_position_flag() {
        let item = draw_item(false);
        let model = Mat4::from_scale(Vec3::new(2.0, 3.0, 4.0));

        let selection = select_model_for_vertex_stream(&item, model);

        assert_eq!(selection.model, model);
        assert!(!selection.position_stream_world_space);
        assert_eq!(selection.view_proj_correction, Mat4::IDENTITY);
    }
}
