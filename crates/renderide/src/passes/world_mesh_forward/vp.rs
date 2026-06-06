//! Per-draw view-projection and model matrices for world mesh forward shading.
//!
//! See module docs on [`super::WorldMeshForwardOpaquePass`] for VR vs overlay rules.

use glam::{Mat4, Vec3};

use crate::camera::HostCameraFrame;
use crate::camera::view_matrix_for_host_world_mesh_space;
use crate::materials::RasterPipelineKind;
use crate::scene::SceneCoordinator;
use crate::shared::RenderingContext;
use crate::world_mesh::WorldMeshDrawItem;

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

/// Fixed view matrix used by desktop overlay-camera draws.
#[inline]
pub(crate) fn overlay_camera_view_matrix() -> Mat4 {
    Mat4::from_translation(Vec3::new(0.0, 0.0, -1.0))
}

/// Projection-view matrix for desktop overlay-camera draws.
#[inline]
pub(crate) fn overlay_view_projection(overlay_proj: Option<Mat4>, world_proj: Mat4) -> Mat4 {
    projection_for_world_mesh_draw(true, overlay_proj, world_proj) * overlay_camera_view_matrix()
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
///
/// Overlay-layer items (`is_overlay == true`) bypass the overlay-space head-output re-rooting and
/// use the transform chain relative to the nearest overlay-layer ancestor. Combined with an
/// overlay-camera view matrix in [`compute_per_draw_vp_matrices`], this puts overlay objects at their
/// authored local position in normalized screen space (CSS-overlay style) regardless of where the
/// camera is in the world, matching the host `RadiantDash` desktop layout (`UpdateProjection`
/// scales `VisualsRoot` against `WindowResolution` so the dash fits a unit-height ortho frustum
/// centered on the view).
fn resolved_model_matrix(
    scene: &SceneCoordinator,
    item: &WorldMeshDrawItem,
    hc: &HostCameraFrame,
    render_context: RenderingContext,
) -> Mat4 {
    if item.is_overlay {
        if let Some(model) = item.rigid_world_matrix {
            return model;
        }
        return scene
            .overlay_layer_model_matrix_for_context(
                item.space_id,
                item.node_id as usize,
                render_context,
            )
            .or_else(|| {
                scene.world_matrix_for_context(item.space_id, item.node_id as usize, render_context)
            })
            .unwrap_or(Mat4::IDENTITY);
    }
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
    hc: &HostCameraFrame,
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
///
/// **Overlay-layer rendering**: items with `is_overlay == true` (host `LayerType.Overlay`) render
/// in normalized screen space using the fixed desktop overlay-camera view matrix. The model matrix is the local
/// hierarchy matrix (no overlay-space head-output re-rooting; see [`resolved_model_matrix`]) so
/// overlay objects sit at their authored local position regardless of camera placement. The
/// projection comes from the dedicated overlay ortho built in
/// [`crate::camera::WorldProjectionSet::from_scene_host`], not the world-camera projection.
pub(crate) fn compute_per_draw_vp_matrices(
    scene: &SceneCoordinator,
    item: &WorldMeshDrawItem,
    hc: &HostCameraFrame,
    render_context: RenderingContext,
    world_proj: Mat4,
    overlay_proj: Option<Mat4>,
) -> PerDrawVpMatrices {
    let Some(space) = scene.space(item.space_id) else {
        return PerDrawVpMatrices::identity();
    };
    if item.is_overlay {
        let overlay_vp = overlay_view_projection(overlay_proj, world_proj);
        let model = resolve_model_selection(scene, item, hc, render_context);
        return PerDrawVpMatrices::new(overlay_vp, overlay_vp, model);
    }
    let model = || resolve_model_selection(scene, item, hc, render_context);
    let view = view_matrix_for_host_world_mesh_space(scene, space, hc);
    let vr_stereo_view = Mat4::IDENTITY;
    if let Some(stereo) = hc.active_stereo() {
        let (sl, sr) = stereo.view_proj_pair();
        PerDrawVpMatrices::new(sl * vr_stereo_view, sr * vr_stereo_view, model())
    } else {
        let proj = projection_for_world_mesh_draw(false, overlay_proj, world_proj);
        let base_vp = proj * view;
        PerDrawVpMatrices::new(base_vp, base_vp, model())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use glam::{Mat4, Quat, Vec3, Vec4};

    use crate::camera::{CameraClipPlanes, HostCameraFrame, Viewport};
    use crate::materials::RasterPipelineKind;
    use crate::scene::{RenderSpaceId, SceneCoordinator};
    use crate::shared::{LayerType, RenderTransform, RenderingContext};
    use crate::world_mesh::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};

    use super::{
        compute_per_draw_vp_matrices, overlay_camera_view_matrix, overlay_view_projection,
        projection_for_world_mesh_draw, select_model_for_vertex_stream,
    };

    fn identity_transform() -> RenderTransform {
        RenderTransform {
            position: Vec3::ZERO,
            scale: Vec3::ONE,
            rotation: Quat::IDENTITY,
        }
    }

    fn draw_item(skinned: bool) -> crate::world_mesh::WorldMeshDrawItem {
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
    fn overlay_view_projection_applies_fixed_camera_view_shift() {
        let world = Mat4::from_scale(Vec3::splat(2.0));
        let overlay = Mat4::from_translation(Vec3::new(3.0, 0.0, 0.0));

        assert_eq!(
            overlay_view_projection(Some(overlay), world),
            overlay * overlay_camera_view_matrix()
        );
    }

    #[test]
    fn overlay_draw_projection_places_dash_geometry_inside_camera_volume() {
        let mut scene = SceneCoordinator::new();
        let space_id = RenderSpaceId(103);
        scene.test_seed_space_identity_worlds(
            space_id,
            vec![
                identity_transform(),
                RenderTransform {
                    position: Vec3::new(12.0, -5.0, 8.0),
                    scale: Vec3::splat(3.0),
                    rotation: Quat::from_axis_angle(Vec3::Y, 1.2),
                },
                identity_transform(),
                RenderTransform {
                    position: Vec3::ZERO,
                    scale: Vec3::splat(1.5),
                    rotation: Quat::IDENTITY,
                },
                identity_transform(),
            ],
            vec![-1, 0, 1, 2, 3],
        );
        scene.test_set_space_overlay(space_id, true);
        scene.test_push_layer_assignment(space_id, 2, LayerType::Overlay);

        let mut item = draw_item(false);
        item.space_id = space_id;
        item.node_id = 4;
        item.renderable_index = 4;
        item.is_overlay = true;

        let viewport = Viewport::from_tuple((1920, 1080));
        let overlay_proj =
            HostCameraFrame::overlay_projection(viewport, CameraClipPlanes::new(0.1, 100.0));
        let matrices = compute_per_draw_vp_matrices(
            &scene,
            &item,
            &HostCameraFrame::default(),
            RenderingContext::UserView,
            Mat4::IDENTITY,
            Some(overlay_proj),
        );

        let origin_clip = matrices.view_proj_left * matrices.model * Vec4::new(0.0, 0.0, 0.0, 1.0);
        let origin_ndc = origin_clip.truncate() / origin_clip.w;
        let origin_ndc_x = origin_ndc.x;
        let origin_ndc_y = origin_ndc.y;
        let origin_ndc_z = origin_ndc.z;
        assert!(
            origin_ndc_x.abs() < 1e-4,
            "expected NDC x at screen center, got {origin_ndc_x}",
        );
        assert!(
            origin_ndc_y.abs() < 1e-4,
            "expected NDC y at screen center, got {origin_ndc_y}",
        );
        assert!(
            (0.0..=1.0).contains(&origin_ndc_z),
            "expected reverse-Z NDC z inside [0, 1], got {origin_ndc_z}",
        );

        let right_clip = matrices.view_proj_left * matrices.model * Vec4::new(0.5, 0.0, 0.0, 1.0);
        let right_ndc_x = right_clip.x / right_clip.w;
        let expected_x = 0.75 * 1080.0 / 1920.0;
        assert!(
            (right_ndc_x - expected_x).abs() < 1e-3,
            "expected NDC x {expected_x}, got {right_ndc_x}",
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
