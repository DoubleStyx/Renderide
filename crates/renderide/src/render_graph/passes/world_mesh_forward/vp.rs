//! Per-draw view–projection and model matrices for world mesh forward shading.
//!
//! See module docs on [`super::WorldMeshForwardPass`] for VR vs overlay rules.

use glam::Mat4;

use crate::render_graph::camera::view_matrix_for_world_mesh_render_space;
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

/// Computes `(vp_left, vp_right, model)` for one sorted draw.
pub(crate) fn compute_per_draw_vp_triple(
    scene: &SceneCoordinator,
    item: &WorldMeshDrawItem,
    hc: HostCameraFrame,
    render_context: RenderingContext,
    world_proj: Mat4,
    overlay_proj: Option<Mat4>,
) -> (Mat4, Mat4, Mat4) {
    let Some(space) = scene.space(item.space_id) else {
        return (Mat4::IDENTITY, Mat4::IDENTITY, Mat4::IDENTITY);
    };
    let view = hc
        .secondary_camera_world_to_view
        .unwrap_or_else(|| view_matrix_for_world_mesh_render_space(scene, space));
    let vr_stereo_view = Mat4::IDENTITY;
    if let (true, Some((sl, sr))) = (hc.vr_active, hc.stereo_view_proj) {
        if item.is_overlay {
            let op = projection_for_world_mesh_draw(true, overlay_proj, world_proj);
            let base_vp = op * view;
            if item.world_space_deformed {
                (base_vp, base_vp, Mat4::IDENTITY)
            } else {
                let model = item.rigid_world_matrix.unwrap_or_else(|| {
                    scene
                        .world_matrix_for_render_context(
                            item.space_id,
                            item.node_id as usize,
                            render_context,
                            hc.head_output_transform,
                        )
                        .unwrap_or(Mat4::IDENTITY)
                });
                (base_vp, base_vp, model)
            }
        } else if item.world_space_deformed {
            (sl * vr_stereo_view, sr * vr_stereo_view, Mat4::IDENTITY)
        } else {
            let model = item.rigid_world_matrix.unwrap_or_else(|| {
                scene
                    .world_matrix_for_render_context(
                        item.space_id,
                        item.node_id as usize,
                        render_context,
                        hc.head_output_transform,
                    )
                    .unwrap_or(Mat4::IDENTITY)
            });
            (sl * vr_stereo_view, sr * vr_stereo_view, model)
        }
    } else {
        let proj = projection_for_world_mesh_draw(item.is_overlay, overlay_proj, world_proj);
        let base_vp = proj * view;
        if item.world_space_deformed {
            (base_vp, base_vp, Mat4::IDENTITY)
        } else {
            let model = item.rigid_world_matrix.unwrap_or_else(|| {
                scene
                    .world_matrix_for_render_context(
                        item.space_id,
                        item.node_id as usize,
                        render_context,
                        hc.head_output_transform,
                    )
                    .unwrap_or(Mat4::IDENTITY)
            });
            (base_vp, base_vp, model)
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::{Mat4, Vec3};

    use super::projection_for_world_mesh_draw;

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
}
