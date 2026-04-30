//! World-to-view construction and overlay-space rules for mesh rendering.

use glam::{Mat4, Vec3};

use crate::scene::render_transform_to_matrix;
use crate::scene::{RenderSpaceState, SceneCoordinator};
use crate::shared::RenderTransform;

/// Clamps scale for view matrix construction: if any axis is nearly zero, use unit scale.
pub fn filter_scale_legacy(scale: Vec3) -> Vec3 {
    if scale.x.min(scale.y).min(scale.z) <= 1e-8 {
        Vec3::splat(1.0)
    } else {
        scale
    }
}

/// Z-flip for RH engine space to Vulkan/WebGPU-style clip.
#[inline]
pub fn apply_view_handedness_fix(view: Mat4) -> Mat4 {
    let z_flip = Mat4::from_scale(Vec3::new(1.0, 1.0, -1.0));
    z_flip * view
}

/// World-to-view matrix from a host [`RenderTransform`] (camera / eye TRS).
///
/// Applies scale filtering and handedness fix so `view_proj * world_pos` matches the mesh pass.
pub fn view_matrix_from_render_transform(tr: &RenderTransform) -> Mat4 {
    let mut t = *tr;
    let fs = filter_scale_legacy(Vec3::new(tr.scale.x, tr.scale.y, tr.scale.z));
    t.scale.x = fs.x;
    t.scale.y = fs.y;
    t.scale.z = fs.z;
    let cam = render_transform_to_matrix(&t);
    apply_view_handedness_fix(cam.inverse())
}

/// World-to-view for mesh rendering in `space`, accounting for [`RenderSpaceState::is_overlay`].
///
/// Overlay render spaces re-root object meshes into the main world's coordinates via
/// [`SceneCoordinator::world_matrix_for_render_context`]; the camera view must therefore match the
/// active main (non-overlay) space, not the overlay space's own view transform (Unity-style
/// head output + overlay positioning parity).
pub fn view_matrix_for_world_mesh_render_space(
    scene: &SceneCoordinator,
    space: &RenderSpaceState,
) -> Mat4 {
    if space.is_overlay {
        scene.active_main_space().map_or_else(
            || view_matrix_from_render_transform(&space.view_transform),
            |main| view_matrix_from_render_transform(&main.view_transform),
        )
    } else {
        view_matrix_from_render_transform(&space.view_transform)
    }
}
