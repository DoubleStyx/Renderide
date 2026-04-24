//! View matrices, OpenXR pose conversion, and host tracking-space alignment for the stereo render path.

use glam::{Mat4, Quat, Vec3};
use openxr as xr;

use crate::render_graph::{apply_view_handedness_fix, reverse_z_perspective_openxr_fov};
use crate::scene::render_transform_to_matrix;
use crate::shared::RenderTransform;

/// `T_renderer_world_from_view`: maps view-local points into the renderer's world basis.
///
/// Scene/object transforms are still expressed in the host's LH basis, so the HMD pose must be
/// converted into that same basis before the mesh path builds its `z_flip * inverse(camera)` view
/// matrix.
#[inline]
pub(crate) fn ref_from_view_matrix(pose: &xr::Posef) -> Mat4 {
    let (translation, rotation) = openxr_pose_to_engine(pose);
    Mat4::from_rotation_translation(rotation, translation)
}

/// Per-eye view–projection from OpenXR [`xr::View`] (reverse-Z, renderer world basis).
pub fn view_projection_from_xr_view(view: &xr::View, near: f32, far: f32) -> Mat4 {
    view_projection_from_xr_view_aligned(view, near, far, Mat4::IDENTITY)
}

/// Per-eye view–projection from OpenXR [`xr::View`] after applying the host render-space rig
/// transform that maps tracking space into renderer world space.
pub fn view_projection_from_xr_view_aligned(
    view: &xr::View,
    near: f32,
    far: f32,
    world_from_tracking: Mat4,
) -> Mat4 {
    let ref_from_view = world_from_tracking * ref_from_view_matrix(&view.pose);
    let view_mat = apply_view_handedness_fix(ref_from_view.inverse());
    let proj = reverse_z_perspective_openxr_fov(&view.fov, near, far);
    proj * view_mat
}

/// Per-eye **view-only** matrix (world-to-view, handedness-fixed) for clustered lighting decomposition.
pub fn view_from_xr_view_aligned(view: &xr::View, world_from_tracking: Mat4) -> Mat4 {
    let ref_from_view = world_from_tracking * ref_from_view_matrix(&view.pose);
    apply_view_handedness_fix(ref_from_view.inverse())
}

fn averaged_stereo_fov(views: &[xr::View]) -> Option<xr::Fovf> {
    match views {
        [] => None,
        [view] => Some(view.fov),
        [left, right, ..] => {
            let avg_angle = |a: f32, b: f32| ((a.tan() + b.tan()) * 0.5).atan();
            Some(xr::Fovf {
                angle_left: avg_angle(left.fov.angle_left, right.fov.angle_left),
                angle_right: avg_angle(left.fov.angle_right, right.fov.angle_right),
                angle_up: avg_angle(left.fov.angle_up, right.fov.angle_up),
                angle_down: avg_angle(left.fov.angle_down, right.fov.angle_down),
            })
        }
    }
}

/// Center-eye desktop mirror projection from stereo OpenXR views after applying host tracking-space
/// alignment. This is used for the desktop window only; headset submission still uses true left/right
/// per-eye matrices.
pub fn center_view_projection_from_stereo_views_aligned(
    views: &[xr::View],
    near: f32,
    far: f32,
    world_from_tracking: Mat4,
) -> Option<Mat4> {
    let (position, rotation) = headset_center_pose_from_stereo_views(views)?;
    let fov = averaged_stereo_fov(views)?;
    let world_from_view = world_from_tracking * Mat4::from_rotation_translation(rotation, position);
    let view_mat = apply_view_handedness_fix(world_from_view.inverse());
    let proj = reverse_z_perspective_openxr_fov(&fov, near, far);
    Some(proj * view_mat)
}

/// Maps an OpenXR [`xr::Posef`] to the renderer's world translation + rotation.
///
/// The renderer currently keeps scene/object transforms in the same host/Unity-style LH basis as
/// [`crate::shared::RenderTransform`]. Use the same conversion as host tracking here so stereo HMD
/// views and host scene transforms live in one basis. The later `apply_view_handedness_fix`
/// handles the clip-space-facing `Z` flip used by the render graph.
pub fn openxr_pose_to_engine(pose: &xr::Posef) -> (Vec3, Quat) {
    openxr_pose_to_host_tracking(pose)
}

/// Position and orientation for **host IPC** (FrooxEngine [`crate::shared::HeadsetState`]).
///
/// FrooxEngine/Resonite uses Unity left-handed space (+Z forward), while OpenXR is right-handed
/// (-Z forward). Conversion: mirror Z on position and reflect the rotation basis with `S*R*S`
/// where `S = diag(1, 1, -1)`.
///   position:  `(x, y, -z)`
///   rotation:  `(-qx, -qy, qz, qw)`
pub fn openxr_pose_to_host_tracking(pose: &xr::Posef) -> (Vec3, Quat) {
    let p = Vec3::new(pose.position.x, pose.position.y, -pose.position.z);
    let o = pose.orientation;
    let q = Quat::from_xyzw(-o.x, -o.y, o.z, o.w);
    let len_sq = q.length_squared();
    let q = if len_sq.is_finite() && len_sq >= 1e-10 {
        q.normalize()
    } else {
        Quat::IDENTITY
    };
    (p, q)
}

/// Headset pose for IPC in host tracking space ([`openxr_pose_to_host_tracking`]).
pub fn headset_pose_from_xr_view(view: &xr::View) -> (Vec3, Quat) {
    openxr_pose_to_host_tracking(&view.pose)
}

/// Approximates **center eye** (Unity `XRNode.CenterEye`): averages per-eye positions and slerps
/// orientations from the first two stereo [`xr::View`] entries using [`openxr_pose_to_host_tracking`].
pub fn headset_center_pose_from_stereo_views(views: &[xr::View]) -> Option<(Vec3, Quat)> {
    match views.len() {
        0 => None,
        1 => Some(headset_pose_from_xr_view(&views[0])),
        _ => {
            let (p0, r0) = openxr_pose_to_host_tracking(&views[0].pose);
            let (p1, r1) = openxr_pose_to_host_tracking(&views[1].pose);
            let pos = (p0 + p1) * 0.5;
            let rot = r0.slerp(r1, 0.5).normalize();
            Some((pos, rot))
        }
    }
}

/// Reconstructs the same tracking-space -> world-space rig alignment used by Unity's
/// `HeadOutput.UpdatePositioning` / `UpdateOverridenView`.
///
/// - Without override-view, the tracking origin is simply rooted at `root_transform`.
/// - With override-view, the rig is additionally shifted/rotated/scaled so the current tracked
///   center-eye lands on `view_transform`.
pub fn tracking_space_to_world_matrix(
    root_transform: &RenderTransform,
    view_transform: &RenderTransform,
    override_view_position: bool,
    center_pose_tracking: Option<(Vec3, Quat)>,
) -> Mat4 {
    if !override_view_position {
        return render_transform_to_matrix(root_transform);
    }
    let center_from_tracking = center_pose_tracking
        .map(|(position, rotation)| Mat4::from_rotation_translation(rotation, position))
        .unwrap_or(Mat4::IDENTITY);
    render_transform_to_matrix(view_transform) * center_from_tracking.inverse()
}
