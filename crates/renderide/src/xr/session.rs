//! OpenXR session frame loop: wait, begin, locate views, end.
//!
//! OpenXR [`xr::Posef`] transforms **from the view-local frame to the reference (stage) frame**
//! (right-handed, Y-up, −Z forward). Scene content and render-space rigs from the host use a
//! Unity-style left-handed world basis for parity with FrooxEngine; [`openxr_pose_to_engine`] and
//! [`openxr_pose_to_host_tracking`] apply the same RH→LH mapping used for IPC so HMD views and
//! scene transforms share one world basis before [`crate::render_graph::apply_view_handedness_fix`]
//! applies the clip-space Z handling for the mesh path.
//!
//! ## Stereo convention (runtime `views` order)
//!
//! For the primary stereo view configuration (`PRIMARY_STEREO`), `views[0]` is the left eye and
//! `views[1]` the right eye. [`headset_center_pose_from_stereo_views`] averages both for the
//! center-eye pose sent over IPC via [`openxr_pose_to_host_tracking`].

#![allow(clippy::items_after_test_module)] // `mod tests` precedes `tracking_space_to_world_matrix`; reorder in a follow-up.

use glam::{Mat4, Quat, Vec3};
use openxr as xr;
use openxr::{CompositionLayerProjection, CompositionLayerProjectionView, SwapchainSubImage};

use crate::render_graph::{apply_view_handedness_fix, reverse_z_perspective_openxr_fov};
use crate::scene::render_transform_to_matrix;
use crate::shared::RenderTransform;

/// `T_renderer_world_from_view`: maps view-local points into the renderer's world basis.
///
/// Scene/object transforms are still expressed in the host/Unity-style LH basis, so the HMD pose
/// must be converted into that same basis before building the legacy `z_flip * inverse(camera)`
/// view matrix used by the mesh path.
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

/// OpenXR requires a unit quaternion; some runtimes briefly report `(0,0,0,0)`, which makes
/// `xrEndFrame` fail with `XR_ERROR_POSE_INVALID`.
fn sanitize_pose_for_end_frame(pose: xr::Posef) -> xr::Posef {
    let o = pose.orientation;
    let len_sq = o.x * o.x + o.y * o.y + o.z * o.z + o.w * o.w;
    if len_sq.is_finite() && len_sq >= 1e-10 {
        pose
    } else {
        xr::Posef {
            orientation: xr::Quaternionf {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            },
            position: pose.position,
        }
    }
}

/// Owns OpenXR session objects (constructed in [`super::bootstrap::init_wgpu_openxr`]).
pub struct XrSessionState {
    pub(super) xr_instance: xr::Instance,
    /// Dropped before [`Self::xr_instance`] so the messenger handle is destroyed first.
    #[allow(dead_code)]
    pub(super) openxr_debug_messenger: Option<super::debug_utils::OpenxrDebugUtilsMessenger>,
    pub(super) environment_blend_mode: xr::EnvironmentBlendMode,
    pub(super) session: xr::Session<xr::Vulkan>,
    pub(super) session_running: bool,
    pub(super) frame_wait: xr::FrameWaiter,
    pub(super) frame_stream: xr::FrameStream<xr::Vulkan>,
    pub(super) stage: xr::Space,
    pub(super) event_storage: xr::EventDataBuffer,
}

impl XrSessionState {
    pub(super) fn new(
        xr_instance: xr::Instance,
        openxr_debug_messenger: Option<super::debug_utils::OpenxrDebugUtilsMessenger>,
        environment_blend_mode: xr::EnvironmentBlendMode,
        session: xr::Session<xr::Vulkan>,
        frame_wait: xr::FrameWaiter,
        frame_stream: xr::FrameStream<xr::Vulkan>,
        stage: xr::Space,
    ) -> Self {
        Self {
            xr_instance,
            openxr_debug_messenger,
            environment_blend_mode,
            session,
            session_running: false,
            frame_wait,
            frame_stream,
            stage,
            event_storage: xr::EventDataBuffer::new(),
        }
    }

    /// Poll events and return `false` if the session should exit.
    pub fn poll_events(&mut self) -> Result<bool, xr::sys::Result> {
        while let Some(event) = self.xr_instance.poll_event(&mut self.event_storage)? {
            use xr::Event::*;
            match event {
                SessionStateChanged(e) => match e.state() {
                    xr::SessionState::READY => {
                        self.session
                            .begin(xr::ViewConfigurationType::PRIMARY_STEREO)?;
                        self.session_running = true;
                    }
                    xr::SessionState::STOPPING => {
                        self.session.end()?;
                        self.session_running = false;
                    }
                    xr::SessionState::EXITING | xr::SessionState::LOSS_PENDING => {
                        return Ok(false);
                    }
                    _ => {}
                },
                InstanceLossPending(_) => return Ok(false),
                _ => {}
            }
        }
        Ok(true)
    }

    /// Whether the OpenXR session is running.
    pub fn session_running(&self) -> bool {
        self.session_running
    }

    /// OpenXR instance handle (swapchain creation, view enumeration).
    pub fn xr_instance(&self) -> &xr::Instance {
        &self.xr_instance
    }

    /// Underlying Vulkan session (swapchain lifetime).
    pub fn xr_vulkan_session(&self) -> &xr::Session<xr::Vulkan> {
        &self.session
    }

    /// Stage reference space used for [`Self::locate_views`] and controller [`xr::Space`] location.
    pub fn stage_space(&self) -> &xr::Space {
        &self.stage
    }

    /// Blocks until the next frame, begins the frame stream. Returns `None` if not ready or idle.
    pub fn wait_frame(&mut self) -> Result<Option<xr::FrameState>, xr::sys::Result> {
        if !self.session_running {
            std::thread::sleep(std::time::Duration::from_millis(10));
            return Ok(None);
        }
        let state = self.frame_wait.wait()?;
        self.frame_stream.begin()?;
        Ok(Some(state))
    }

    /// Ends the frame with no composition layers (mirror path until swapchain submission is wired).
    pub fn end_frame_empty(
        &mut self,
        predicted_display_time: xr::Time,
    ) -> Result<(), xr::sys::Result> {
        self.frame_stream
            .end(predicted_display_time, self.environment_blend_mode, &[])
    }

    /// Submits a stereo projection layer referencing the acquired swapchain image (`image_rect` in pixels).
    ///
    /// For the primary stereo view configuration (`PRIMARY_STEREO`), `views[0]` is the left eye and
    /// `views[1]` the right eye. Composition layer 0 / `image_array_index` 0 is the left eye, layer 1
    /// / index 1 the right eye, matching multiview `view_index` in the stereo path.
    pub fn end_frame_projection(
        &mut self,
        predicted_display_time: xr::Time,
        swapchain: &xr::Swapchain<xr::Vulkan>,
        views: &[xr::View],
        image_rect: xr::Rect2Di,
    ) -> Result<(), xr::sys::Result> {
        if views.len() < 2 {
            return self.end_frame_empty(predicted_display_time);
        }
        let v0 = &views[0]; // left eye
        let v1 = &views[1]; // right eye
        let pose0 = sanitize_pose_for_end_frame(v0.pose);
        let pose1 = sanitize_pose_for_end_frame(v1.pose);
        let projection_views = [
            CompositionLayerProjectionView::new()
                .pose(pose0)
                .fov(v0.fov)
                .sub_image(
                    SwapchainSubImage::new()
                        .swapchain(swapchain)
                        .image_array_index(0)
                        .image_rect(image_rect),
                ),
            CompositionLayerProjectionView::new()
                .pose(pose1)
                .fov(v1.fov)
                .sub_image(
                    SwapchainSubImage::new()
                        .swapchain(swapchain)
                        .image_array_index(1)
                        .image_rect(image_rect),
                ),
        ];
        let layer = CompositionLayerProjection::new()
            .space(&self.stage)
            .views(&projection_views);
        self.frame_stream.end(
            predicted_display_time,
            self.environment_blend_mode,
            &[&layer],
        )
    }

    /// Locates stereo views for the predicted display time.
    pub fn locate_views(
        &self,
        predicted_display_time: xr::Time,
    ) -> Result<Vec<xr::View>, xr::sys::Result> {
        let (_, views) = self.session.locate_views(
            xr::ViewConfigurationType::PRIMARY_STEREO,
            predicted_display_time,
            &self.stage,
        )?;
        Ok(views)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec4;
    use openxr as xr;

    fn pose_identity() -> xr::Posef {
        xr::Posef {
            orientation: xr::Quaternionf {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            },
            position: xr::Vector3f {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        }
    }

    #[test]
    fn identity_pose_maps_to_identity_ref_from_view() {
        let m = ref_from_view_matrix(&pose_identity());
        assert!(
            m.abs_diff_eq(Mat4::IDENTITY, 1e-4),
            "expected identity ref_from_view, got {m:?}"
        );
    }

    #[test]
    fn identity_openxr_pose_maps_to_identity_engine_quat() {
        let (_p, q) = openxr_pose_to_engine(&pose_identity());
        assert!(
            q.abs_diff_eq(Quat::IDENTITY, 1e-4),
            "expected identity engine orientation, got {q:?}"
        );
    }

    #[test]
    fn host_tracking_pose_converts_to_unity_lh() {
        // OpenXR RH (-Z forward) -> FrooxEngine/Unity LH (+Z forward):
        //   position: (x, y, -z)
        //   rotation: (-qx, -qy, qz, qw)
        let pose = xr::Posef {
            orientation: xr::Quaternionf {
                x: 0.1,
                y: 0.2,
                z: 0.3,
                w: 0.9,
            },
            position: xr::Vector3f {
                x: 1.0,
                y: 2.0,
                z: -3.0,
            },
        };
        let (p, q) = openxr_pose_to_host_tracking(&pose);
        assert!(p.abs_diff_eq(Vec3::new(1.0, 2.0, 3.0), 1e-5));
        let o = pose.orientation;
        let q_expected = Quat::from_xyzw(-o.x, -o.y, o.z, o.w).normalize();
        assert!(q.abs_diff_eq(q_expected, 1e-4));
    }

    #[test]
    fn headset_center_pose_averages_positions_and_slerps_rotation() {
        let pose_l = xr::Posef {
            orientation: xr::Quaternionf {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            },
            position: xr::Vector3f {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        };
        let pose_r = xr::Posef {
            orientation: xr::Quaternionf {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            },
            position: xr::Vector3f {
                x: 0.2,
                y: 0.0,
                z: 0.0,
            },
        };
        let views = [
            xr::View {
                pose: pose_l,
                fov: xr::Fovf {
                    angle_left: 0.0,
                    angle_right: 0.0,
                    angle_up: 0.0,
                    angle_down: 0.0,
                },
            },
            xr::View {
                pose: pose_r,
                fov: xr::Fovf {
                    angle_left: 0.0,
                    angle_right: 0.0,
                    angle_up: 0.0,
                    angle_down: 0.0,
                },
            },
        ];
        let (p, q) = headset_center_pose_from_stereo_views(&views).expect("center pose");
        let (pl, _) = openxr_pose_to_host_tracking(&pose_l);
        let (pr, _) = openxr_pose_to_host_tracking(&pose_r);
        let expected_p = (pl + pr) * 0.5;
        assert!(
            p.abs_diff_eq(expected_p, 1e-4),
            "p={p:?} expected {expected_p:?}"
        );
        assert!(q.abs_diff_eq(Quat::IDENTITY, 1e-4));
    }

    #[test]
    fn pitch_up_moves_forward_point_up_in_clip_space() {
        // OpenXR uses right-handed pose rotations with -Z forward, so physical "look up"
        // corresponds to a negative X rotation.
        let angle = -0.3_f32;
        let q_xr = Quat::from_rotation_x(angle);
        let view = xr::View {
            pose: xr::Posef {
                orientation: xr::Quaternionf {
                    x: q_xr.x,
                    y: q_xr.y,
                    z: q_xr.z,
                    w: q_xr.w,
                },
                position: xr::Vector3f {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
            },
            fov: xr::Fovf {
                angle_left: -0.45,
                angle_right: 0.45,
                angle_up: 0.45,
                angle_down: -0.45,
            },
        };
        let vp = view_projection_from_xr_view(&view, 0.01, 100.0);
        // Host/scene forward is +Z (Unity LH basis). Looking up should move a forward point upward
        // in clip space, not downward.
        let clip = vp * Vec4::new(0.0, 0.0, 1.0, 1.0);
        let ndc_y = clip.y / clip.w;
        assert!(
            ndc_y > 0.0,
            "pitch up should move a forward point upward in clip space, clip={clip:?}"
        );
    }

    #[test]
    fn yaw_right_moves_forward_point_left_in_clip_space() {
        // OpenXR uses right-handed pose rotations with -Z forward, so physical "look right"
        // corresponds to a negative Y rotation.
        let angle = -0.3_f32;
        let q_xr = Quat::from_rotation_y(angle);
        let view = xr::View {
            pose: xr::Posef {
                orientation: xr::Quaternionf {
                    x: q_xr.x,
                    y: q_xr.y,
                    z: q_xr.z,
                    w: q_xr.w,
                },
                position: xr::Vector3f {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
            },
            fov: xr::Fovf {
                angle_left: -0.45,
                angle_right: 0.45,
                angle_up: 0.45,
                angle_down: -0.45,
            },
        };
        let vp = view_projection_from_xr_view(&view, 0.01, 100.0);
        // Host/scene forward is +Z (Unity LH basis). Looking right should move a forward point to
        // the left in clip space, not to the right.
        let clip = vp * Vec4::new(0.0, 0.0, 1.0, 1.0);
        let ndc_x = clip.x / clip.w;
        assert!(
            ndc_x < 0.0,
            "yaw right should move a forward point left in clip space, clip={clip:?}"
        );
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
