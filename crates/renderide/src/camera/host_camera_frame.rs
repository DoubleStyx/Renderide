//! Per-frame host camera state and the shared HostCameraFrame builder.
//!
//! `HostCameraFrame` is populated by the host frame submit and consumed by world-mesh culling,
//! cluster lighting, world-mesh forward draw prep, the render graph's per-view planning, and
//! diagnostics. It lives in `crate::camera` (and not in `render_graph/`) so non-graph modules
//! can talk about cameras and views without depending on the graph framework.

use glam::{Mat4, Vec3};

use crate::shared::{CameraProjection, HeadOutputDevice};

use super::geometry::{
    CameraClipPlanes, CameraPose, EyeView, OrthographicProjectionSpec, Viewport,
};
use super::stereo::StereoViewMatrices;

/// Projection family used by shader helpers that need to distinguish perspective from orthographic math.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CameraProjectionKind {
    /// Perspective projection with rays converging on the camera position.
    #[default]
    Perspective,
    /// Orthographic projection with parallel camera rays.
    Orthographic,
}

/// Latest camera-related fields from host [`crate::shared::FrameSubmitData`], updated each `frame_submit`.
#[derive(Clone, Copy, Debug)]
pub struct HostCameraFrame {
    /// Host lock-step frame index (`-1` before the first submit in standalone).
    pub frame_index: i32,
    /// Clip distances from the host frame submission.
    pub clip: CameraClipPlanes,
    /// Vertical field of view in **degrees** (matches host `desktopFOV`).
    pub desktop_fov_degrees: f32,
    /// Whether the host reported VR output as active for this frame.
    pub vr_active: bool,
    /// Init-time head output device selected by the host.
    pub output_device: HeadOutputDevice,
    /// Active projection family for the view represented by this frame.
    pub projection_kind: CameraProjectionKind,
    /// First orthographic render-task projection (overlay main-camera ortho override).
    pub primary_ortho_task: Option<OrthographicProjectionSpec>,
    /// Per-eye stereo matrices when this frame renders the OpenXR multiview view; [`None`] on
    /// desktop or secondary-RT views. Set together via [`StereoViewMatrices`] so the view-projection,
    /// view-only matrices, and per-eye camera positions cannot drift out of sync.
    pub stereo: Option<StereoViewMatrices>,
    /// Host `HeadOutput.transform` in renderer world space.
    pub head_output_transform: Mat4,
    /// Explicit per-view camera data (e.g. secondary render-texture cameras).
    pub explicit_view: Option<EyeView>,
    /// Eye/camera world position derived from the active main render space's `view_transform`.
    pub eye_world_position: Option<Vec3>,
    /// Skips Hi-Z temporal state and uses uncull or frustum-only paths for this view.
    pub suppress_occlusion_temporal: bool,
}

impl Default for HostCameraFrame {
    fn default() -> Self {
        Self {
            frame_index: -1,
            clip: CameraClipPlanes::default(),
            desktop_fov_degrees: 60.0,
            vr_active: false,
            output_device: HeadOutputDevice::Screen,
            projection_kind: CameraProjectionKind::Perspective,
            primary_ortho_task: None,
            stereo: None,
            head_output_transform: Mat4::IDENTITY,
            explicit_view: None,
            eye_world_position: None,
            suppress_occlusion_temporal: false,
        }
    }
}

impl HostCameraFrame {
    /// Returns the near clip distance.
    #[cfg(test)]
    pub const fn near_clip(&self) -> f32 {
        self.clip.near
    }

    /// Returns the far clip distance.
    #[cfg(test)]
    pub const fn far_clip(&self) -> f32 {
        self.clip.far
    }

    /// Returns the explicit world-to-view override when present.
    #[inline]
    pub fn explicit_world_to_view(&self) -> Option<Mat4> {
        self.explicit_view.map(|view| view.view)
    }

    /// Returns the explicit camera world position when present.
    #[inline]
    pub fn explicit_world_position(&self) -> Option<Vec3> {
        self.explicit_view.map(|view| view.world_position)
    }

    /// Returns the explicit view and projection override when present.
    #[inline]
    pub fn explicit_view_projection(&self) -> Option<(Mat4, Mat4)> {
        self.explicit_view.map(|view| (view.view, view.proj))
    }

    /// Returns active stereo only when the host frame is currently VR-active.
    #[inline]
    pub fn active_stereo(&self) -> Option<&StereoViewMatrices> {
        if self.vr_active {
            self.stereo.as_ref()
        } else {
            None
        }
    }

    /// Returns the dedicated screen-overlay orthographic projection.
    #[inline]
    pub fn overlay_projection(viewport: Viewport, fallback_clip: CameraClipPlanes) -> Mat4 {
        OrthographicProjectionSpec::new(1.0, fallback_clip).projection(viewport)
    }

    /// Resolves the world-space origin used for view-distance sorting.
    #[inline]
    pub fn view_origin_world(&self) -> Vec3 {
        self.explicit_world_position()
            .or(self.eye_world_position)
            .unwrap_or_else(|| self.head_output_transform.col(3).truncate())
    }

    /// Resolves left/right world camera positions for frame globals.
    pub fn camera_world_pair(&self) -> (Vec3, Vec3) {
        if let Some(camera_world) = self.explicit_world_position() {
            return (camera_world, camera_world);
        }
        if let Some(stereo) = self.stereo.as_ref() {
            return stereo.world_position_pair();
        }
        let camera_world = self
            .eye_world_position
            .unwrap_or_else(|| self.head_output_transform.col(3).truncate());
        (camera_world, camera_world)
    }
}

/// Resolved per-camera inputs for [`build_single_camera_frame`].
pub(super) struct SingleCameraInputs {
    /// Pixel viewport the camera renders into.
    pub viewport: Viewport,
    /// Camera world pose (world matrix and decoded world position).
    pub pose: CameraPose,
    /// Near/far clip distances, already validated/clamped by the caller.
    pub clip: CameraClipPlanes,
    /// Vertical FOV in degrees, already clamped by the caller.
    pub fov_degrees: f32,
    /// Orthographic half-height (consulted only for orthographic projections).
    pub orthographic_size: f32,
    /// Wire-format projection selector from the host.
    pub projection: CameraProjection,
    /// When `true`, the resulting view skips Hi-Z temporal state.
    pub suppress_occlusion_temporal: bool,
}

/// Builds a [`HostCameraFrame`] for a single-camera view (secondary render texture or host
/// camera readback task) from already-resolved pose, clip, and projection inputs.
///
/// Carries forward `frame_index`, `output_device`, and `head_output_transform` from `base`.
/// `orthographic_size` is consulted only when `projection` is orthographic.
pub(super) fn build_single_camera_frame(
    base: &HostCameraFrame,
    inputs: SingleCameraInputs,
) -> HostCameraFrame {
    let SingleCameraInputs {
        viewport,
        pose,
        clip,
        fov_degrees,
        orthographic_size,
        projection,
        suppress_occlusion_temporal,
    } = inputs;
    let (explicit_view, primary_ortho_task, projection_kind) = match projection {
        CameraProjection::Orthographic => {
            let spec = OrthographicProjectionSpec::new(orthographic_size, clip);
            (
                EyeView::from_pose_projection(pose, spec.projection(viewport)),
                Some(spec),
                CameraProjectionKind::Orthographic,
            )
        }
        CameraProjection::Perspective | CameraProjection::Panoramic => (
            EyeView::perspective_from_pose(pose, viewport, fov_degrees, clip),
            None,
            CameraProjectionKind::Perspective,
        ),
    };

    HostCameraFrame {
        frame_index: base.frame_index,
        clip,
        desktop_fov_degrees: fov_degrees,
        vr_active: false,
        output_device: base.output_device,
        projection_kind,
        primary_ortho_task,
        stereo: None,
        head_output_transform: base.head_output_transform,
        explicit_view: Some(explicit_view),
        eye_world_position: Some(pose.world_position),
        suppress_occlusion_temporal,
    }
}

#[cfg(test)]
mod tests {
    use glam::{Mat4, Vec3};

    use super::{CameraProjectionKind, EyeView, HostCameraFrame, StereoViewMatrices};

    fn eye_at(position: Vec3) -> EyeView {
        EyeView::new(Mat4::IDENTITY, Mat4::IDENTITY, Mat4::IDENTITY, position)
    }

    #[test]
    fn view_origin_prefers_explicit_then_eye_then_head_output() {
        let mut camera = HostCameraFrame {
            head_output_transform: Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0)),
            ..Default::default()
        };
        assert_eq!(camera.view_origin_world(), Vec3::new(1.0, 2.0, 3.0));

        camera.eye_world_position = Some(Vec3::new(4.0, 5.0, 6.0));
        assert_eq!(camera.view_origin_world(), Vec3::new(4.0, 5.0, 6.0));

        camera.explicit_view = Some(eye_at(Vec3::new(7.0, 8.0, 9.0)));
        assert_eq!(camera.view_origin_world(), Vec3::new(7.0, 8.0, 9.0));
    }

    #[test]
    fn camera_world_pair_prefers_explicit_then_stereo_then_eye() {
        let mut camera = HostCameraFrame {
            eye_world_position: Some(Vec3::new(1.0, 0.0, 0.0)),
            ..Default::default()
        };
        assert_eq!(
            camera.camera_world_pair(),
            (Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0))
        );

        camera.stereo = Some(StereoViewMatrices::new(
            eye_at(Vec3::new(2.0, 0.0, 0.0)),
            eye_at(Vec3::new(3.0, 0.0, 0.0)),
        ));
        assert_eq!(
            camera.camera_world_pair(),
            (Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 0.0, 0.0))
        );

        camera.explicit_view = Some(eye_at(Vec3::new(4.0, 0.0, 0.0)));
        assert_eq!(
            camera.camera_world_pair(),
            (Vec3::new(4.0, 0.0, 0.0), Vec3::new(4.0, 0.0, 0.0))
        );
    }

    #[test]
    fn host_camera_defaults_to_perspective_projection_kind() {
        assert_eq!(
            HostCameraFrame::default().projection_kind,
            CameraProjectionKind::Perspective
        );
    }
}
