//! Host camera state types: per-frame camera fields, stereo bundle, and view identity.
//!
//! These types are populated by the host frame submit and consumed by world-mesh culling,
//! cluster lighting, world-mesh forward draw prep, the render graph's per-view planning, and
//! diagnostics. They live in `crate::camera` (and not in `render_graph/`) so non-graph
//! modules can talk about cameras and views without depending on the graph framework.

use glam::{Mat4, Vec3};

use crate::scene::RenderSpaceId;
use crate::shared::HeadOutputDevice;

/// Stable logical identity for one secondary camera view.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SecondaryCameraId {
    /// Render space containing the camera.
    pub render_space_id: RenderSpaceId,
    /// Dense host camera renderable index within the render space.
    pub renderable_index: i32,
}

impl SecondaryCameraId {
    /// Builds a secondary-camera id from the host render-space and dense camera row.
    pub const fn new(render_space_id: RenderSpaceId, renderable_index: i32) -> Self {
        Self {
            render_space_id,
            renderable_index,
        }
    }
}

/// Identifies one logical render view for view-scoped resources and temporal state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ViewId {
    /// Main window or OpenXR multiview (shared primary-view state).
    Main,
    /// Secondary camera, tracked independently from the render target asset it writes.
    SecondaryCamera(SecondaryCameraId),
}

impl ViewId {
    /// Builds the stable logical identity for one secondary camera view.
    pub const fn secondary_camera(render_space_id: RenderSpaceId, renderable_index: i32) -> Self {
        Self::SecondaryCamera(SecondaryCameraId::new(render_space_id, renderable_index))
    }
}

/// Per-eye matrices for an OpenXR stereo multiview view.
///
/// Consolidates the view-projection (stage → clip), view-only (world → view), and eye positions
/// so that callers cannot set one without the others. Present only on the HMD view; non-HMD views
/// carry [`None`] for this slot on [`HostCameraFrame::stereo`].
#[derive(Clone, Copy, Debug)]
pub struct StereoViewMatrices {
    /// Per-eye view–projection (reverse-Z), mapping **stage** space to clip. World mesh passes
    /// combine this with object transforms; the host `view_transform` is not multiplied again.
    pub view_proj: (Mat4, Mat4),
    /// Per-eye **view** matrices (world-to-view, handedness fix applied). Clustered lighting
    /// decomposes view and projection per eye without re-deriving from HMD poses.
    pub view_only: (Mat4, Mat4),
    /// Per-eye world-space camera positions used by shader view-vector math.
    pub eye_world_position: (Vec3, Vec3),
}

/// Latest camera-related fields from host [`crate::shared::FrameSubmitData`], updated each `frame_submit`.
#[derive(Clone, Copy, Debug)]
pub struct HostCameraFrame {
    /// Host lock-step frame index (`-1` before the first submit in standalone).
    pub frame_index: i32,
    /// Near clip distance from the host frame submission.
    pub near_clip: f32,
    /// Far clip distance from the host frame submission.
    pub far_clip: f32,
    /// Vertical field of view in **degrees** (matches host `desktopFOV`).
    pub desktop_fov_degrees: f32,
    /// Whether the host reported VR output as active for this frame.
    pub vr_active: bool,
    /// Init-time head output device selected by the host.
    pub output_device: HeadOutputDevice,
    /// `(orthographic_half_height, near, far)` from the first [`crate::shared::CameraRenderTask`] whose
    /// parameters use orthographic projection (overlay main-camera ortho override).
    pub primary_ortho_task: Option<(f32, f32, f32)>,
    /// Per-eye stereo matrices when this frame renders the OpenXR multiview view; [`None`] on
    /// desktop or secondary-RT views. Set together via [`StereoViewMatrices`] so the view-projection,
    /// view-only matrices, and per-eye camera positions cannot drift out of sync.
    pub stereo: Option<StereoViewMatrices>,
    /// Legacy Unity `HeadOutput.transform` in renderer world space.
    pub head_output_transform: Mat4,
    /// Explicit per-view world-to-view matrix override (e.g. secondary render-texture cameras).
    pub explicit_world_to_view: Option<Mat4>,
    /// Optional override view matrix for cluster + forward projection.
    pub cluster_view_override: Option<Mat4>,
    /// Optional override projection for clustered light assignment (reverse-Z).
    pub cluster_proj_override: Option<Mat4>,
    /// Explicit camera world position for `@group(0)` camera uniforms.
    pub explicit_camera_world_position: Option<Vec3>,
    /// Eye/camera world position derived from the active main render space's `view_transform`.
    pub eye_world_position: Option<Vec3>,
    /// Skips Hi-Z temporal state and uses uncull or frustum-only paths for this view.
    pub suppress_occlusion_temporal: bool,
}

impl Default for HostCameraFrame {
    fn default() -> Self {
        Self {
            frame_index: -1,
            near_clip: 0.01,
            far_clip: 10_000.0,
            desktop_fov_degrees: 60.0,
            vr_active: false,
            output_device: HeadOutputDevice::Screen,
            primary_ortho_task: None,
            stereo: None,
            head_output_transform: Mat4::IDENTITY,
            explicit_world_to_view: None,
            cluster_view_override: None,
            cluster_proj_override: None,
            explicit_camera_world_position: None,
            eye_world_position: None,
            suppress_occlusion_temporal: false,
        }
    }
}
