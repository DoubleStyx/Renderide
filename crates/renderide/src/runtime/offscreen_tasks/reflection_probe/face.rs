//! Reflection-probe cubemap face camera, clear, clip, and filter helpers shared by queued bakes
//! and OnChanges multi-tick captures.

use hashbrown::HashSet;

use glam::Vec3;

use crate::camera::{CameraClipPlanes, HostCameraFrame};
use crate::render_graph::FrameViewClear;
use crate::scene::reflection_probe_skybox_only;
use crate::shared::{CameraClearMode, ReflectionProbeClear, ReflectionProbeState};
use crate::world_mesh::CameraTransformDrawFilter;

use super::super::cube_capture::host_camera_frame_for_cube_face;
pub(super) use super::super::cube_capture::{CUBE_FACE_COUNT, CubeCaptureFace as ProbeCubeFace};

pub(super) fn host_camera_frame_for_probe_face(
    base: &HostCameraFrame,
    state: ReflectionProbeState,
    viewport_px: (u32, u32),
    position: Vec3,
    face: ProbeCubeFace,
) -> HostCameraFrame {
    host_camera_frame_for_cube_face(
        base,
        reflection_probe_clip(state),
        viewport_px,
        position,
        face,
    )
}

pub(super) fn reflection_probe_clip(state: ReflectionProbeState) -> CameraClipPlanes {
    let near = finite_positive_or(state.near_clip, CameraClipPlanes::default().near).max(0.01);
    let far_default = CameraClipPlanes::default().far;
    let far = finite_positive_or(state.far_clip, far_default).max(near + 0.01);
    CameraClipPlanes::new(near, far)
}

pub(super) fn finite_positive_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        fallback
    }
}

pub(super) fn clear_from_reflection_probe_state(state: ReflectionProbeState) -> FrameViewClear {
    if state.clear_flags == ReflectionProbeClear::Color {
        FrameViewClear {
            mode: CameraClearMode::Color,
            color: state.background_color,
        }
    } else {
        FrameViewClear::skybox()
    }
}

pub(super) fn draw_filter_from_reflection_probe_state(
    state: &ReflectionProbeState,
) -> CameraTransformDrawFilter {
    if reflection_probe_skybox_only(state.flags) {
        CameraTransformDrawFilter {
            only: Some(HashSet::new()),
            exclude: HashSet::new(),
        }
    } else {
        CameraTransformDrawFilter {
            only: None,
            exclude: HashSet::new(),
        }
    }
}
