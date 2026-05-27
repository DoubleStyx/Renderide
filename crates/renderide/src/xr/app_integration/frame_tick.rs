//! OpenXR frame wait/locate and host camera sync.

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use openxr as xr;

use crate::camera::{StereoViewMatrices, effective_head_output_clip_planes};
use crate::diagnostics::gpu_flight_recorder::{
    GpuFlightCallResult, GpuFlightEventKind, GpuFlightOpenXrCall, GpuFlightRecorder,
};
use crate::diagnostics::log_throttle::LogThrottle;
use crate::gpu::GpuQueueAccessGate;
use crate::xr::{XrHostCameraSync, XrWgpuHandles};

use super::types::OpenxrFrameTick;

static WAIT_FRAME_FAILURE_STREAK: AtomicU32 = AtomicU32::new(0);
static LOCATE_VIEWS_FAILURE_STREAK: AtomicU32 = AtomicU32::new(0);
static POLL_EVENTS_FAILURE_LOG: LogThrottle = LogThrottle::new();
static WAIT_FRAME_FAILURE_LOG: LogThrottle = LogThrottle::new();
static LOCATE_VIEWS_FAILURE_LOG: LogThrottle = LogThrottle::new();

/// Single `wait_frame` + `locate_views` for stereo uniforms; used for both mirror and HMD paths.
pub fn openxr_begin_frame_tick(
    handles: &mut XrWgpuHandles,
    runtime: &mut impl XrHostCameraSync,
    gpu_queue_access_gate: &GpuQueueAccessGate,
    flight_recorder: &GpuFlightRecorder,
) -> Option<OpenxrFrameTick> {
    profiling::scope!("xr::begin_frame_tick");
    poll_openxr_events(handles, flight_recorder);
    if handles.xr_session.exit_requested() {
        return None;
    }
    let fs = wait_for_openxr_frame(handles, runtime, gpu_queue_access_gate, flight_recorder)?;
    let views = locate_openxr_views(
        handles,
        runtime,
        fs.should_render,
        fs.predicted_display_time,
        flight_recorder,
    );
    Some(build_openxr_frame_tick(
        runtime,
        fs.predicted_display_time,
        fs.should_render,
        views,
    ))
}

/// Polls pending OpenXR events and records the outcome without failing the frame tick.
fn poll_openxr_events(handles: &mut XrWgpuHandles, flight_recorder: &GpuFlightRecorder) {
    profiling::scope!("xr::poll_events");
    match handles.xr_session.poll_events() {
        Ok(_) => {
            flight_recorder.record(GpuFlightEventKind::OpenXrCall {
                call: GpuFlightOpenXrCall::PollEvents,
                result: GpuFlightCallResult::Ok,
                image_index: None,
                predicted_display_time_nanos: None,
            });
        }
        Err(e) => {
            flight_recorder.record(GpuFlightEventKind::OpenXrCall {
                call: GpuFlightOpenXrCall::PollEvents,
                result: GpuFlightCallResult::failed_debug(e),
                image_index: None,
                predicted_display_time_nanos: None,
            });
            if let Some(occurrence) = POLL_EVENTS_FAILURE_LOG.should_log(8, 64) {
                logger::warn!("OpenXR poll_events failed: {e:?} occurrence={occurrence}");
            }
        }
    }
}

/// Waits for the next OpenXR frame and maps recoverable failures to a skipped renderer tick.
fn wait_for_openxr_frame(
    handles: &mut XrWgpuHandles,
    runtime: &mut impl XrHostCameraSync,
    gpu_queue_access_gate: &GpuQueueAccessGate,
    flight_recorder: &GpuFlightRecorder,
) -> Option<xr::FrameState> {
    profiling::scope!("xr::wait_frame");
    let wait_start = Instant::now();
    let frame_state = handles
        .xr_session
        .wait_frame(gpu_queue_access_gate, flight_recorder);
    runtime.note_frame_timing_excluded_wait(wait_start.elapsed());
    match frame_state {
        Ok(Some(state)) => {
            WAIT_FRAME_FAILURE_STREAK.store(0, Ordering::Relaxed);
            Some(state)
        }
        Ok(None) => {
            flight_recorder.record(GpuFlightEventKind::OpenXrCall {
                call: GpuFlightOpenXrCall::WaitFrame,
                result: GpuFlightCallResult::Skipped,
                image_index: None,
                predicted_display_time_nanos: None,
            });
            None
        }
        Err(e) => {
            let streak = WAIT_FRAME_FAILURE_STREAK.fetch_add(1, Ordering::Relaxed) + 1;
            if let Some(occurrence) = WAIT_FRAME_FAILURE_LOG.should_log(8, 64) {
                logger::warn!(
                    "OpenXR wait_frame failed: {e:?} consecutive_failures={streak} occurrence={occurrence}"
                );
            }
            runtime.note_openxr_wait_frame_failed();
            None
        }
    }
}

/// Locates stereo views for a renderable frame and records throttled failure diagnostics.
fn locate_openxr_views(
    handles: &XrWgpuHandles,
    runtime: &mut impl XrHostCameraSync,
    should_render: bool,
    predicted_display_time: xr::Time,
    flight_recorder: &GpuFlightRecorder,
) -> Vec<xr::View> {
    if !should_render {
        return Vec::new();
    }
    profiling::scope!("xr::locate_views");
    match handles.xr_session.locate_views(predicted_display_time) {
        Ok(v) => {
            LOCATE_VIEWS_FAILURE_STREAK.store(0, Ordering::Relaxed);
            flight_recorder.record(GpuFlightEventKind::OpenXrCall {
                call: GpuFlightOpenXrCall::LocateViews,
                result: GpuFlightCallResult::Ok,
                image_index: None,
                predicted_display_time_nanos: Some(predicted_display_time.as_nanos()),
            });
            v
        }
        Err(e) => {
            let streak = LOCATE_VIEWS_FAILURE_STREAK.fetch_add(1, Ordering::Relaxed) + 1;
            flight_recorder.record(GpuFlightEventKind::OpenXrCall {
                call: GpuFlightOpenXrCall::LocateViews,
                result: GpuFlightCallResult::failed_debug(e),
                image_index: None,
                predicted_display_time_nanos: Some(predicted_display_time.as_nanos()),
            });
            if let Some(occurrence) = LOCATE_VIEWS_FAILURE_LOG.should_log(8, 64) {
                logger::warn!(
                    "OpenXR locate_views failed: {e:?} consecutive_failures={streak} occurrence={occurrence}"
                );
            }
            runtime.note_openxr_locate_views_failed();
            Vec::new()
        }
    }
}

/// Builds the cached frame tick and updates host-camera stereo state when VR is active.
fn build_openxr_frame_tick(
    runtime: &mut impl XrHostCameraSync,
    predicted_display_time: xr::Time,
    should_render: bool,
    views: Vec<xr::View>,
) -> OpenxrFrameTick {
    if views.len() >= 2 && runtime.vr_active() {
        apply_stereo_views_to_runtime(runtime, &views);
    }
    OpenxrFrameTick {
        predicted_display_time,
        should_render,
        views,
    }
}

/// Updates runtime head and eye transforms from the first two located OpenXR views.
fn apply_stereo_views_to_runtime(runtime: &mut impl XrHostCameraSync, views: &[xr::View]) {
    let (near, far) = effective_head_output_clip_planes(
        runtime.near_clip(),
        runtime.far_clip(),
        runtime.output_device(),
        runtime.scene_root_scale_for_clip(),
    );
    let center_pose = crate::xr::headset_center_pose_from_stereo_views(views);
    let world_from_tracking = runtime.world_from_tracking(center_pose);
    runtime.set_head_output_transform(world_from_tracking);
    let left = crate::xr::eye_view_from_xr_view_aligned(&views[0], near, far, world_from_tracking);
    let right = crate::xr::eye_view_from_xr_view_aligned(&views[1], near, far, world_from_tracking);
    runtime.set_eye_world_position((left.world_position + right.world_position) * 0.5);
    let stereo = StereoViewMatrices::new(left, right);
    runtime.set_stereo(Some(&stereo));
}
