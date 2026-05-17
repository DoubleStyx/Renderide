//! HMD multiview submission into the OpenXR stereo swapchain.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use crate::diagnostics::gpu_flight_recorder::{
    GpuFlightCallResult, GpuFlightEventKind, GpuFlightOpenXrCall,
};
use crate::diagnostics::log_throttle::LogThrottle;
use crate::gpu::GpuContext;
use crate::gpu::driver_thread::BlockingCallWatchdog as EndFrameWatchdog;
use crate::render_graph::ExternalFrameTargets;
use crate::xr::{XR_COLOR_FORMAT, XrFrameRenderer};
use openxr as xr;
use parking_lot::Mutex;

use super::super::swapchain::{XrAcquiredSwapchainImage, XrStereoSwapchain};
use super::resources::{ensure_owned_hmd_targets, ensure_stereo_swapchain};
use super::types::{OpenxrFrameTick, XrSessionBundle};

/// Deadline for a single `xrWaitSwapchainImage` call before the watchdog logs a compositor stall.
///
/// Observation only: the call keeps its original `xr::Duration::INFINITE` because openxr 0.21
/// swallows `XR_TIMEOUT_EXPIRED` (returns `Ok(())` identically to success), making a bounded
/// timeout indistinguishable from a real image release.
const WAIT_IMAGE_WATCHDOG_TIMEOUT: Duration = Duration::from_millis(500);
/// Throttle for expected HMD submit skips where no OpenXR call failed.
static HMD_SUBMIT_SKIP_LOG: LogThrottle = LogThrottle::new();
/// Throttle for HMD submit failures that should be visible in INFO-level crash logs.
static HMD_SUBMIT_FAILURE_LOG: LogThrottle = LogThrottle::new();

/// Low-cardinality reason why the HMD projection path did not submit this tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HmdSubmitSkipReason {
    /// OpenXR session is not running.
    SessionNotRunning,
    /// OpenXR session is not visible to the user.
    SessionNotVisible,
    /// Runtime does not currently consider VR submission active.
    RuntimeVrInactive,
    /// The selected wgpu device cannot render stereo multiview.
    MissingMultiviewFeature,
    /// The current OpenXR frame tick does not request rendering.
    TickShouldNotRender,
    /// OpenXR located fewer than two projection views for `PRIMARY_STEREO`.
    LocatedViewCount {
        /// Number of located views reported by OpenXR.
        view_count: usize,
    },
    /// The stereo swapchain could not be created or refreshed.
    StereoSwapchainUnavailable,
    /// Swapchain creation reported success but no swapchain was stored.
    StereoSwapchainMissingAfterEnsure,
    /// The renderer-owned HMD color/depth targets could not be created or refreshed.
    OwnedHmdTargetsUnavailable,
    /// The stereo swapchain disappeared before image acquisition.
    StereoSwapchainMissingBeforeAcquire,
    /// `xrAcquireSwapchainImage` failed.
    SwapchainAcquireFailed,
    /// `xrWaitSwapchainImage` failed.
    SwapchainWaitFailed,
    /// The acquired OpenXR image could not be imported into wgpu.
    SwapchainImageImportFailed,
    /// The renderer-owned HMD targets disappeared after resize handling.
    OwnedHmdTargetsMissingAfterResize,
    /// The renderer failed while submitting the HMD graph.
    SubmitHmdViewFailed,
}

impl fmt::Display for HmdSubmitSkipReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SessionNotRunning => f.write_str("session_not_running"),
            Self::SessionNotVisible => f.write_str("session_not_visible"),
            Self::RuntimeVrInactive => f.write_str("runtime_vr_inactive"),
            Self::MissingMultiviewFeature => f.write_str("missing_multiview_feature"),
            Self::TickShouldNotRender => f.write_str("tick_should_not_render"),
            Self::LocatedViewCount { view_count } => {
                write!(f, "located_view_count_{view_count}")
            }
            Self::StereoSwapchainUnavailable => f.write_str("stereo_swapchain_unavailable"),
            Self::StereoSwapchainMissingAfterEnsure => {
                f.write_str("stereo_swapchain_missing_after_ensure")
            }
            Self::OwnedHmdTargetsUnavailable => f.write_str("owned_hmd_targets_unavailable"),
            Self::StereoSwapchainMissingBeforeAcquire => {
                f.write_str("stereo_swapchain_missing_before_acquire")
            }
            Self::SwapchainAcquireFailed => f.write_str("swapchain_acquire_failed"),
            Self::SwapchainWaitFailed => f.write_str("swapchain_wait_failed"),
            Self::SwapchainImageImportFailed => f.write_str("swapchain_image_import_failed"),
            Self::OwnedHmdTargetsMissingAfterResize => {
                f.write_str("owned_hmd_targets_missing_after_resize")
            }
            Self::SubmitHmdViewFailed => f.write_str("submit_hmd_view_failed"),
        }
    }
}

/// Result of attempting to render and submit the OpenXR HMD projection for one tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HmdSubmitOutcome {
    /// HMD rendering did not start; callers should still render secondary cameras.
    SkippedBeforeRender,
    /// HMD rendering was queued, but no OpenXR projection layer was queued.
    RenderedWithoutProjection,
    /// HMD rendering, final OpenXR copy, and projection finalize were queued.
    ProjectionQueued,
}

impl HmdSubmitOutcome {
    /// Returns `true` when an OpenXR projection layer finalize was queued.
    pub const fn projection_queued(self) -> bool {
        matches!(self, Self::ProjectionQueued)
    }

    /// Returns `true` when HMD/secondary graph work was not queued by the HMD submit path.
    pub const fn should_render_non_hmd_views(self) -> bool {
        matches!(self, Self::SkippedBeforeRender)
    }
}

/// Renders to renderer-owned stereo targets, copies them to OpenXR, and queues
/// `xrReleaseSwapchainImage` + `xrEndFrame` onto the driver thread.
///
/// Uses the same [`xr::FrameState`] as [`crate::xr::openxr_begin_frame_tick`] -- no second
/// `wait_frame`. After this returns successfully the next tick's `wait_frame` blocks on the
/// matching finalize signal before issuing `xrBeginFrame`, preserving OpenXR begin/end ordering
/// across the deferred handoff.
pub fn try_openxr_hmd_multiview_submit(
    gpu: &mut GpuContext,
    bundle: &mut XrSessionBundle,
    runtime: &mut impl XrFrameRenderer,
    tick: &OpenxrFrameTick,
) -> HmdSubmitOutcome {
    let Some(extent) = ensure_hmd_submit_resources(gpu, bundle, runtime, tick) else {
        return HmdSubmitOutcome::SkippedBeforeRender;
    };
    let Some(sc) = bundle.stereo_swapchain.as_ref() else {
        log_hmd_submit_failure(HmdSubmitSkipReason::StereoSwapchainMissingBeforeAcquire);
        return HmdSubmitOutcome::SkippedBeforeRender;
    };
    let swapchain_handle = Arc::clone(&sc.handle);
    let Some(hmd_targets) = bundle.hmd_targets.as_ref() else {
        logger::debug!("OpenXR owned HMD targets missing after resize");
        log_hmd_submit_failure(HmdSubmitSkipReason::OwnedHmdTargetsMissingAfterResize);
        return HmdSubmitOutcome::SkippedBeforeRender;
    };
    let ext = ExternalFrameTargets {
        color_view: hmd_targets.color_array_view(),
        depth_texture: hmd_targets.depth_texture(),
        depth_view: hmd_targets.depth_view(),
        extent_px: extent,
        surface_format: XR_COLOR_FORMAT,
    };
    let rect = xr::Rect2Di {
        offset: xr::Offset2Di { x: 0, y: 0 },
        extent: xr::Extent2Di {
            width: extent.0 as i32,
            height: extent.1 as i32,
        },
    };
    // Unified submit: HMD stereo + every active secondary RT in one `execute_multi_view_frame`
    // call. The HMD view replaces the main camera for this tick.
    {
        profiling::scope!("xr::submit_hmd_view");
        if let Err(error) = runtime.submit_hmd_view(gpu, ext) {
            log_hmd_submit_failure_with_display_error(
                HmdSubmitSkipReason::SubmitHmdViewFailed,
                &error,
            );
            return HmdSubmitOutcome::SkippedBeforeRender;
        }
    }
    let Some(projection_views) = stereo_views(&tick.views) else {
        return HmdSubmitOutcome::RenderedWithoutProjection;
    };
    let Some(acquired_image) = acquire_imported_hmd_image(gpu, sc) else {
        return HmdSubmitOutcome::RenderedWithoutProjection;
    };
    let final_copy_cmd = {
        let Some(targets) = bundle.hmd_targets.as_ref() else {
            logger::debug!("OpenXR owned HMD targets missing before final copy");
            log_hmd_submit_failure(HmdSubmitSkipReason::OwnedHmdTargetsMissingAfterResize);
            return HmdSubmitOutcome::RenderedWithoutProjection;
        };
        profiling::scope!("xr::hmd_final_copy_encode");
        bundle.mirror_blit.encode_owned_hmd_to_openxr_and_staging(
            gpu,
            extent,
            targets.color_array_view(),
            targets.mirror_eye_view(),
            acquired_image.array_view(),
        )
    };
    let handles = &mut bundle.handles;
    let (finalize, rx) = handles.xr_session.build_projection_finalize(
        swapchain_handle,
        acquired_image.into_texture(),
        tick.predicted_display_time,
        projection_views,
        rect,
    );
    gpu.submit_frame_batch_with_xr_finalize(vec![final_copy_cmd], finalize);
    handles.xr_session.set_pending_finalize(rx);
    HmdSubmitOutcome::ProjectionQueued
}

/// Ensures the OpenXR frame, stereo swapchain, and owned HMD targets can submit HMD work.
fn ensure_hmd_submit_resources(
    gpu: &GpuContext,
    bundle: &mut XrSessionBundle,
    runtime: &impl XrFrameRenderer,
    tick: &OpenxrFrameTick,
) -> Option<(u32, u32)> {
    if let Some(reason) = multiview_submit_prereq_failure(gpu, bundle, runtime, tick) {
        log_hmd_submit_skip(reason);
        return None;
    }
    if !ensure_stereo_swapchain(bundle) {
        log_hmd_submit_failure(HmdSubmitSkipReason::StereoSwapchainUnavailable);
        return None;
    }
    let extent = if let Some(s) = bundle.stereo_swapchain.as_ref() {
        s.resolution
    } else {
        log_hmd_submit_failure(HmdSubmitSkipReason::StereoSwapchainMissingAfterEnsure);
        return None;
    };
    if !ensure_owned_hmd_targets(gpu, bundle, extent) {
        log_hmd_submit_failure(HmdSubmitSkipReason::OwnedHmdTargetsUnavailable);
        return None;
    }
    Some(extent)
}

/// Acquires, waits, and imports the current OpenXR stereo swapchain image.
fn acquire_imported_hmd_image(
    gpu: &GpuContext,
    sc: &XrStereoSwapchain,
) -> Option<XrAcquiredSwapchainImage> {
    let image_index = {
        profiling::scope!("xr::swapchain_acquire");
        match acquire_swapchain_image(gpu, &sc.handle) {
            Ok(i) => i,
            Err(e) => {
                gpu.record_gpu_flight_event(GpuFlightEventKind::OpenXrCall {
                    call: GpuFlightOpenXrCall::AcquireImage,
                    result: GpuFlightCallResult::failed_debug(e),
                    image_index: None,
                    predicted_display_time_nanos: None,
                });
                log_hmd_submit_failure_with_error(HmdSubmitSkipReason::SwapchainAcquireFailed, e);
                return None;
            }
        }
    };
    if !wait_for_acquired_swapchain_image(gpu, &sc.handle) {
        return None;
    }
    match sc.import_acquired_image(gpu.device().as_ref(), image_index) {
        Ok(image) => Some(image),
        Err(e) => {
            let _ = release_swapchain_image(gpu, &sc.handle);
            log_hmd_submit_failure_with_display_error(
                HmdSubmitSkipReason::SwapchainImageImportFailed,
                &e,
            );
            None
        }
    }
}

/// Returns the first unmet prerequisite for submitting an HMD projection layer.
fn multiview_submit_prereq_failure(
    gpu: &GpuContext,
    bundle: &XrSessionBundle,
    runtime: &impl XrFrameRenderer,
    tick: &OpenxrFrameTick,
) -> Option<HmdSubmitSkipReason> {
    let handles = &bundle.handles;
    if !handles.xr_session.session_running() {
        return Some(HmdSubmitSkipReason::SessionNotRunning);
    }
    if !handles.xr_session.is_visible() {
        return Some(HmdSubmitSkipReason::SessionNotVisible);
    }
    if !runtime.vr_active() {
        return Some(HmdSubmitSkipReason::RuntimeVrInactive);
    }
    if !gpu.device().features().contains(wgpu::Features::MULTIVIEW) {
        return Some(HmdSubmitSkipReason::MissingMultiviewFeature);
    }
    if !tick.should_render {
        return Some(HmdSubmitSkipReason::TickShouldNotRender);
    }
    if tick.views.len() < 2 {
        return Some(HmdSubmitSkipReason::LocatedViewCount {
            view_count: tick.views.len(),
        });
    }
    None
}

/// Acquires one OpenXR swapchain image while holding the shared Vulkan queue access gate.
///
/// Briefly locks the swapchain mutex around `xrAcquireSwapchainImage`. The lock is dropped
/// before `xrWaitSwapchainImage` so the long compositor wait does not block the driver
/// thread's `xrReleaseSwapchainImage` for an unrelated frame.
fn acquire_swapchain_image(
    gpu: &GpuContext,
    swapchain: &Mutex<xr::Swapchain<xr::Vulkan>>,
) -> Result<usize, xr::sys::Result> {
    let _gate = gpu.gpu_queue_access_gate().lock();
    let result = swapchain
        .lock()
        .acquire_image()
        .inspect_err(|e| {
            logger::warn!("OpenXR swapchain acquire_image failed: {e:?}");
        })
        .map(|i| i as usize);
    if let Ok(image_index) = &result {
        gpu.record_gpu_flight_event(GpuFlightEventKind::OpenXrCall {
            call: GpuFlightOpenXrCall::AcquireImage,
            result: GpuFlightCallResult::Ok,
            image_index: Some(*image_index as u32),
            predicted_display_time_nanos: None,
        });
    }
    result
}

/// Releases one OpenXR swapchain image while holding the shared Vulkan queue access gate.
///
/// Used by the failure recovery paths where no finalize work was queued for the driver
/// thread. The success path releases on the driver thread instead, see
/// [`crate::gpu::driver_thread::XrFinalizeKind::Projection`].
fn release_swapchain_image(
    gpu: &GpuContext,
    swapchain: &Mutex<xr::Swapchain<xr::Vulkan>>,
) -> Result<(), xr::sys::Result> {
    let _gate = gpu.gpu_queue_access_gate().lock();
    let result = swapchain.lock().release_image().inspect_err(|e| {
        logger::warn!("OpenXR swapchain release_image failed: {e:?}");
    });
    let flight_result = result.as_ref().map_or_else(
        |error| GpuFlightCallResult::failed_debug(*error),
        |()| GpuFlightCallResult::Ok,
    );
    gpu.record_gpu_flight_event(GpuFlightEventKind::OpenXrCall {
        call: GpuFlightOpenXrCall::ReleaseImage,
        result: flight_result,
        image_index: None,
        predicted_display_time_nanos: None,
    });
    result
}

/// Waits for an acquired OpenXR swapchain image and releases it on failure.
fn wait_for_acquired_swapchain_image(
    gpu: &GpuContext,
    swapchain: &Mutex<xr::Swapchain<xr::Vulkan>>,
) -> bool {
    profiling::scope!("xr::swapchain_wait_image");
    let wd = EndFrameWatchdog::arm(WAIT_IMAGE_WATCHDOG_TIMEOUT, "xr::wait_image");
    let res = swapchain.lock().wait_image(xr::Duration::INFINITE);
    wd.disarm();
    if let Err(e) = res {
        gpu.record_gpu_flight_event(GpuFlightEventKind::OpenXrCall {
            call: GpuFlightOpenXrCall::WaitImage,
            result: GpuFlightCallResult::failed_debug(e),
            image_index: None,
            predicted_display_time_nanos: None,
        });
        // OpenXR requires every successful `acquire_image` to be paired with
        // `release_image`, even when `wait_image` fails. Without this release the
        // runtime considers the image still in flight and `xrEndFrame` blocks until
        // the swapchain is destroyed.
        let _ = release_swapchain_image(gpu, swapchain);
        log_hmd_submit_failure_with_error(HmdSubmitSkipReason::SwapchainWaitFailed, e);
        return false;
    }
    gpu.record_gpu_flight_event(GpuFlightEventKind::OpenXrCall {
        call: GpuFlightOpenXrCall::WaitImage,
        result: GpuFlightCallResult::Ok,
        image_index: None,
        predicted_display_time_nanos: None,
    });
    true
}

/// Logs a non-error HMD submit skip at info level with throttling.
fn log_hmd_submit_skip(reason: HmdSubmitSkipReason) {
    if let Some(occurrence) = HMD_SUBMIT_SKIP_LOG.should_log(4, 128) {
        logger::info!("OpenXR HMD submit skipped: reason={reason} occurrence={occurrence}");
    }
}

/// Logs an HMD submit failure at warn level with throttling.
fn log_hmd_submit_failure(reason: HmdSubmitSkipReason) {
    if let Some(occurrence) = HMD_SUBMIT_FAILURE_LOG.should_log(8, 64) {
        logger::warn!("OpenXR HMD submit failed: reason={reason} occurrence={occurrence}");
    }
}

/// Logs an HMD submit failure with an OpenXR result at warn level with throttling.
fn log_hmd_submit_failure_with_error(reason: HmdSubmitSkipReason, error: xr::sys::Result) {
    if let Some(occurrence) = HMD_SUBMIT_FAILURE_LOG.should_log(8, 64) {
        logger::warn!(
            "OpenXR HMD submit failed: reason={reason} error={error:?} occurrence={occurrence}"
        );
    }
}

/// Logs an HMD submit failure with a displayable error at warn level with throttling.
fn log_hmd_submit_failure_with_display_error(
    reason: HmdSubmitSkipReason,
    error: &dyn fmt::Display,
) {
    if let Some(occurrence) = HMD_SUBMIT_FAILURE_LOG.should_log(8, 64) {
        logger::warn!(
            "OpenXR HMD submit failed: reason={reason} error={error} occurrence={occurrence}"
        );
    }
}

/// Returns `Some([left, right])` when `views` carries the standard stereo pair OpenXR
/// reports for `PRIMARY_STEREO`; `None` otherwise so the caller can fall back to the
/// empty-end-frame path.
fn stereo_views(views: &[xr::View]) -> Option<[xr::View; 2]> {
    if views.len() < 2 {
        return None;
    }
    Some([views[0], views[1]])
}
