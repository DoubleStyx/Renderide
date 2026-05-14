//! HMD multiview submission into the OpenXR stereo swapchain.

use std::sync::Arc;
use std::time::Duration;

use crate::diagnostics::log_throttle::LogThrottle;
use crate::gpu::driver_thread::BlockingCallWatchdog as EndFrameWatchdog;
use crate::gpu::{GpuContext, VR_MIRROR_EYE_LAYER};
use crate::render_graph::ExternalFrameTargets;
use crate::xr::{XR_COLOR_FORMAT, XrFrameRenderer};
use openxr as xr;
use parking_lot::Mutex;

use super::resources::{ensure_stereo_depth_texture, ensure_stereo_swapchain};
use super::types::{OpenxrFrameTick, XrSessionBundle};

/// Deadline for a single `xrWaitSwapchainImage` call before the watchdog logs a compositor stall.
///
/// Observation only: the call keeps its original `xr::Duration::INFINITE` because openxr 0.21
/// swallows `XR_TIMEOUT_EXPIRED` (returns `Ok(())` identically to success), making a bounded
/// timeout indistinguishable from a real image release.
const WAIT_IMAGE_WATCHDOG_TIMEOUT: Duration = Duration::from_millis(500);
static HMD_SUBMIT_SKIP_LOG: LogThrottle = LogThrottle::new();

/// Renders to the OpenXR stereo swapchain and queues `xrReleaseSwapchainImage` + `xrEndFrame`
/// onto the driver thread.
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
) -> bool {
    if !multiview_submit_prereqs(gpu, bundle, runtime, tick) {
        log_hmd_submit_skip("prerequisites not met");
        return false;
    }
    if !ensure_stereo_swapchain(bundle) {
        log_hmd_submit_skip("stereo swapchain unavailable");
        return false;
    }
    let extent = if let Some(s) = bundle.stereo_swapchain.as_ref() {
        s.resolution
    } else {
        log_hmd_submit_skip("stereo swapchain missing after ensure");
        return false;
    };
    if !ensure_stereo_depth_texture(gpu, bundle, extent) {
        log_hmd_submit_skip("stereo depth texture unavailable");
        return false;
    }
    let Some(sc) = bundle.stereo_swapchain.as_ref() else {
        log_hmd_submit_skip("stereo swapchain missing before acquire");
        return false;
    };
    let image_index = {
        profiling::scope!("xr::swapchain_acquire");
        match acquire_swapchain_image(gpu, &sc.handle) {
            Ok(i) => i,
            Err(e) => {
                log_hmd_submit_skip_with_error("swapchain acquire_image failed", e);
                return false;
            }
        }
    };
    if !wait_for_acquired_swapchain_image(gpu, &sc.handle) {
        return false;
    }
    let acquired_image = match sc.import_acquired_image(gpu.device().as_ref(), image_index) {
        Ok(image) => image,
        Err(e) => {
            let _ = release_swapchain_image(gpu, &sc.handle);
            log_hmd_submit_skip_with_display_error("swapchain image import failed", &e);
            return false;
        }
    };
    let Some(stereo_depth) = bundle.stereo_depth.as_ref() else {
        logger::debug!("OpenXR stereo depth texture missing after resize");
        let _ = release_swapchain_image(gpu, &sc.handle);
        log_hmd_submit_skip("stereo depth missing after resize");
        return false;
    };
    let ext = ExternalFrameTargets {
        color_view: acquired_image.array_view(),
        depth_texture: &stereo_depth.0,
        depth_view: &stereo_depth.1,
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
    let handles = &mut bundle.handles;
    // Unified submit: HMD stereo + every active secondary RT in one `execute_multi_view_frame`
    // call. The HMD view replaces the main camera for this tick.
    {
        profiling::scope!("xr::submit_hmd_view");
        if runtime.submit_hmd_view(gpu, ext).is_err() {
            // Synchronous release is correct here: no finalize work was queued for the
            // driver thread, so `xrReleaseSwapchainImage` cannot be deferred.
            let _ = release_swapchain_image(gpu, &sc.handle);
            log_hmd_submit_skip("render graph submit_hmd_view failed");
            return false;
        }
    }
    let Some(projection_views) = stereo_views(&tick.views) else {
        // Locate-views returned <2 views; fall back to an empty end-frame on the driver.
        let (finalize, rx) = handles
            .xr_session
            .build_empty_finalize(tick.predicted_display_time);
        gpu.submit_finalize_only(finalize);
        handles.xr_session.set_pending_finalize(rx);
        return true;
    };
    let mirror_layer_view = acquired_image.color_layer_view(VR_MIRROR_EYE_LAYER);
    let (finalize, rx) = handles.xr_session.build_projection_finalize(
        Arc::clone(&sc.handle),
        acquired_image.into_texture(),
        tick.predicted_display_time,
        projection_views,
        rect,
    );
    if let Some(layer_view) = mirror_layer_view {
        // Attach finalize to the mirror staging blit so the driver runs both submits and
        // then `xrReleaseSwapchainImage` + `xrEndFrame` in FIFO order with no main-thread
        // wait between them.
        profiling::scope!("xr::mirror_staging_submit");
        bundle
            .mirror_blit
            .submit_eye_to_staging_with_finalize(gpu, extent, &layer_view, finalize);
    } else {
        // No mirror layer this frame; push the finalize on its own batch.
        gpu.submit_finalize_only(finalize);
    }
    handles.xr_session.set_pending_finalize(rx);
    true
}

/// Returns `true` when the session/runtime/GPU/tick state can submit an HMD projection layer.
fn multiview_submit_prereqs(
    gpu: &GpuContext,
    bundle: &XrSessionBundle,
    runtime: &impl XrFrameRenderer,
    tick: &OpenxrFrameTick,
) -> bool {
    let handles = &bundle.handles;
    if !handles.xr_session.session_running() {
        return false;
    }
    if !handles.xr_session.is_visible() {
        return false;
    }
    if !runtime.vr_active() {
        return false;
    }
    if !gpu.device().features().contains(wgpu::Features::MULTIVIEW) {
        return false;
    }
    if !tick.should_render || tick.views.len() < 2 {
        return false;
    }
    true
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
    swapchain
        .lock()
        .acquire_image()
        .inspect_err(|e| {
            logger::warn!("OpenXR swapchain acquire_image failed: {e:?}");
        })
        .map(|i| i as usize)
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
    swapchain.lock().release_image().inspect_err(|e| {
        logger::warn!("OpenXR swapchain release_image failed: {e:?}");
    })
}

fn wait_for_acquired_swapchain_image(
    gpu: &GpuContext,
    swapchain: &Mutex<xr::Swapchain<xr::Vulkan>>,
) -> bool {
    profiling::scope!("xr::swapchain_wait_image");
    let wd = EndFrameWatchdog::arm(WAIT_IMAGE_WATCHDOG_TIMEOUT, "xr::wait_image");
    let res = swapchain.lock().wait_image(xr::Duration::INFINITE);
    wd.disarm();
    if let Err(e) = res {
        // OpenXR requires every successful `acquire_image` to be paired with
        // `release_image`, even when `wait_image` fails. Without this release the
        // runtime considers the image still in flight and `xrEndFrame` blocks until
        // the swapchain is destroyed.
        let _ = release_swapchain_image(gpu, swapchain);
        log_hmd_submit_skip_with_error("swapchain wait_image failed", e);
        return false;
    }
    true
}

fn log_hmd_submit_skip(reason: &'static str) {
    if let Some(occurrence) = HMD_SUBMIT_SKIP_LOG.should_log(4, 128) {
        logger::debug!("OpenXR HMD submit skipped: reason={reason} occurrence={occurrence}");
    }
}

fn log_hmd_submit_skip_with_error(reason: &'static str, error: xr::sys::Result) {
    if let Some(occurrence) = HMD_SUBMIT_SKIP_LOG.should_log(4, 128) {
        logger::debug!(
            "OpenXR HMD submit skipped: reason={reason} error={error:?} occurrence={occurrence}"
        );
    }
}

fn log_hmd_submit_skip_with_display_error(reason: &'static str, error: &dyn std::fmt::Display) {
    if let Some(occurrence) = HMD_SUBMIT_SKIP_LOG.should_log(4, 128) {
        logger::debug!(
            "OpenXR HMD submit skipped: reason={reason} error={error} occurrence={occurrence}"
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
