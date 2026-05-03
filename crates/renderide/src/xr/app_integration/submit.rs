//! HMD multiview submission into the OpenXR stereo swapchain.

use std::sync::Arc;
use std::time::Duration;

use crate::gpu::{GpuContext, VR_MIRROR_EYE_LAYER};
use crate::render_graph::ExternalFrameTargets;
use crate::xr::{XR_COLOR_FORMAT, XrFrameRenderer};
use openxr as xr;

use super::super::session::end_frame_watchdog::EndFrameWatchdog;
use super::planning::multiview_submit_prereqs;
use super::resources::{ensure_stereo_depth_texture, ensure_stereo_swapchain};
use super::swapchain_access::{acquire_swapchain_image, release_swapchain_image};
use super::types::{OpenxrFrameTick, XrSessionBundle};

/// Deadline for a single `xrWaitSwapchainImage` call before the watchdog logs a compositor stall.
///
/// Observation only: the call keeps its original `xr::Duration::INFINITE` because openxr 0.21
/// swallows `XR_TIMEOUT_EXPIRED` (returns `Ok(())` identically to success), making a bounded
/// timeout indistinguishable from a real image release.
const WAIT_IMAGE_WATCHDOG_TIMEOUT: Duration = Duration::from_millis(500);

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
        return false;
    }
    if !ensure_stereo_swapchain(bundle) {
        return false;
    }
    let extent = match bundle.stereo_swapchain.as_ref() {
        Some(s) => s.resolution,
        None => return false,
    };
    if !ensure_stereo_depth_texture(gpu, bundle, extent) {
        return false;
    }
    let Some(sc) = bundle.stereo_swapchain.as_mut() else {
        return false;
    };
    let image_index = {
        profiling::scope!("xr::swapchain_acquire");
        match acquire_swapchain_image(gpu, &sc.handle) {
            Ok(i) => i,
            Err(_) => return false,
        }
    };
    {
        profiling::scope!("xr::swapchain_wait_image");
        let wd = EndFrameWatchdog::arm(WAIT_IMAGE_WATCHDOG_TIMEOUT, "wait_image");
        let res = sc.handle.lock().wait_image(xr::Duration::INFINITE);
        wd.disarm();
        if res.is_err() {
            // OpenXR requires every successful `acquire_image` to be paired with
            // `release_image`, even when `wait_image` fails. Without this release the
            // runtime considers the image still in flight and `xrEndFrame` blocks until
            // the swapchain is destroyed.
            let _ = release_swapchain_image(gpu, &sc.handle);
            return false;
        }
    }
    let Some(color_view) = sc.color_view_for_image(image_index) else {
        let _ = release_swapchain_image(gpu, &sc.handle);
        return false;
    };
    let Some(stereo_depth) = bundle.stereo_depth.as_ref() else {
        logger::debug!("OpenXR stereo depth texture missing after resize");
        let _ = release_swapchain_image(gpu, &sc.handle);
        return false;
    };
    let ext = ExternalFrameTargets {
        color_view,
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
    let (finalize, rx) = handles.xr_session.build_projection_finalize(
        Arc::clone(&sc.handle),
        tick.predicted_display_time,
        projection_views,
        rect,
    );
    if let Some(layer_view) = sc.color_layer_view_for_image(image_index, VR_MIRROR_EYE_LAYER) {
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

/// Returns `Some([left, right])` when `views` carries the standard stereo pair OpenXR
/// reports for `PRIMARY_STEREO`; `None` otherwise so the caller can fall back to the
/// empty-end-frame path.
fn stereo_views(views: &[xr::View]) -> Option<[xr::View; 2]> {
    if views.len() < 2 {
        return None;
    }
    Some([views[0], views[1]])
}
