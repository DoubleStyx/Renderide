//! HMD multiview submission into the OpenXR stereo swapchain.

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

/// Renders to the OpenXR stereo swapchain and calls [`crate::xr::session::XrSessionState::end_frame_projection`].
///
/// Uses the same [`xr::FrameState`] as [`openxr_begin_frame_tick`] -- no second `wait_frame`.
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
        match acquire_swapchain_image(gpu, &mut sc.handle) {
            Ok(i) => i,
            Err(_) => return false,
        }
    };
    {
        profiling::scope!("xr::swapchain_wait_image");
        let wd = EndFrameWatchdog::arm(WAIT_IMAGE_WATCHDOG_TIMEOUT, "wait_image");
        let res = sc.handle.wait_image(xr::Duration::INFINITE);
        wd.disarm();
        if res.is_err() {
            // OpenXR requires every successful `acquire_image` to be paired with
            // `release_image`, even when `wait_image` fails. Without this release the
            // runtime considers the image still in flight and `xrEndFrame` blocks until
            // the swapchain is destroyed.
            let _ = release_swapchain_image(gpu, &mut sc.handle);
            return false;
        }
    }
    let Some(color_view) = sc.color_view_for_image(image_index) else {
        let _ = release_swapchain_image(gpu, &mut sc.handle);
        return false;
    };
    let Some(stereo_depth) = bundle.stereo_depth.as_ref() else {
        logger::debug!("OpenXR stereo depth texture missing after resize");
        let _ = release_swapchain_image(gpu, &mut sc.handle);
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
    let views_ref = tick.views.as_slice();
    let handles = &mut bundle.handles;
    // Unified submit: HMD stereo + every active secondary RT in one `execute_multi_view_frame`
    // call. The HMD view replaces the main camera for this tick.
    {
        profiling::scope!("xr::submit_hmd_view");
        if runtime.submit_hmd_view(gpu, ext).is_err() {
            let _ = release_swapchain_image(gpu, &mut sc.handle);
            return false;
        }
    }
    if let Some(layer_view) = sc.color_layer_view_for_image(image_index, VR_MIRROR_EYE_LAYER) {
        profiling::scope!("xr::mirror_staging_submit");
        bundle
            .mirror_blit
            .submit_eye_to_staging(gpu, extent, &layer_view);
    }
    // Ensure all queued work touching the OpenXR swapchain is submitted before release/end_frame.
    {
        profiling::scope!("xr::flush_driver_before_release");
        gpu.flush_driver();
    };
    {
        profiling::scope!("xr::swapchain_release");
        if release_swapchain_image(gpu, &mut sc.handle).is_err() {
            return false;
        }
    }
    if handles
        .xr_session
        .end_frame_projection(
            tick.predicted_display_time,
            &sc.handle,
            views_ref,
            rect,
            gpu.gpu_queue_access_gate(),
        )
        .is_err()
    {
        return false;
    }
    true
}
