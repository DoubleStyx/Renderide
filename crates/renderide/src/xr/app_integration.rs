//! OpenXR helpers used by the winit [`crate::app::RenderideApp`] loop: frame tick state and HMD multiview submission.

use crate::gpu::{GpuContext, GpuQueueAccessGate, VrMirrorBlitResources, VR_MIRROR_EYE_LAYER};
use crate::render_graph::{
    effective_head_output_clip_planes, ExternalFrameTargets, StereoViewMatrices,
};
use crate::xr::{
    create_stereo_depth_texture, XrFrameRenderer, XrHostCameraSync, XrStereoSwapchain,
    XrWgpuHandles, XR_COLOR_FORMAT, XR_VIEW_COUNT,
};
use openxr as xr;
use std::time::Duration;

use super::session::end_frame_watchdog::EndFrameWatchdog;

/// Deadline for a single `xrWaitSwapchainImage` call before the watchdog logs a compositor stall.
///
/// Observation only: the call keeps its original `xr::Duration::INFINITE` because openxr 0.21
/// swallows `XR_TIMEOUT_EXPIRED` (returns `Ok(())` identically to success), making a bounded
/// timeout indistinguishable from a real image release.
const WAIT_IMAGE_WATCHDOG_TIMEOUT: Duration = Duration::from_millis(500);

/// App-loop ownership for the OpenXR GPU path: Vulkan/wgpu [`XrWgpuHandles`], lazily created stereo
/// swapchain and depth targets, and the desktop mirror blit ([`VrMirrorBlitResources`]).
///
/// Populated when [`crate::xr::init_wgpu_openxr`] succeeds and the window uses the shared device; kept
/// together for [`openxr_begin_frame_tick`] and [`try_openxr_hmd_multiview_submit`].
pub struct XrSessionBundle {
    /// Bootstrap handles (instance, session, device, queue, input).
    pub handles: XrWgpuHandles,
    /// Stereo array swapchain; created on first successful HMD frame path.
    pub stereo_swapchain: Option<XrStereoSwapchain>,
    /// Depth texture matching the stereo color resolution and layer count.
    pub stereo_depth: Option<(wgpu::Texture, wgpu::TextureView)>,
    /// Left-eye staging blit to the desktop mirror surface.
    pub mirror_blit: VrMirrorBlitResources,
}

impl XrSessionBundle {
    /// Wraps successful OpenXR bootstrap handles; swapchain and depth are filled when the multiview path runs.
    pub fn new(handles: XrWgpuHandles) -> Self {
        Self {
            handles,
            stereo_swapchain: None,
            stereo_depth: None,
            mirror_blit: VrMirrorBlitResources::new(),
        }
    }
}

/// Cached OpenXR frame state after a single `wait_frame` (no second wait per tick).
///
/// Stereo view data is consumed by the multiview HMD path and host IPC; the desktop window mirror
/// is a GPU blit of the left eye (see [`crate::gpu::VrMirrorBlitResources`]), not a second camera render.
pub struct OpenxrFrameTick {
    /// Predicted display time for this frame (input sampling, `end_frame`).
    pub predicted_display_time: xr::Time,
    /// Whether the runtime expects rendering work this frame.
    pub should_render: bool,
    /// Stereo views from `locate_views` (may be empty when `should_render` is false).
    pub views: Vec<xr::View>,
}

/// Single `wait_frame` + `locate_views` for stereo uniforms; used for both mirror and HMD paths.
pub fn openxr_begin_frame_tick(
    handles: &mut XrWgpuHandles,
    runtime: &mut impl XrHostCameraSync,
    gpu_queue_access_gate: &GpuQueueAccessGate,
) -> Option<OpenxrFrameTick> {
    profiling::scope!("xr::begin_frame_tick");
    {
        profiling::scope!("xr::poll_events");
        match handles.xr_session.poll_events() {
            Ok(_) => {}
            Err(e) => logger::warn!("OpenXR poll_events failed: {e:?}"),
        }
    }
    if handles.xr_session.exit_requested() {
        // Exit is driven by the app loop reading `exit_requested()`; here we just skip starting a
        // new frame so we don't `xrBeginFrame` on a terminating session.
        return None;
    }
    let fs = {
        profiling::scope!("xr::wait_frame");
        match handles.xr_session.wait_frame(gpu_queue_access_gate) {
            Ok(Some(state)) => state,
            Ok(None) => return None,
            Err(e) => {
                logger::warn!("OpenXR wait_frame failed: {e:?}");
                runtime.note_openxr_wait_frame_failed();
                return None;
            }
        }
    };
    let views = if fs.should_render {
        profiling::scope!("xr::locate_views");
        match handles.xr_session.locate_views(fs.predicted_display_time) {
            Ok(v) => v,
            Err(e) => {
                logger::warn!("OpenXR locate_views failed: {e:?}");
                runtime.note_openxr_locate_views_failed();
                Vec::new()
            }
        }
    } else {
        Vec::new()
    };
    if views.len() >= 2 {
        if runtime.vr_active() {
            let (near, far) = effective_head_output_clip_planes(
                runtime.near_clip(),
                runtime.far_clip(),
                runtime.output_device(),
                runtime.scene_root_scale_for_clip(),
            );
            let center_pose = crate::xr::headset_center_pose_from_stereo_views(&views);
            let world_from_tracking = runtime.world_from_tracking(center_pose);
            runtime.set_head_output_transform(world_from_tracking);
            let l = crate::xr::view_projection_from_xr_view_aligned(
                &views[0],
                near,
                far,
                world_from_tracking,
            );
            let r = crate::xr::view_projection_from_xr_view_aligned(
                &views[1],
                near,
                far,
                world_from_tracking,
            );
            let vl = crate::xr::view_from_xr_view_aligned(&views[0], world_from_tracking);
            let vr_view = crate::xr::view_from_xr_view_aligned(&views[1], world_from_tracking);
            runtime.set_stereo(Some(StereoViewMatrices {
                view_proj: (l, r),
                view_only: (vl, vr_view),
            }));
            return Some(OpenxrFrameTick {
                predicted_display_time: fs.predicted_display_time,
                should_render: fs.should_render,
                views,
            });
        }
        // Desktop (`!vr_active`): keep [`HostCameraFrame::head_output_transform`] from
        // [`RendererRuntime::on_frame_submit`](crate::runtime::RendererRuntime) (host `root_transform`), matching Unity
        // `HeadOutput.UpdatePositioning`. OpenXR still supplies views for IPC pose.
        return Some(OpenxrFrameTick {
            predicted_display_time: fs.predicted_display_time,
            should_render: fs.should_render,
            views,
        });
    }
    Some(OpenxrFrameTick {
        predicted_display_time: fs.predicted_display_time,
        should_render: fs.should_render,
        views,
    })
}

/// Acquires one OpenXR swapchain image while holding the shared Vulkan queue access gate.
fn acquire_swapchain_image(
    gpu: &GpuContext,
    swapchain: &mut xr::Swapchain<xr::Vulkan>,
) -> Result<usize, xr::sys::Result> {
    let _gate = gpu.gpu_queue_access_gate().lock();
    swapchain.acquire_image().map(|i| i as usize)
}

/// Releases one OpenXR swapchain image while holding the shared Vulkan queue access gate.
fn release_swapchain_image(
    gpu: &GpuContext,
    swapchain: &mut xr::Swapchain<xr::Vulkan>,
) -> Result<(), xr::sys::Result> {
    let _gate = gpu.gpu_queue_access_gate().lock();
    swapchain.release_image()
}

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
    // Compositor is not displaying this app right now. Skip real projection-layer submission;
    // the outer path closes the open frame via `end_frame_if_open` in `present_and_diagnostics`.
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

/// Creates the lazy stereo swapchain on first successful HMD path.
fn ensure_stereo_swapchain(bundle: &mut XrSessionBundle) -> bool {
    profiling::scope!("xr::ensure_stereo_swapchain");
    if bundle.stereo_swapchain.is_some() {
        return true;
    }
    let handles = &bundle.handles;
    let sys_id = handles.xr_system_id;
    let session = handles.xr_session.xr_vulkan_session();
    let inst = handles.xr_session.xr_instance();
    let dev = handles.device.as_ref();
    // SAFETY: `dev` is the wgpu device that was constructed in `xr::bootstrap` from the same
    // `VkDevice` that owns `session`, so the `VkDevice` identity required by `XrStereoSwapchain`
    // holds.
    let res = unsafe { XrStereoSwapchain::new(session, inst, sys_id, dev) };
    match res {
        Ok(sc) => {
            logger::info!(
                "OpenXR swapchain {}×{} (stereo array)",
                sc.resolution.0,
                sc.resolution.1
            );
            bundle.stereo_swapchain = Some(sc);
            true
        }
        Err(e) => {
            logger::debug!("OpenXR swapchain not created: {e}");
            false
        }
    }
}

/// Resizes the wgpu depth texture when the swapchain resolution or layer count changes.
fn ensure_stereo_depth_texture(
    gpu: &mut GpuContext,
    bundle: &mut XrSessionBundle,
    extent: (u32, u32),
) -> bool {
    profiling::scope!("xr::ensure_stereo_depth_texture");
    let need_new_depth = bundle
        .stereo_depth
        .as_ref()
        .map(|(tex, _)| {
            tex.size().width != extent.0
                || tex.size().height != extent.1
                || tex.size().depth_or_array_layers != XR_VIEW_COUNT
        })
        .unwrap_or(true);
    if need_new_depth {
        let limits = gpu.limits().clone();
        bundle.stereo_depth = create_stereo_depth_texture(gpu.device().as_ref(), &limits, extent);
    }
    bundle.stereo_depth.is_some()
}

/// Renders to the OpenXR stereo swapchain and calls [`crate::xr::session::XrSessionState::end_frame_projection`].
///
/// Uses the same [`xr::FrameState`] as [`openxr_begin_frame_tick`] — no second `wait_frame`.
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
    }
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
