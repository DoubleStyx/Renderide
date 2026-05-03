//! App-loop XR session bundle and cached frame tick state.

use crate::gpu::VrMirrorBlitResources;
use crate::xr::{XrStereoSwapchain, XrWgpuHandles};
use openxr as xr;

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

impl Drop for XrSessionBundle {
    fn drop(&mut self) {
        // Defensive: in normal shutdown the parent `RenderTarget` drops `gpu` (and with it
        // `DriverThread`, which joins the worker draining queued finalize) before the
        // session bundle. If a future code path swaps out a bundle while the driver is
        // still processing a finalize for it, the in-flight `xrEndFrame` would race with
        // the drop of `XrSessionState::frame_stream`. Waiting here closes that hole.
        self.handles.xr_session.await_finalize_pending();
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
