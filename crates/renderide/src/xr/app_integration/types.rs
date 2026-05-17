//! App-loop XR session bundle and cached frame tick state.

use crate::gpu::VrMirrorBlitResources;
use crate::xr::{XrStereoSwapchain, XrWgpuHandles};
use openxr as xr;

/// App-loop ownership for the OpenXR GPU path: Vulkan/wgpu [`XrWgpuHandles`], lazily created stereo
/// swapchain, owned HMD render targets, and the desktop mirror blit ([`VrMirrorBlitResources`]).
///
/// Populated when [`crate::xr::init_wgpu_openxr`] succeeds and the window uses the shared device; kept
/// together for [`openxr_begin_frame_tick`] and [`try_openxr_hmd_multiview_submit`].
pub struct XrSessionBundle {
    /// Bootstrap handles (instance, session, device, queue, input).
    pub handles: XrWgpuHandles,
    /// Stereo array swapchain; created on first successful HMD frame path.
    pub stereo_swapchain: Option<XrStereoSwapchain>,
    /// Renderer-owned stereo color and depth targets used by the HMD graph.
    pub hmd_targets: Option<XrOwnedHmdTargets>,
    /// Left-eye staging blit to the desktop mirror surface.
    pub mirror_blit: VrMirrorBlitResources,
}

impl XrSessionBundle {
    /// Wraps successful OpenXR bootstrap handles; swapchain and owned HMD targets are filled when
    /// the multiview path runs.
    pub fn new(handles: XrWgpuHandles) -> Self {
        Self {
            handles,
            stereo_swapchain: None,
            hmd_targets: None,
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
/// Stereo view data is consumed by the multiview HMD path and host IPC; the desktop window mirror is
/// a GPU blit of the renderer-owned left-eye color target, not a second camera render.
pub struct OpenxrFrameTick {
    /// Predicted display time for this frame (input sampling, `end_frame`).
    pub predicted_display_time: xr::Time,
    /// Whether the runtime expects rendering work this frame.
    pub should_render: bool,
    /// Stereo views from `locate_views` (may be empty when `should_render` is false).
    pub views: Vec<xr::View>,
}

/// Renderer-owned stereo color/depth target set for the HMD render graph.
pub struct XrOwnedHmdTargets {
    /// Two-layer color texture written by the HMD graph and sampled by final blits.
    color_texture: wgpu::Texture,
    /// Two-layer color view used as the graph's multiview render target and final blit source.
    color_array_view: wgpu::TextureView,
    /// Single-layer left-eye view used by the desktop VR mirror staging blit.
    mirror_eye_view: wgpu::TextureView,
    /// Two-layer depth texture used by the HMD graph.
    depth_texture: wgpu::Texture,
    /// Two-layer depth-stencil view used by the HMD graph.
    depth_view: wgpu::TextureView,
    /// Per-eye pixel extent shared by color and depth attachments.
    extent_px: (u32, u32),
}

impl XrOwnedHmdTargets {
    /// Builds a new renderer-owned HMD target set from pre-created texture handles.
    pub(super) fn new(
        color_texture: wgpu::Texture,
        color_array_view: wgpu::TextureView,
        mirror_eye_view: wgpu::TextureView,
        depth_texture: wgpu::Texture,
        depth_view: wgpu::TextureView,
        extent_px: (u32, u32),
    ) -> Self {
        Self {
            color_texture,
            color_array_view,
            mirror_eye_view,
            depth_texture,
            depth_view,
            extent_px,
        }
    }

    /// Returns the two-layer color view used by the multiview render graph.
    pub(super) fn color_array_view(&self) -> &wgpu::TextureView {
        &self.color_array_view
    }

    /// Returns the single-layer left-eye color view used by the desktop mirror.
    pub(super) fn mirror_eye_view(&self) -> &wgpu::TextureView {
        &self.mirror_eye_view
    }

    /// Returns the backing depth texture used by graph depth/snapshot helpers.
    pub(super) fn depth_texture(&self) -> &wgpu::Texture {
        &self.depth_texture
    }

    /// Returns the two-layer depth view used by the multiview render graph.
    pub(super) fn depth_view(&self) -> &wgpu::TextureView {
        &self.depth_view
    }

    /// Returns `true` when all target attachments still match `extent_px`.
    pub(super) fn matches_extent(&self, extent_px: (u32, u32)) -> bool {
        self.extent_px == extent_px
            && self.color_texture.size().width == extent_px.0
            && self.color_texture.size().height == extent_px.1
            && self.color_texture.size().depth_or_array_layers == crate::xr::XR_VIEW_COUNT
            && self.depth_texture.size().width == extent_px.0
            && self.depth_texture.size().height == extent_px.1
            && self.depth_texture.size().depth_or_array_layers == crate::xr::XR_VIEW_COUNT
    }
}
