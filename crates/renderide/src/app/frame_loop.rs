//! Desktop vs OpenXR frame submission helpers for [`super::renderide_app::RenderideApp`].
//!
//! Keeps [`super::renderide_app::RenderideApp::tick_frame`] readable while preserving ordering: OpenXR
//! `wait_frame` / `locate_views` before lock-step [`crate::runtime::RendererRuntime::pre_frame`].

use winit::window::Window;

use crate::gpu::{GpuContext, VrMirrorBlitResources};
use crate::present::PresentClearError;
use crate::runtime::RendererRuntime;
use crate::xr::{OpenxrFrameTick, XrSessionBundle, XrWgpuHandles};

/// Runs OpenXR `wait_frame` + view pose for stereo uniforms and IPC head tracking.
pub(crate) fn begin_openxr_frame_tick(
    handles: &mut XrWgpuHandles,
    runtime: &mut RendererRuntime,
) -> Option<OpenxrFrameTick> {
    crate::xr::openxr_begin_frame_tick(handles, runtime)
}

/// Renders to the HMD multiview swapchain when VR is active; returns whether a projection layer was submitted.
pub(crate) fn try_hmd_multiview_submit(
    gpu: &mut GpuContext,
    bundle: &mut XrSessionBundle,
    runtime: &mut RendererRuntime,
    window: &Window,
    tick: &OpenxrFrameTick,
) -> bool {
    crate::xr::try_openxr_hmd_multiview_submit(gpu, bundle, runtime, window, tick)
}

/// Blits the last HMD eye staging texture to the window (VR mirror); no full scene render.
pub(crate) fn present_vr_mirror_blit(
    gpu: &mut GpuContext,
    window: &Window,
    mirror_blit: &mut VrMirrorBlitResources,
) -> Result<(), PresentClearError> {
    mirror_blit.present_staging_to_surface(gpu, window)
}
