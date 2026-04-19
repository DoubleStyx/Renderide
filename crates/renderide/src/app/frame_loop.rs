//! Desktop vs OpenXR frame submission helpers for [`super::renderide_app::RenderideApp`].
//!
//! Keeps [`super::renderide_app::RenderideApp::tick_frame`] readable while preserving ordering: OpenXR
//! `wait_frame` / `locate_views` before lock-step [`crate::runtime::RendererRuntime::pre_frame`].

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
    tick: &OpenxrFrameTick,
) -> bool {
    crate::xr::try_openxr_hmd_multiview_submit(gpu, bundle, runtime, tick)
}

/// Blits the last HMD eye staging texture to the window (VR mirror); no full scene render.
///
/// `overlay` runs on the same encoder after the mirror pass (e.g. Dear ImGui composite with `LoadOp::Load`).
pub(crate) fn present_vr_mirror_blit<F, E>(
    gpu: &mut GpuContext,
    mirror_blit: &mut VrMirrorBlitResources,
    overlay: F,
) -> Result<(), PresentClearError>
where
    F: FnOnce(&mut wgpu::CommandEncoder, &wgpu::TextureView, &mut GpuContext) -> Result<(), E>,
    E: std::fmt::Display,
{
    mirror_blit.present_staging_to_surface_overlay(gpu, overlay)
}
