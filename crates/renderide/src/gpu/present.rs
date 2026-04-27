//! Swapchain presentation: surface acquire helpers and a minimal clear pass (no mesh or UI draws).
//!
//! Serves as the minimal integration test for surface acquire, encoder submission, and present.
//! The render graph reuses [`acquire_surface_outcome_traced`] and [`record_swapchain_clear_pass`].

use crate::gpu::GpuContext;

/// Clear color used for the skeleton swapchain clear (dark blue).
pub const SWAPCHAIN_CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 0.02,
    g: 0.05,
    b: 0.12,
    a: 1.0,
};

/// Failure to obtain a presentable surface texture after recovery attempts.
#[derive(Debug, thiserror::Error)]
#[error("could not acquire surface texture ({status:?})")]
pub struct PresentClearError {
    /// Status from [`wgpu::Surface::get_current_texture`] after reconfiguration.
    pub status: wgpu::CurrentSurfaceTexture,
}

/// Result of attempting to acquire the swapchain for one frame.
#[derive(Debug)]
pub enum SurfaceFrameOutcome {
    /// Timeout or occluded: skip recording and present for this frame.
    Skip,
    /// Validation error: swapchain was reconfigured; skip this frame.
    Reconfigured,
    /// Ready to record; caller must submit and [`wgpu::SurfaceTexture::present`].
    Acquired(wgpu::SurfaceTexture),
}

/// Static Tracy labels for desktop surface acquisition sites.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SurfaceAcquireTrace {
    /// Render graph acquisition for the main desktop backbuffer.
    DesktopGraph,
    /// VR mirror acquisition for blitting the latest HMD eye to the desktop window.
    VrMirror,
    /// VR clear/fallback acquisition when no mirror image is available or mirror blit fails.
    VrClear,
    /// Generic clear acquisition used outside the VR-specific fallback path.
    ClearFallback,
}

/// Static Tracy labels for desktop surface submit sites.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SurfaceSubmitTrace {
    /// VR mirror submit for the desktop mirror blit.
    VrMirror,
    /// VR clear/fallback submit when no mirror image is available or mirror blit fails.
    VrClear,
    /// Generic clear submit used outside the VR-specific fallback path.
    ClearFallback,
}

/// Acquires the next surface texture with the same policy as [`present_clear_frame`].
///
/// Uses the window stored inside `gpu` for surface recovery, so callers do not need to thread
/// `&Window` through. Headless contexts have no surface and short-circuit to a hard error.
pub fn acquire_surface_outcome(
    gpu: &mut GpuContext,
) -> Result<SurfaceFrameOutcome, PresentClearError> {
    match gpu.acquire_with_recovery() {
        Ok(f) => Ok(SurfaceFrameOutcome::Acquired(f)),
        Err(wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded) => {
            logger::debug!("surface timeout or occluded; skipping frame");
            Ok(SurfaceFrameOutcome::Skip)
        }
        Err(wgpu::CurrentSurfaceTexture::Validation) => {
            logger::error!("surface validation error during acquire; reconfiguring");
            let (w, h) = gpu
                .window_inner_size()
                .unwrap_or_else(|| gpu.surface_extent_px());
            gpu.reconfigure(w, h);
            Ok(SurfaceFrameOutcome::Reconfigured)
        }
        Err(e) => Err(PresentClearError { status: e }),
    }
}

/// Acquires the next surface texture under a source-specific Tracy scope.
pub fn acquire_surface_outcome_traced(
    gpu: &mut GpuContext,
    trace: SurfaceAcquireTrace,
) -> Result<SurfaceFrameOutcome, PresentClearError> {
    match trace {
        SurfaceAcquireTrace::DesktopGraph => {
            profiling::scope!("gpu::surface_acquire.desktop_graph");
            acquire_surface_outcome(gpu)
        }
        SurfaceAcquireTrace::VrMirror => {
            profiling::scope!("gpu::surface_acquire.vr_mirror");
            acquire_surface_outcome(gpu)
        }
        SurfaceAcquireTrace::VrClear => {
            profiling::scope!("gpu::surface_acquire.vr_clear");
            acquire_surface_outcome(gpu)
        }
        SurfaceAcquireTrace::ClearFallback => {
            profiling::scope!("gpu::surface_acquire.clear_fallback");
            acquire_surface_outcome(gpu)
        }
    }
}

/// Submits command buffers with a presentable surface texture under a source-specific Tracy scope.
pub fn submit_surface_frame_traced(
    gpu: &mut GpuContext,
    command_buffers: Vec<wgpu::CommandBuffer>,
    frame: wgpu::SurfaceTexture,
    trace: SurfaceSubmitTrace,
) {
    match trace {
        SurfaceSubmitTrace::VrMirror => {
            profiling::scope!("gpu::surface_submit.vr_mirror");
            gpu.submit_frame_batch(command_buffers, Some(frame), None);
        }
        SurfaceSubmitTrace::VrClear => {
            profiling::scope!("gpu::surface_submit.vr_clear");
            gpu.submit_frame_batch(command_buffers, Some(frame), None);
        }
        SurfaceSubmitTrace::ClearFallback => {
            profiling::scope!("gpu::surface_submit.clear_fallback");
            gpu.submit_frame_batch(command_buffers, Some(frame), None);
        }
    }
}

/// Records a render pass that clears `view` to `load_color` (load op clear).
pub fn record_swapchain_clear_pass(
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    load_color: wgpu::Color,
    render_pass_label: Option<&str>,
) {
    let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: render_pass_label.or(Some("clear")),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(load_color),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        occlusion_query_set: None,
        timestamp_writes: None,
        multiview_mask: None,
    });
}

/// Clears the swapchain texture to [`SWAPCHAIN_CLEAR_COLOR`] and presents.
pub fn present_clear_frame(gpu: &mut GpuContext) -> Result<(), PresentClearError> {
    present_clear_frame_overlay(gpu, |_, _, _| Ok::<(), String>(()))
}

/// Clears the swapchain, optionally composites an overlay (e.g. Dear ImGui with `LoadOp::Load`), then presents.
pub fn present_clear_frame_overlay<F, E>(
    gpu: &mut GpuContext,
    overlay: F,
) -> Result<(), PresentClearError>
where
    F: FnOnce(&mut wgpu::CommandEncoder, &wgpu::TextureView, &mut GpuContext) -> Result<(), E>,
    E: std::fmt::Display,
{
    present_clear_frame_overlay_traced(
        gpu,
        SurfaceAcquireTrace::ClearFallback,
        SurfaceSubmitTrace::ClearFallback,
        overlay,
    )
}

/// Clears the swapchain, composites an overlay, and presents under source-specific Tracy scopes.
pub fn present_clear_frame_overlay_traced<F, E>(
    gpu: &mut GpuContext,
    acquire_trace: SurfaceAcquireTrace,
    submit_trace: SurfaceSubmitTrace,
    overlay: F,
) -> Result<(), PresentClearError>
where
    F: FnOnce(&mut wgpu::CommandEncoder, &wgpu::TextureView, &mut GpuContext) -> Result<(), E>,
    E: std::fmt::Display,
{
    let frame = match acquire_surface_outcome_traced(gpu, acquire_trace)? {
        SurfaceFrameOutcome::Skip | SurfaceFrameOutcome::Reconfigured => return Ok(()),
        SurfaceFrameOutcome::Acquired(f) => f,
    };

    let view = frame
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = gpu
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("skeleton-clear"),
        });
    record_swapchain_clear_pass(&mut encoder, &view, SWAPCHAIN_CLEAR_COLOR, Some("clear"));
    if let Err(e) = overlay(&mut encoder, &view, gpu) {
        logger::warn!("debug HUD overlay (clear frame): {e}");
    }
    // Hand submit + present to the driver thread so `Queue::submit` runs before
    // `SurfaceTexture::present`. Calling `present()` on the main thread immediately after
    // `submit_tracked_frame_commands` (which only enqueues on the driver) destroys the surface
    // texture, which makes the driver's deferred `Queue::submit` reject the command buffer:
    // "Texture with '<Surface Texture>' label has been destroyed".
    submit_surface_frame_traced(gpu, vec![encoder.finish()], frame, submit_trace);
    Ok(())
}
