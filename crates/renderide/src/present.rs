//! Swapchain presentation: surface acquire helpers and a minimal clear pass (no mesh or UI draws).
//!
//! Serves as the minimal integration test for surface acquire, encoder submission, and present.
//! The render graph reuses [`acquire_surface_outcome`] and [`record_swapchain_clear_pass`].

use winit::window::Window;

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

/// Acquires the next surface texture with the same policy as [`present_clear_frame`].
pub fn acquire_surface_outcome(
    gpu: &mut GpuContext,
    window: &Window,
) -> Result<SurfaceFrameOutcome, PresentClearError> {
    match gpu.acquire_with_recovery(window) {
        Ok(f) => Ok(SurfaceFrameOutcome::Acquired(f)),
        Err(wgpu::CurrentSurfaceTexture::Timeout) | Err(wgpu::CurrentSurfaceTexture::Occluded) => {
            logger::debug!("surface timeout or occluded; skipping frame");
            Ok(SurfaceFrameOutcome::Skip)
        }
        Err(wgpu::CurrentSurfaceTexture::Validation) => {
            logger::error!("surface validation error during acquire; reconfiguring");
            let s = window.inner_size();
            gpu.reconfigure(s.width, s.height);
            Ok(SurfaceFrameOutcome::Reconfigured)
        }
        Err(e) => Err(PresentClearError { status: e }),
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
pub fn present_clear_frame(gpu: &mut GpuContext, window: &Window) -> Result<(), PresentClearError> {
    let frame = match acquire_surface_outcome(gpu, window)? {
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
    gpu.queue()
        .lock()
        .expect("queue mutex poisoned")
        .submit(std::iter::once(encoder.finish()));
    frame.present();
    Ok(())
}
