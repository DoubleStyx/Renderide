//! Swapchain presentation: a single clear pass (no mesh or UI draws).
//!
//! Serves as the minimal integration test for surface acquire, encoder submission, and present.

use std::error::Error;
use std::fmt;

use winit::window::Window;

use crate::gpu::GpuContext;

/// Failure to obtain a presentable surface texture after recovery attempts.
#[derive(Debug)]
pub struct PresentClearError {
    /// Status from [`wgpu::Surface::get_current_texture`] after reconfiguration.
    pub status: wgpu::CurrentSurfaceTexture,
}

impl fmt::Display for PresentClearError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "could not acquire surface texture ({:?})", self.status)
    }
}

impl Error for PresentClearError {}

/// Clears the swapchain texture to a dark blue and presents.
pub fn present_clear_frame(gpu: &mut GpuContext, window: &Window) -> Result<(), PresentClearError> {
    let frame = match gpu.acquire_with_recovery(window) {
        Ok(f) => f,
        Err(wgpu::CurrentSurfaceTexture::Timeout) | Err(wgpu::CurrentSurfaceTexture::Occluded) => {
            logger::debug!("surface timeout or occluded; skipping frame");
            return Ok(());
        }
        Err(wgpu::CurrentSurfaceTexture::Validation) => {
            logger::error!("surface validation error during acquire; reconfiguring");
            let s = window.inner_size();
            gpu.reconfigure(s.width, s.height);
            return Ok(());
        }
        Err(e) => return Err(PresentClearError { status: e }),
    };

    let view = frame
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = gpu
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("skeleton-clear"),
        });
    {
        let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.02,
                        g: 0.05,
                        b: 0.12,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });
    }
    gpu.queue()
        .lock()
        .expect("queue mutex poisoned")
        .submit(std::iter::once(encoder.finish()));
    frame.present();
    Ok(())
}
