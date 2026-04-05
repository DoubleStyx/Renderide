//! Swapchain presentation: a single clear pass (no mesh or UI draws).
//!
//! Serves as the minimal integration test for surface acquire, encoder submission, and present.

use wgpu::SurfaceError;
use winit::window::Window;

use crate::gpu::GpuContext;

/// Clears the swapchain texture to a dark blue and presents.
pub fn present_clear_frame(gpu: &mut GpuContext, window: &Window) -> Result<(), SurfaceError> {
    let frame = match gpu.acquire_with_recovery(window) {
        Ok(f) => f,
        Err(e @ SurfaceError::Timeout) => {
            logger::debug!("surface timeout: {e:?}");
            return Ok(());
        }
        Err(e @ SurfaceError::OutOfMemory) => {
            logger::error!("surface OOM: {e:?}");
            let s = window.inner_size();
            gpu.reconfigure(s.width, s.height);
            return Ok(());
        }
        Err(e) => return Err(e),
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
        });
    }
    gpu.queue()
        .lock()
        .expect("queue mutex poisoned")
        .submit(std::iter::once(encoder.finish()));
    frame.present();
    Ok(())
}
