//! Surface texture acquisition with one-shot reconfigure on loss or outdated swapchain.

use winit::window::Window;

use super::GpuContext;

/// Acquires the next frame, reconfiguring once on [`wgpu::CurrentSurfaceTexture::Lost`] or
/// [`wgpu::CurrentSurfaceTexture::Outdated`].
pub(super) fn acquire_with_recovery(
    ctx: &mut GpuContext,
    window: &Window,
) -> Result<wgpu::SurfaceTexture, wgpu::CurrentSurfaceTexture> {
    match ctx.surface().get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => {
            Ok(t)
        }
        wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
            logger::info!("surface Lost or Outdated — reconfiguring");
            let s = window.inner_size();
            ctx.reconfigure(s.width, s.height);
            match ctx.surface().get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(t)
                | wgpu::CurrentSurfaceTexture::Suboptimal(t) => Ok(t),
                other => Err(other),
            }
        }
        other => Err(other),
    }
}
