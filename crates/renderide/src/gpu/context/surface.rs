//! Surface lifecycle methods on [`GpuContext`]: present mode / latency hot-reload,
//! resize, swapchain acquire-with-recovery, and small surface accessors.

use winit::dpi::PhysicalSize;

use crate::config::VsyncMode;

use super::GpuContext;

impl GpuContext {
    /// Updates the swapchain present mode and reconfigures the surface (hot-reload from settings).
    ///
    /// Resolves [`VsyncMode`] against the surface's actual capabilities via
    /// [`VsyncMode::resolve_present_mode`] so the result is guaranteed to be one of the variants
    /// the swapchain advertises (no risk of `surface.configure` rejecting an unsupported mode).
    /// Early-returns when the resolved mode matches the active configuration, so per-frame calls
    /// from the runtime are cheap.
    pub fn set_present_mode(&mut self, mode: VsyncMode) {
        let resolved = mode.resolve_present_mode(&self.supported_present_modes);
        if self.config.present_mode == resolved {
            return;
        }
        self.config.present_mode = resolved;
        if let Some(surface) = self.surface.as_ref() {
            self.wait_for_previous_present();
            surface.configure(&self.device, &self.config);
        }
        logger::info!(
            "Present mode set to {:?} (vsync={:?})",
            self.config.present_mode,
            mode
        );
    }

    /// Updates [`wgpu::SurfaceConfiguration::desired_maximum_frame_latency`] and reconfigures the
    /// surface (hot-reload from settings).
    ///
    /// Early-returns when the requested value already matches the active configuration so the
    /// per-tick call from the runtime is cheap. Headless contexts still update the cached config
    /// for parity, but skip the surface reconfigure since there is no surface.
    ///
    /// Callers must pass a value in `1..=3`; use
    /// [`crate::config::RenderingSettings::resolved_max_frame_latency`] which clamps for you.
    pub fn set_max_frame_latency(&mut self, max_frame_latency: u32) {
        if self.config.desired_maximum_frame_latency == max_frame_latency {
            return;
        }
        let previous = self.config.desired_maximum_frame_latency;
        self.config.desired_maximum_frame_latency = max_frame_latency;
        if let Some(surface) = self.surface.as_ref() {
            self.wait_for_previous_present();
            surface.configure(&self.device, &self.config);
        }
        logger::info!(
            "desired_maximum_frame_latency set to {} (was {})",
            max_frame_latency,
            previous
        );
    }

    /// Current swapchain configuration extent.
    pub fn size(&self) -> PhysicalSize<u32> {
        PhysicalSize::new(self.config.width, self.config.height)
    }

    /// Swapchain pixel size `(width, height)`.
    pub fn surface_extent_px(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Reconfigures the swapchain after resize or after [`wgpu::CurrentSurfaceTexture::Lost`] /
    /// [`wgpu::CurrentSurfaceTexture::Outdated`].
    pub fn reconfigure(&mut self, width: u32, height: u32) {
        profiling::scope!("gpu::reconfigure_surface");
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        if let Some(surface) = self.surface.as_ref() {
            self.wait_for_previous_present();
            surface.configure(&self.device, &self.config);
        }
        self.depth_attachment = None;
        self.depth_extent_px = (0, 0);
    }

    /// Borrows the configured surface for acquire/submit; [`None`] in headless mode.
    pub fn surface(&self) -> Option<&wgpu::Surface<'static>> {
        self.surface.as_ref()
    }

    /// Whether this context drives a real swapchain surface (vs. headless offscreen primary target).
    pub fn is_headless(&self) -> bool {
        self.surface.is_none()
    }

    /// Live `inner_size` of the window stored inside this context, if windowed.
    ///
    /// Re-queries the window each call so callers handling `WindowEvent::ScaleFactorChanged` can
    /// pick up the new logical size without holding a separate `Arc<Window>`. Returns [`None`] in
    /// headless mode.
    pub fn window_inner_size(&self) -> Option<(u32, u32)> {
        self.window.as_ref().map(|w| {
            let s = w.inner_size();
            (s.width, s.height)
        })
    }

    /// Swapchain color format from the active surface configuration.
    pub fn config_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    /// Swapchain present mode (vsync policy).
    pub fn present_mode(&self) -> wgpu::PresentMode {
        self.config.present_mode
    }

    /// Acquires the next frame, reconfiguring once on [`wgpu::CurrentSurfaceTexture::Lost`] or
    /// [`wgpu::CurrentSurfaceTexture::Outdated`].
    ///
    /// Returns [`wgpu::CurrentSurfaceTexture::Lost`] when this context is headless (no surface).
    /// Uses the stored [`Self::window`] for size queries on recovery so render-path callers do
    /// not have to thread `&Window` through their signatures.
    pub fn acquire_with_recovery(
        &mut self,
    ) -> Result<wgpu::SurfaceTexture, wgpu::CurrentSurfaceTexture> {
        let Some(surface) = self.surface.as_ref() else {
            return Err(wgpu::CurrentSurfaceTexture::Lost);
        };
        self.wait_for_previous_present();
        match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => Ok(t),
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                logger::info!("surface Lost or Outdated -- reconfiguring");
                let size = self.window.as_ref().map(|w| w.inner_size());
                if let Some(s) = size {
                    self.reconfigure(s.width, s.height);
                }
                let Some(surface) = self.surface.as_ref() else {
                    return Err(wgpu::CurrentSurfaceTexture::Lost);
                };
                match surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(t)
                    | wgpu::CurrentSurfaceTexture::Suboptimal(t) => Ok(t),
                    other => Err(other),
                }
            }
            other => Err(other),
        }
    }
}
