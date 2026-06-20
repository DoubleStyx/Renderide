//! Surface lifecycle methods on [`GpuContext`]: present mode hot-reload, resize,
//! swapchain acquire-with-recovery, and small surface accessors.

use std::time::{Duration, Instant};

use crate::config::PresentationModeSetting;
use crate::gpu::flight_recorder::{
    GpuFlightEventKind, GpuFlightSurfaceReconfigureOutcome, GpuFlightSurfaceReconfigureSite,
};

use super::GpuContext;

impl GpuContext {
    /// Configures `surface` and reports wgpu validation/internal/OOM errors to the caller.
    pub(super) fn configure_surface_checked(
        surface: &wgpu::Surface<'_>,
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
    ) -> Result<(), String> {
        let out_of_memory_scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let internal_scope = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let validation_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        surface.configure(device, config);
        let validation_error = pollster::block_on(validation_scope.pop());
        let internal_error = pollster::block_on(internal_scope.pop());
        let out_of_memory_error = pollster::block_on(out_of_memory_scope.pop());
        validation_error
            .or(internal_error)
            .or(out_of_memory_error)
            .map_or(Ok(()), |error| Err(format_wgpu_error(error)))
    }

    fn configure_current_surface(&mut self) -> Result<(), String> {
        if self.device_lost() {
            self.surface_configured = false;
            return Err(String::from("device lost"));
        }
        if self.surface.is_none() {
            self.surface_configured = false;
            return Ok(());
        }
        self.wait_for_previous_present();
        let Some(surface) = self.surface.as_ref() else {
            self.surface_configured = false;
            return Ok(());
        };
        match Self::configure_surface_checked(surface, &self.device, &self.config) {
            Ok(()) => {
                self.surface_configured = true;
                Ok(())
            }
            Err(error) => {
                self.surface_configured = false;
                Err(error)
            }
        }
    }

    /// Updates the swapchain present mode and reconfigures the surface (hot-reload from settings).
    ///
    /// Resolves [`PresentationModeSetting`] against the surface's actual capabilities via
    /// [`PresentationModeSetting::resolve_present_mode`] so explicit unsupported modes fall back
    /// before `surface.configure`.
    /// Early-returns when the resolved mode matches the active configuration, so per-frame calls
    /// from the runtime are cheap.
    pub fn set_present_mode(&mut self, mode: PresentationModeSetting) {
        if self.device_lost() {
            self.record_surface_reconfigure(
                GpuFlightSurfaceReconfigureSite::PresentMode,
                GpuFlightSurfaceReconfigureOutcome::SkippedDeviceLost,
                self.surface_extent_px(),
                self.surface_extent_px(),
            );
            return;
        }
        let resolved = mode.resolve_present_mode(&self.supported_present_modes);
        if self.config.present_mode == resolved
            && (self.surface.is_none() || self.surface_configured)
        {
            return;
        }
        let previous = self.config.present_mode;
        let old_extent = self.surface_extent_px();
        self.config.present_mode = resolved;
        if let Err(error) = self.configure_current_surface() {
            self.record_surface_reconfigure(
                GpuFlightSurfaceReconfigureSite::PresentMode,
                GpuFlightSurfaceReconfigureOutcome::Failed(error.clone()),
                old_extent,
                self.surface_extent_px(),
            );
            logger::warn!(
                "Present mode reconfigure failed: {:?} -> {:?} (presentation_mode={:?} extent={}x{} format={:?}): {error}",
                previous,
                self.config.present_mode,
                mode,
                self.config.width,
                self.config.height,
                self.config.format,
            );
            return;
        }
        self.record_surface_reconfigure(
            GpuFlightSurfaceReconfigureSite::PresentMode,
            GpuFlightSurfaceReconfigureOutcome::Succeeded,
            old_extent,
            self.surface_extent_px(),
        );
        logger::info!(
            "Present mode set: {:?} -> {:?} (presentation_mode={:?} extent={}x{} format={:?})",
            previous,
            self.config.present_mode,
            mode,
            self.config.width,
            self.config.height,
            self.config.format,
        );
    }

    /// Swapchain pixel size `(width, height)`.
    pub fn surface_extent_px(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Reconfigures the swapchain after resize or after [`wgpu::CurrentSurfaceTexture::Lost`] /
    /// [`wgpu::CurrentSurfaceTexture::Outdated`].
    pub fn reconfigure(&mut self, width: u32, height: u32) {
        profiling::scope!("gpu::reconfigure_surface");
        let old = (self.config.width, self.config.height);
        if self.device_lost() {
            self.surface_configured = false;
            self.record_surface_reconfigure(
                GpuFlightSurfaceReconfigureSite::Resize,
                GpuFlightSurfaceReconfigureOutcome::SkippedDeviceLost,
                old,
                old,
            );
            logger::warn!("Surface reconfigure skipped because the GPU device is lost");
            return;
        }
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.depth_attachment = None;
        self.depth_extent_px = (0, 0);
        self.invalidate_primary_offscreen_targets();
        if let Err(error) = self.configure_current_surface() {
            self.record_surface_reconfigure(
                GpuFlightSurfaceReconfigureSite::Resize,
                GpuFlightSurfaceReconfigureOutcome::Failed(error.clone()),
                old,
                self.surface_extent_px(),
            );
            logger::warn!(
                "Surface reconfigure failed: old_extent={}x{} new_extent={}x{} format={:?} present_mode={:?}: {error}",
                old.0,
                old.1,
                self.config.width,
                self.config.height,
                self.config.format,
                self.config.present_mode,
            );
            return;
        }
        self.record_surface_reconfigure(
            GpuFlightSurfaceReconfigureSite::Resize,
            GpuFlightSurfaceReconfigureOutcome::Succeeded,
            old,
            self.surface_extent_px(),
        );
        logger::info!(
            "Surface reconfigured: old_extent={}x{} new_extent={}x{} format={:?} present_mode={:?}",
            old.0,
            old.1,
            self.config.width,
            self.config.height,
            self.config.format,
            self.config.present_mode,
        );
    }

    /// Records a surface reconfiguration event in the GPU flight recorder.
    fn record_surface_reconfigure(
        &self,
        site: GpuFlightSurfaceReconfigureSite,
        outcome: GpuFlightSurfaceReconfigureOutcome,
        old_extent: (u32, u32),
        new_extent: (u32, u32),
    ) {
        self.record_gpu_flight_event(GpuFlightEventKind::SurfaceReconfigure {
            site,
            outcome,
            old_extent,
            new_extent,
            format: self.config.format,
            present_mode: self.config.present_mode,
        });
    }

    /// Whether this context drives a real swapchain surface (vs. headless offscreen primary target).
    pub fn is_headless(&self) -> bool {
        self.surface.is_none()
    }

    /// Live `surface_size` of the window stored inside this context, if windowed.
    ///
    /// Re-queries the window each call so callers handling `WindowEvent::ScaleFactorChanged` can
    /// pick up the new logical size without holding a separate `Arc<dyn Window>`. Returns [`None`] in
    /// headless mode.
    pub fn window_surface_size(&self) -> Option<(u32, u32)> {
        self.window.as_ref().map(|w| {
            let s = w.surface_size();
            (s.width, s.height)
        })
    }

    /// Swapchain color format from the active surface configuration.
    pub fn config_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    /// Swapchain present mode.
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
        if self.device_lost() {
            return Err(wgpu::CurrentSurfaceTexture::Lost);
        }
        let Some(surface) = self.surface.as_ref() else {
            return Err(wgpu::CurrentSurfaceTexture::Lost);
        };
        if !self.surface_configured {
            logger::debug!("surface acquire skipped because the surface is not configured");
            return Err(wgpu::CurrentSurfaceTexture::Validation);
        }
        self.wait_for_previous_present();
        let (first_acquire, first_acquire_wait) = timed_surface_get_current_texture(surface);
        self.record_frame_timing_excluded_wait(first_acquire_wait);
        match first_acquire {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => Ok(t),
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                logger::info!(
                    "surface Lost or Outdated -- reconfiguring (current_extent={}x{} present_mode={:?})",
                    self.config.width,
                    self.config.height,
                    self.config.present_mode,
                );
                let size = self.window.as_ref().map(|w| w.surface_size());
                if let Some(s) = size {
                    self.reconfigure(s.width, s.height);
                }
                if !self.surface_configured {
                    return Err(wgpu::CurrentSurfaceTexture::Validation);
                }
                let Some(surface) = self.surface.as_ref() else {
                    return Err(wgpu::CurrentSurfaceTexture::Lost);
                };
                let (second_acquire, second_acquire_wait) =
                    timed_surface_get_current_texture(surface);
                self.record_frame_timing_excluded_wait(second_acquire_wait);
                match second_acquire {
                    wgpu::CurrentSurfaceTexture::Success(t)
                    | wgpu::CurrentSurfaceTexture::Suboptimal(t) => Ok(t),
                    other => Err(other),
                }
            }
            other => Err(other),
        }
    }
}

/// Calls `Surface::get_current_texture` and returns the elapsed wall-clock time.
fn timed_surface_get_current_texture(
    surface: &wgpu::Surface<'_>,
) -> (wgpu::CurrentSurfaceTexture, Duration) {
    profiling::scope!("gpu::surface_get_current_texture");
    let start = Instant::now();
    let texture = surface.get_current_texture();
    let wait = start.elapsed();
    crate::profiling::plot_surface_get_current_texture_ms(wait);
    (texture, wait)
}

fn format_wgpu_error(error: wgpu::Error) -> String {
    match error {
        wgpu::Error::OutOfMemory { source } => format!("out of memory ({source})"),
        wgpu::Error::Validation {
            description,
            source,
        } => {
            format!("{description} ({source})")
        }
        wgpu::Error::Internal {
            description,
            source,
        } => {
            format!("{description} ({source})")
        }
    }
}
