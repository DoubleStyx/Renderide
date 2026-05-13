//! OpenXR-bootstrap [`GpuContext::new_from_openxr_bootstrap`] constructor.

use std::sync::Arc;

use winit::window::Window;

use super::super::super::adapter::device::{install_uncaptured_error_handler, try_gpu_profiler};
use super::super::super::adapter::msaa_support::MsaaSupport;
use super::super::super::limits::GpuLimits;
use super::super::super::sync::mapped_buffer_health::GpuMappedBufferHealth;
use super::super::{GpuContext, GpuError};
use super::shared::{
    GpuContextParts, GpuRuntimeHandles, assemble_context, log_device_capability_summary,
};
use crate::config::VsyncMode;
use crate::gpu::submission_state::GpuSubmissionState;

impl GpuContext {
    /// Builds GPU state using an existing wgpu instance/device from OpenXR bootstrap (mirror window).
    ///
    /// The mirror surface uses the same capability-aware [`VsyncMode`] mapping and fixed
    /// `max_frame_latency` semantics as the desktop constructor so windowed presentation behaves
    /// consistently across desktop and VR startup paths.
    pub fn new_from_openxr_bootstrap(
        instance: &wgpu::Instance,
        adapter: &wgpu::Adapter,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        window: Arc<dyn Window>,
        vsync: VsyncMode,
        max_frame_latency: u32,
    ) -> Result<Self, GpuError> {
        let requested_size = window.surface_size();
        logger::info!(
            "GPU startup request (OpenXR mirror): vsync={:?} max_frame_latency={} initial_extent={}x{}",
            vsync,
            max_frame_latency,
            requested_size.width,
            requested_size.height,
        );
        let mapped_buffer_health = Arc::new(GpuMappedBufferHealth::new());
        install_uncaptured_error_handler(device.as_ref(), Arc::clone(&mapped_buffer_health));
        // `Arc<dyn Window>` is `Into<SurfaceTarget<'static>>`, so the returned `Surface` is
        // already `'static` -- no `transmute` is required to extend the borrow.
        let surface_safe: wgpu::Surface<'static> = instance
            .create_surface(window.clone())
            .map_err(|e| GpuError::Surface(format!("{e:?}")))?;
        let size = window.surface_size();
        let supported_present_modes = surface_safe.get_capabilities(adapter).present_modes;
        let mut config = surface_safe
            .get_default_config(adapter, size.width.max(1), size.height.max(1))
            .ok_or(GpuError::SurfaceUnsupported)?;
        config.present_mode = vsync.resolve_present_mode(&supported_present_modes);
        config.desired_maximum_frame_latency = max_frame_latency;
        GpuContext::configure_surface_checked(&surface_safe, device.as_ref(), &config)
            .map_err(GpuError::SurfaceConfigure)?;
        let adapter_info = adapter.get_info();
        let limits = GpuLimits::try_new(device.as_ref(), adapter)?;
        let depth_stencil_format = crate::gpu::main_forward_depth_stencil_format(device.features());
        let msaa = MsaaSupport::discover(
            adapter,
            config.format,
            depth_stencil_format,
            device.features(),
            "GPU (OpenXR path)",
        );
        logger::info!(
            "GPU (OpenXR path): adapter={} backend={:?} extent={}x{} format={:?} vsync={:?} present_mode={:?} \
             supported_present_modes={:?} desired_maximum_frame_latency={} \
             msaa_supported_sample_counts={:?} msaa_max_sample_count={} \
             msaa_supported_sample_counts_stereo={:?} msaa_max_sample_count_stereo={}",
            adapter_info.name,
            adapter_info.backend,
            config.width,
            config.height,
            config.format,
            vsync,
            config.present_mode,
            supported_present_modes,
            config.desired_maximum_frame_latency,
            &msaa.desktop,
            msaa.desktop_max(),
            &msaa.stereo,
            msaa.stereo_max()
        );
        log_device_capability_summary("GPU (OpenXR path)", device.as_ref());
        let gpu_profiler = try_gpu_profiler(
            adapter,
            device.as_ref(),
            queue.as_ref(),
            "GPU profiler unavailable (OpenXR path): adapter lacks \
             TIMESTAMP_QUERY; Tracy GPU timeline will be empty",
        );
        let runtime = GpuRuntimeHandles::new(
            Arc::clone(&device),
            queue,
            Arc::clone(&mapped_buffer_health),
        )?;
        let submission = GpuSubmissionState::new(
            runtime.driver_thread,
            runtime.frame_timing,
            runtime.frame_bracket,
            gpu_profiler,
            runtime.latest_gpu_pass_timings,
        );
        Ok(assemble_context(GpuContextParts {
            submission,
            adapter_info,
            msaa,
            limits,
            device,
            queue: runtime.queue,
            gpu_queue_access_gate: runtime.gpu_queue_access_gate,
            mapped_buffer_health,
            surface: Some(surface_safe),
            config,
            supported_present_modes,
            window: Some(window),
        }))
    }
}
