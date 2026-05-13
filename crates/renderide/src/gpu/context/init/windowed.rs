//! Windowed [`GpuContext::new`] constructor.

use std::sync::Arc;

use winit::window::Window;

use super::super::super::adapter::device::{request_device_for_adapter, try_gpu_profiler};
use super::super::super::adapter::features::adapter_render_features_intersection;
use super::super::super::adapter::msaa_support::MsaaSupport;
use super::super::super::limits::GpuLimits;
use super::super::super::sync::mapped_buffer_health::GpuMappedBufferHealth;
use super::super::{GpuContext, GpuError};
use super::shared::{
    GpuContextParts, GpuRuntimeHandles, WindowAdapterLogFields, assemble_context,
    log_device_capability_summary, log_windowed_gpu_selection_summary,
    log_windowed_gpu_startup_request, select_window_adapters_with_fallback,
};
use crate::config::{GraphicsApiSetting, VsyncMode};
use crate::gpu::submission_state::GpuSubmissionState;

struct PreparedWindowGpu {
    adapter_info: wgpu::AdapterInfo,
    mapped_buffer_health: Arc<GpuMappedBufferHealth>,
    msaa: MsaaSupport,
    limits: Arc<GpuLimits>,
    device: Arc<wgpu::Device>,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    supported_present_modes: Vec<wgpu::PresentMode>,
}

impl GpuContext {
    /// Asynchronously builds GPU state for `window`.
    ///
    /// `gpu_validation_layers` selects whether to request backend validation before `WGPU_*` env
    /// overrides; see [`crate::gpu::instance_flags_for_gpu_init`]. `power_preference` is sourced
    /// from [`crate::config::DebugSettings::power_preference`] and used to rank enumerated
    /// adapters (discrete first when [`wgpu::PowerPreference::HighPerformance`], integrated first
    /// when [`wgpu::PowerPreference::LowPower`]).
    ///
    /// `vsync` is resolved against the surface's actual present-mode capabilities via
    /// [`VsyncMode::resolve_present_mode`] (so e.g. [`VsyncMode::On`] picks `FifoRelaxed` when
    /// available, then falls back to plain `Fifo`).
    ///
    /// `max_frame_latency` is the initial fixed value for
    /// [`wgpu::SurfaceConfiguration::desired_maximum_frame_latency`]. The renderer uses `2`,
    /// allowing CPU recording for frame N+1 to overlap with GPU work for frame N without adding
    /// another queued frame.
    ///
    /// `graphics_api` chooses the first backend set used for instance and adapter selection. An
    /// explicit API is retried with [`GraphicsApiSetting::Auto`] when it finds no compatible
    /// adapter. The final backend set may still be overridden by `WGPU_BACKEND`.
    pub async fn new(
        window: Arc<dyn Window>,
        vsync: VsyncMode,
        max_frame_latency: u32,
        gpu_validation_layers: bool,
        power_preference: wgpu::PowerPreference,
        graphics_api: GraphicsApiSetting,
    ) -> Result<Self, GpuError> {
        log_windowed_gpu_startup_request(
            window.as_ref(),
            vsync,
            max_frame_latency,
            gpu_validation_layers,
            power_preference,
            graphics_api,
        );
        let selection = select_window_adapters_with_fallback(
            &window,
            graphics_api,
            gpu_validation_layers,
            power_preference,
        )
        .await?;
        let selection_log = WindowAdapterLogFields {
            graphics_api: selection.graphics_api,
            active_backends: selection.active_backends,
            instance_flags: selection.instance_flags,
        };
        let surface_safe = selection.surface;
        let mut failures = Vec::new();
        for adapter in selection.adapters {
            let adapter_info = adapter.get_info();
            logger::info!(
                "wgpu adapter attempt: {} type={:?} backend={:?} (preference={:?})",
                adapter_info.name,
                adapter_info.device_type,
                adapter_info.backend,
                power_preference,
            );
            match prepare_window_gpu_for_adapter(
                &adapter,
                &surface_safe,
                &window,
                vsync,
                max_frame_latency,
            )
            .await
            {
                Ok(prepared) => {
                    logger::info!(
                        "wgpu adapter selected: {} type={:?} backend={:?} (preference={:?})",
                        prepared.adapter_info.name,
                        prepared.adapter_info.device_type,
                        prepared.adapter_info.backend,
                        power_preference,
                    );
                    return assemble_window_context(
                        &adapter,
                        surface_safe,
                        window,
                        selection_log,
                        prepared,
                        vsync,
                    );
                }
                Err(error) => {
                    let failure = format!(
                        "{} type={:?} backend={:?}: {error}",
                        adapter_info.name, adapter_info.device_type, adapter_info.backend,
                    );
                    logger::warn!("Windowed adapter rejected: {failure}");
                    failures.push(failure);
                }
            }
        }

        Err(windowed_surface_configure_error(failures))
    }
}

async fn prepare_window_gpu_for_adapter(
    adapter: &wgpu::Adapter,
    surface: &wgpu::Surface<'_>,
    window: &Arc<dyn Window>,
    vsync: VsyncMode,
    max_frame_latency: u32,
) -> Result<PreparedWindowGpu, GpuError> {
    let mapped_buffer_health = Arc::new(GpuMappedBufferHealth::new());
    let required_features = adapter_render_features_intersection(adapter);
    let (device, queue) = request_device_for_adapter(
        adapter,
        required_features,
        Arc::clone(&mapped_buffer_health),
    )
    .await?;

    let limits = GpuLimits::try_new(device.as_ref(), adapter)?;
    let size = window.surface_size();
    let supported_present_modes = surface.get_capabilities(adapter).present_modes;
    let mut config = surface
        .get_default_config(adapter, size.width.max(1), size.height.max(1))
        .ok_or(GpuError::SurfaceUnsupported)?;
    config.present_mode = vsync.resolve_present_mode(&supported_present_modes);
    config.desired_maximum_frame_latency = max_frame_latency;
    GpuContext::configure_surface_checked(surface, device.as_ref(), &config)
        .map_err(GpuError::SurfaceConfigure)?;

    let adapter_info = adapter.get_info();
    let depth_stencil_format = crate::gpu::main_forward_depth_stencil_format(required_features);
    let msaa = MsaaSupport::discover(
        adapter,
        config.format,
        depth_stencil_format,
        required_features,
        "GPU",
    );
    Ok(PreparedWindowGpu {
        adapter_info,
        mapped_buffer_health,
        msaa,
        limits,
        device,
        queue,
        config,
        supported_present_modes,
    })
}

fn assemble_window_context(
    adapter: &wgpu::Adapter,
    surface: wgpu::Surface<'static>,
    window: Arc<dyn Window>,
    selection_log: WindowAdapterLogFields,
    prepared: PreparedWindowGpu,
    vsync: VsyncMode,
) -> Result<GpuContext, GpuError> {
    log_windowed_gpu_selection_summary(
        &prepared.adapter_info,
        selection_log,
        &prepared.config,
        vsync,
        &prepared.supported_present_modes,
        &prepared.msaa,
    );
    log_device_capability_summary("GPU", prepared.device.as_ref());

    let gpu_profiler = try_gpu_profiler(
        adapter,
        prepared.device.as_ref(),
        &prepared.queue,
        "GPU profiler unavailable: adapter lacks TIMESTAMP_QUERY; \
         Tracy GPU timeline will be empty (CPU spans still work)",
    );
    let runtime = GpuRuntimeHandles::new(
        Arc::clone(&prepared.device),
        Arc::new(prepared.queue),
        Arc::clone(&prepared.mapped_buffer_health),
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
        adapter_info: prepared.adapter_info,
        msaa: prepared.msaa,
        limits: prepared.limits,
        device: prepared.device,
        queue: runtime.queue,
        gpu_queue_access_gate: runtime.gpu_queue_access_gate,
        mapped_buffer_health: prepared.mapped_buffer_health,
        surface: Some(surface),
        config: prepared.config,
        supported_present_modes: prepared.supported_present_modes,
        window: Some(window),
    }))
}

fn windowed_surface_configure_error(failures: Vec<String>) -> GpuError {
    let failure_summary = if failures.is_empty() {
        "no compatible adapters were attempted".to_owned()
    } else {
        failures.join("; ")
    };
    GpuError::SurfaceConfigure(format!(
        "no windowed adapter could configure the surface: {failure_summary}"
    ))
}
