//! Windowed [`GpuContext::new`] constructor.

use std::sync::Arc;

use winit::window::Window;

use super::super::super::adapter::device::{request_device_for_adapter, try_gpu_profiler};
use super::super::super::adapter::features::adapter_render_features_intersection;
use super::super::super::adapter::msaa_support::MsaaSupport;
use super::super::super::limits::GpuLimits;
use super::super::super::sync::device_health::GpuDeviceHealth;
use super::super::super::sync::mapped_buffer_health::GpuMappedBufferHealth;
use super::super::{GpuContext, GpuError};
use super::shared::{
    GpuContextParts, GpuRuntimeHandles, WindowAdapterLogFields, WindowAdapterSelection,
    WindowDisplayHandle, assemble_context, log_device_capability_summary,
    log_windowed_gpu_selection_summary, log_windowed_gpu_startup_request,
    select_window_adapters_with_fallback,
};
use crate::config::{GraphicsApiSetting, VsyncMode};
use crate::diagnostics::gpu_flight_recorder::GpuFlightRecorder;
use crate::gpu::submission_state::GpuSubmissionState;

/// Windowed GPU resources for an adapter that successfully configured the winit surface.
struct ConfiguredWindowGpu {
    /// Adapter metadata used for diagnostics and final context assembly.
    adapter_info: wgpu::AdapterInfo,
    /// Shared invalidation generation for CPU-mapped staging/readback buffers.
    mapped_buffer_health: Arc<GpuMappedBufferHealth>,
    /// Shared device-loss generation updated by wgpu error callbacks.
    device_health: Arc<GpuDeviceHealth>,
    /// Recent GPU lifecycle events retained for crash diagnostics.
    flight_recorder: Arc<GpuFlightRecorder>,
    /// MSAA tiers supported by the selected color/depth formats.
    msaa: MsaaSupport,
    /// Effective device limits validated against Renderide requirements.
    limits: Arc<GpuLimits>,
    /// Logical device created for the adapter.
    device: Arc<wgpu::Device>,
    /// Submission queue paired with the logical device.
    queue: wgpu::Queue,
    /// Surface configuration that has already passed `Surface::configure`.
    config: wgpu::SurfaceConfiguration,
    /// Present modes advertised by the surface for this adapter.
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
    /// `display_handle` is copied from the winit event-loop display that created the window. wgpu
    /// requires this for GLES presentation on some platforms, especially Wayland.
    ///
    /// `vsync` is resolved against the surface's actual present-mode capabilities via
    /// [`VsyncMode::resolve_present_mode`] (so e.g. [`VsyncMode::On`] picks strict `Fifo`
    /// presentation).
    ///
    /// `max_frame_latency` is the initial fixed value for
    /// [`wgpu::SurfaceConfiguration::desired_maximum_frame_latency`]. The renderer uses `2`,
    /// allowing CPU recording for frame N+1 to overlap with GPU work for frame N without adding
    /// another queued frame.
    ///
    /// `graphics_api` chooses the first backend set used for instance and adapter selection. An
    /// explicit API is retried with [`GraphicsApiSetting::Auto`] when it produces no adapter that
    /// can both enumerate for the surface and complete surface configuration. The final backend set
    /// may still be overridden by `WGPU_BACKEND`.
    pub async fn new(
        window: Arc<dyn Window>,
        display_handle: WindowDisplayHandle,
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
            true,
        );
        let selection = select_window_adapters_with_fallback(
            &window,
            display_handle,
            graphics_api,
            gpu_validation_layers,
            power_preference,
        )
        .await?;
        let selected_graphics_api = selection.graphics_api;
        match try_window_context_for_selection(
            &window,
            selection,
            vsync,
            max_frame_latency,
            power_preference,
        )
        .await
        {
            Ok(gpu) => Ok(gpu),
            Err(error)
                if should_retry_auto_after_window_context_failure(
                    selected_graphics_api,
                    &error,
                ) =>
            {
                logger::warn!(
                    "Configured graphics_api={} did not produce a surface-configurable windowed adapter: {error}. Retrying with graphics_api=auto.",
                    selected_graphics_api.as_persist_str()
                );
                let retry_selection = select_window_adapters_with_fallback(
                    &window,
                    display_handle,
                    GraphicsApiSetting::Auto,
                    gpu_validation_layers,
                    power_preference,
                )
                .await?;
                try_window_context_for_selection(
                    &window,
                    retry_selection,
                    vsync,
                    max_frame_latency,
                    power_preference,
                )
                .await
            }
            Err(error) => Err(error),
        }
    }
}

/// Attempts every ranked adapter in one selection result until the surface configures.
async fn try_window_context_for_selection(
    window: &Arc<dyn Window>,
    selection: WindowAdapterSelection,
    vsync: VsyncMode,
    max_frame_latency: u32,
    power_preference: wgpu::PowerPreference,
) -> Result<GpuContext, GpuError> {
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
        match configure_window_gpu_for_adapter(
            &adapter,
            &surface_safe,
            window,
            vsync,
            max_frame_latency,
        )
        .await
        {
            Ok(configured) => {
                logger::info!(
                    "wgpu adapter selected: {} type={:?} backend={:?} (preference={:?})",
                    configured.adapter_info.name,
                    configured.adapter_info.device_type,
                    configured.adapter_info.backend,
                    power_preference,
                );
                return assemble_window_context(
                    &adapter,
                    surface_safe,
                    Arc::clone(window),
                    selection_log,
                    configured,
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

/// Creates a device and proves that the adapter can configure the exact window surface.
async fn configure_window_gpu_for_adapter(
    adapter: &wgpu::Adapter,
    surface: &wgpu::Surface<'_>,
    window: &Arc<dyn Window>,
    vsync: VsyncMode,
    max_frame_latency: u32,
) -> Result<ConfiguredWindowGpu, GpuError> {
    let mapped_buffer_health = Arc::new(GpuMappedBufferHealth::new());
    let device_health = Arc::new(GpuDeviceHealth::new());
    let flight_recorder = Arc::new(GpuFlightRecorder::new());
    let required_features = adapter_render_features_intersection(adapter);
    let (device, queue) = request_device_for_adapter(
        adapter,
        required_features,
        Arc::clone(&mapped_buffer_health),
        Arc::clone(&device_health),
        Arc::clone(&flight_recorder),
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
    Ok(ConfiguredWindowGpu {
        adapter_info,
        mapped_buffer_health,
        device_health,
        flight_recorder,
        msaa,
        limits,
        device,
        queue,
        config,
        supported_present_modes,
    })
}

/// Builds the final [`GpuContext`] from a surface-configured adapter candidate.
fn assemble_window_context(
    adapter: &wgpu::Adapter,
    surface: wgpu::Surface<'static>,
    window: Arc<dyn Window>,
    selection_log: WindowAdapterLogFields,
    prepared: ConfiguredWindowGpu,
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
        Arc::clone(&prepared.device_health),
        Arc::clone(&prepared.flight_recorder),
    )?;
    let submission = GpuSubmissionState::new(
        runtime.driver_thread,
        runtime.frame_timing,
        runtime.frame_bracket,
        gpu_profiler,
        runtime.latest_gpu_profiler_snapshot,
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
        device_health: prepared.device_health,
        flight_recorder: prepared.flight_recorder,
        surface: Some(surface),
        config: prepared.config,
        supported_present_modes: prepared.supported_present_modes,
        window: Some(window),
    }))
}

/// Builds the startup error used when no ranked windowed adapter can configure the surface.
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

/// Returns whether a failed explicit windowed attempt should be retried with automatic backends.
fn should_retry_auto_after_window_context_failure(
    selected_graphics_api: GraphicsApiSetting,
    error: &GpuError,
) -> bool {
    selected_graphics_api.should_retry_auto_on_adapter_failure()
        && matches!(error, GpuError::SurfaceConfigure(_))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_api_surface_configure_failure_retries_auto() {
        let error = GpuError::SurfaceConfigure(String::from("invalid surface"));

        assert!(should_retry_auto_after_window_context_failure(
            GraphicsApiSetting::Vulkan,
            &error
        ));
    }

    #[test]
    fn auto_surface_configure_failure_does_not_retry_auto() {
        let error = GpuError::SurfaceConfigure(String::from("invalid surface"));

        assert!(!should_retry_auto_after_window_context_failure(
            GraphicsApiSetting::Auto,
            &error
        ));
    }

    #[test]
    fn explicit_api_non_surface_failure_does_not_retry_auto() {
        let error = GpuError::Adapter(String::from("adapter list unexpectedly empty"));

        assert!(!should_retry_auto_after_window_context_failure(
            GraphicsApiSetting::Vulkan,
            &error
        ));
    }
}
