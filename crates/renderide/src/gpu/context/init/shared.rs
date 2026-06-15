//! Shared bootstrap pieces used by all three [`super::super::GpuContext`] constructors.

use std::sync::{Arc, Mutex};

use winit::event_loop::OwnedDisplayHandle;
use winit::raw_window_handle::{DisplayHandle, HandleError, HasDisplayHandle, RawDisplayHandle};
use winit::window::Window;

use super::super::super::adapter::msaa_support::MsaaSupport;
use super::super::super::adapter::selection::{
    WgpuDisplayHandle, build_wgpu_instance, select_adapter, select_adapters,
};
use super::super::super::limits::GpuLimits;
use super::super::super::profiling::frame_bracket::FrameBracket;
use super::super::super::profiling::frame_cpu_gpu_timing::{
    FrameCpuGpuTiming, FrameCpuGpuTimingHandle,
};
use super::super::super::sync::device_health::GpuDeviceHealth;
use super::super::super::sync::mapped_buffer_health::GpuMappedBufferHealth;
use super::super::{GpuContext, GpuError, PrimaryOffscreenTargets};
use crate::config::{GraphicsApiSetting, VsyncMode};
use crate::gpu::flight_recorder::GpuFlightRecorder;
use crate::gpu::submission_state::GpuSubmissionState;

/// Runtime handles derived from a queue and shared by all GPU construction paths.
pub(super) struct GpuRuntimeHandles {
    /// Shared queue handle stored on [`GpuContext`].
    pub(super) queue: Arc<wgpu::Queue>,
    /// Driver-thread submit gate paired with [`Self::queue`].
    pub(super) gpu_queue_access_gate: super::super::super::GpuQueueAccessGate,
    /// Dedicated submit/present worker.
    pub(super) driver_thread: super::super::super::driver_thread::DriverThread,
    /// CPU/GPU frame timing accumulator.
    pub(super) frame_timing: FrameCpuGpuTimingHandle,
    /// Real-GPU-timestamp factory for the debug HUD's `gpu_frame_ms`. Always constructed; emits
    /// sessions only when the adapter advertises the required timestamp features.
    pub(super) frame_bracket: FrameBracket,
    /// Latest flattened GPU pass timings and profiler accounting for the HUD.
    pub(super) latest_gpu_profiler_snapshot: Arc<Mutex<crate::profiling::GpuProfilerSnapshot>>,
}

impl GpuRuntimeHandles {
    /// Builds the driver-thread and timing handles for a `(device, queue)` pair.
    pub(super) fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        mapped_buffer_health: Arc<GpuMappedBufferHealth>,
        device_health: Arc<GpuDeviceHealth>,
        flight_recorder: Arc<GpuFlightRecorder>,
    ) -> Result<Self, GpuError> {
        let gpu_queue_access_gate = super::super::super::GpuQueueAccessGate::new();
        let driver_thread = super::super::super::driver_thread::DriverThread::new(
            Arc::clone(&queue),
            gpu_queue_access_gate.clone(),
            device_health,
            Arc::clone(&flight_recorder),
        )
        .map_err(GpuError::DriverThreadSpawn)?;
        let frame_bracket = FrameBracket::new(device, Arc::clone(&queue), mapped_buffer_health);
        Ok(Self {
            queue,
            gpu_queue_access_gate,
            driver_thread,
            frame_timing: Arc::new(Mutex::new(FrameCpuGpuTiming::default())),
            frame_bracket,
            latest_gpu_profiler_snapshot: Arc::new(Mutex::new(
                crate::profiling::GpuProfilerSnapshot::default(),
            )),
        })
    }
}

/// Inputs that differ between the three [`GpuContext`] construction paths.
pub(super) struct GpuContextParts {
    /// Submission, timing, and profiling state.
    pub(super) submission: GpuSubmissionState,
    /// Adapter metadata captured at construction.
    pub(super) adapter_info: wgpu::AdapterInfo,
    /// MSAA support lists for desktop and stereo paths.
    pub(super) msaa: MsaaSupport,
    /// Effective limits and derived caps.
    pub(super) limits: Arc<GpuLimits>,
    /// Logical device.
    pub(super) device: Arc<wgpu::Device>,
    /// Submission queue.
    pub(super) queue: Arc<wgpu::Queue>,
    /// Shared write-texture/submit gate.
    pub(super) gpu_queue_access_gate: super::super::super::GpuQueueAccessGate,
    /// Shared mapped-buffer invalidation generation.
    pub(super) mapped_buffer_health: Arc<GpuMappedBufferHealth>,
    /// Shared device-loss generation.
    pub(super) device_health: Arc<GpuDeviceHealth>,
    /// Recent GPU/XR lifecycle events retained in memory.
    pub(super) flight_recorder: Arc<GpuFlightRecorder>,
    /// Optional window-backed surface.
    pub(super) surface: Option<wgpu::Surface<'static>>,
    /// Active surface/offscreen configuration.
    pub(super) config: wgpu::SurfaceConfiguration,
    /// Surface present modes.
    pub(super) supported_present_modes: Vec<wgpu::PresentMode>,
    /// Optional window owner.
    pub(super) window: Option<Arc<dyn Window>>,
}

/// Windowed adapter-selection result before device creation.
pub(super) struct WindowAdapterSelection {
    /// Graphics API attempt that produced the adapter.
    pub(super) graphics_api: GraphicsApiSetting,
    /// Instance flags after wgpu environment overrides.
    pub(super) instance_flags: wgpu::InstanceFlags,
    /// Active backend set after wgpu environment overrides.
    pub(super) active_backends: wgpu::Backends,
    /// Surface created from the same wgpu instance used for adapter selection.
    pub(super) surface: wgpu::Surface<'static>,
    /// Compatible adapters in the order the windowed path should attempt them.
    pub(super) adapters: Vec<wgpu::Adapter>,
}

#[derive(Clone, Copy)]
pub(super) struct WindowAdapterLogFields {
    pub(super) graphics_api: GraphicsApiSetting,
    pub(super) active_backends: wgpu::Backends,
    pub(super) instance_flags: wgpu::InstanceFlags,
}

/// Headless adapter-selection result before device creation.
pub(super) struct HeadlessAdapterSelection {
    /// Graphics API attempt that produced the adapter.
    pub(super) graphics_api: GraphicsApiSetting,
    /// Instance flags after wgpu environment overrides.
    pub(super) instance_flags: wgpu::InstanceFlags,
    /// Active backend set after wgpu environment overrides.
    pub(super) active_backends: wgpu::Backends,
    /// Selected adapter.
    pub(super) adapter: wgpu::Adapter,
}

pub(super) fn log_windowed_gpu_startup_request(
    window: &dyn Window,
    vsync: VsyncMode,
    max_frame_latency: u32,
    gpu_validation_layers: bool,
    power_preference: wgpu::PowerPreference,
    graphics_api: GraphicsApiSetting,
    display_handle_provided: bool,
) {
    let requested_size = window.surface_size();
    logger::info!(
        "GPU startup request (windowed): graphics_api={} validation={} power_preference={:?} vsync={:?} max_frame_latency={} initial_extent={}x{} display_handle_provided={}",
        graphics_api.as_persist_str(),
        gpu_validation_layers,
        power_preference,
        vsync,
        max_frame_latency,
        requested_size.width,
        requested_size.height,
        display_handle_provided,
    );
}

pub(super) fn log_windowed_gpu_selection_summary(
    adapter_info: &wgpu::AdapterInfo,
    selection: WindowAdapterLogFields,
    config: &wgpu::SurfaceConfiguration,
    vsync: VsyncMode,
    supported_present_modes: &[wgpu::PresentMode],
    msaa: &MsaaSupport,
) {
    logger::info!(
        "GPU: adapter={} type={:?} backend={:?} vendor={:#010x} device={:#010x} pci_bus_id={} driver={} driver_info={} graphics_api={} active_backends={:?} extent={}x{} format={:?} vsync={:?} present_mode={:?} \
         supported_present_modes={:?} desired_maximum_frame_latency={} instance_flags={:?} \
         msaa_supported_sample_counts={:?} msaa_max_sample_count={} \
         msaa_supported_sample_counts_stereo={:?} msaa_max_sample_count_stereo={}",
        adapter_info.name,
        adapter_info.device_type,
        adapter_info.backend,
        adapter_info.vendor,
        adapter_info.device,
        adapter_info_field_or_unreported(&adapter_info.device_pci_bus_id),
        adapter_info_field_or_unreported(&adapter_info.driver),
        adapter_info_field_or_unreported(&adapter_info.driver_info),
        selection.graphics_api.as_persist_str(),
        selection.active_backends,
        config.width,
        config.height,
        config.format,
        vsync,
        config.present_mode,
        supported_present_modes,
        config.desired_maximum_frame_latency,
        selection.instance_flags,
        &msaa.desktop,
        msaa.desktop_max(),
        &msaa.stereo,
        msaa.stereo_max()
    );
}

/// Returns a stable placeholder when wgpu reports an empty adapter identity field.
pub(super) fn adapter_info_field_or_unreported(value: &str) -> &str {
    if value.trim().is_empty() {
        "<unreported>"
    } else {
        value
    }
}

/// Builds the common [`GpuContext`] field set once all path-specific resources are ready.
pub(super) fn assemble_context(parts: GpuContextParts) -> GpuContext {
    GpuContext {
        submission: parts.submission,
        adapter_info: parts.adapter_info,
        msaa: super::super::GpuMsaa::new(parts.msaa),
        limits: parts.limits,
        device: parts.device,
        queue: parts.queue,
        gpu_queue_access_gate: parts.gpu_queue_access_gate,
        mapped_buffer_recovery: super::super::mapped_buffer_recovery::GpuMappedBufferRecovery::new(
            parts.mapped_buffer_health,
        ),
        device_health: parts.device_health,
        flight_recorder: parts.flight_recorder,
        seen_device_lost_generation: 0,
        surface_configured: parts.surface.is_some(),
        surface: parts.surface,
        config: parts.config,
        supported_present_modes: parts.supported_present_modes,
        window: parts.window,
        depth_attachment: None,
        depth_extent_px: (0, 0),
        primary_offscreen: Option::<PrimaryOffscreenTargets>::None,
    }
}

pub(super) async fn select_window_adapters_with_fallback(
    window: &Arc<dyn Window>,
    display_handle: WindowDisplayHandle,
    graphics_api: GraphicsApiSetting,
    gpu_validation_layers: bool,
    power_preference: wgpu::PowerPreference,
) -> Result<WindowAdapterSelection, GpuError> {
    match select_window_adapters(
        window,
        display_handle,
        graphics_api,
        gpu_validation_layers,
        power_preference,
    )
    .await
    {
        Ok(selection) => Ok(selection),
        Err(error) if graphics_api.should_retry_auto_on_adapter_failure() => {
            logger::warn!(
                "Configured graphics_api={} did not produce a compatible windowed adapter: {error}. Retrying with graphics_api=auto.",
                graphics_api.as_persist_str()
            );
            select_window_adapters(
                window,
                display_handle,
                GraphicsApiSetting::Auto,
                gpu_validation_layers,
                power_preference,
            )
            .await
        }
        Err(error) => Err(error),
    }
}

async fn select_window_adapters(
    window: &Arc<dyn Window>,
    display_handle: WindowDisplayHandle,
    graphics_api: GraphicsApiSetting,
    gpu_validation_layers: bool,
    power_preference: wgpu::PowerPreference,
) -> Result<WindowAdapterSelection, GpuError> {
    let (instance, instance_flags, active_backends) = build_wgpu_instance(
        gpu_validation_layers,
        graphics_api.requested_backends(),
        Some(display_handle.into_wgpu_display_handle()),
    );

    // `Arc<dyn Window>` is `Into<SurfaceTarget<'static>>`, so the returned `Surface` is
    // already `'static` -- no `transmute` is required to extend the borrow.
    let surface: wgpu::Surface<'static> = instance
        .create_surface(Arc::clone(window))
        .map_err(|e| GpuError::Surface(format!("{e:?}")))?;

    let adapters =
        select_adapters(&instance, Some(&surface), power_preference, active_backends).await?;

    Ok(WindowAdapterSelection {
        graphics_api,
        instance_flags,
        active_backends,
        surface,
        adapters,
    })
}

pub(super) async fn select_headless_adapter_with_fallback(
    graphics_api: GraphicsApiSetting,
    gpu_validation_layers: bool,
    power_preference: wgpu::PowerPreference,
) -> Result<HeadlessAdapterSelection, GpuError> {
    match select_headless_adapter(graphics_api, gpu_validation_layers, power_preference).await {
        Ok(selection) => Ok(selection),
        Err(error) if graphics_api.should_retry_auto_on_adapter_failure() => {
            logger::warn!(
                "Configured graphics_api={} did not produce a compatible headless adapter: {error}. Retrying with graphics_api=auto.",
                graphics_api.as_persist_str()
            );
            select_headless_adapter(
                GraphicsApiSetting::Auto,
                gpu_validation_layers,
                power_preference,
            )
            .await
        }
        Err(error) => Err(error),
    }
}

async fn select_headless_adapter(
    graphics_api: GraphicsApiSetting,
    gpu_validation_layers: bool,
    power_preference: wgpu::PowerPreference,
) -> Result<HeadlessAdapterSelection, GpuError> {
    let (instance, instance_flags, active_backends) = build_wgpu_instance(
        gpu_validation_layers,
        graphics_api.requested_backends(),
        None,
    );
    let adapter = select_adapter(&instance, None, power_preference, active_backends).await?;
    Ok(HeadlessAdapterSelection {
        graphics_api,
        instance_flags,
        active_backends,
        adapter,
    })
}

/// Raw platform display handle copied from the winit event loop for wgpu instance creation.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WindowDisplayHandle {
    raw: RawDisplayHandle,
}

impl WindowDisplayHandle {
    /// Copies the raw display handle from winit's owned event-loop display handle.
    pub(crate) fn from_winit(display_handle: &OwnedDisplayHandle) -> Result<Self, GpuError> {
        display_handle
            .display_handle()
            .map(|handle| Self {
                raw: handle.as_raw(),
            })
            .map_err(|error| {
                GpuError::Surface(format!("event-loop display handle failed: {error:?}"))
            })
    }

    fn into_wgpu_display_handle(self) -> WgpuDisplayHandle {
        Box::new(self)
    }
}

// SAFETY: The copied raw display handle is immutable and contains only platform identifiers.
// `AppDriver` keeps winit's `OwnedDisplayHandle` alive until after the windowed GPU context and
// surface have been dropped, preserving the display connection behind those identifiers.
unsafe impl Send for WindowDisplayHandle {}

// SAFETY: The wrapper never mutates the display connection and only re-borrows the copied raw
// display handle. The display owner remains alive for the full windowed GPU lifetime.
unsafe impl Sync for WindowDisplayHandle {}

impl HasDisplayHandle for WindowDisplayHandle {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        // SAFETY: `self.raw` was copied from winit's live event-loop display handle, and the app
        // driver keeps that owned display handle alive while wgpu can use this wrapper.
        Ok(unsafe { DisplayHandle::borrow_raw(self.raw) })
    }
}

pub(super) fn log_device_capability_summary(label: &str, device: &wgpu::Device) {
    let limits = device.limits();
    logger::info!(
        "{label} device capabilities: features={:?} max_texture_2d={} max_texture_array_layers={} max_buffer_size={} max_bind_groups={} max_storage_buffers_per_shader_stage={} max_uniform_buffers_per_shader_stage={} max_compute_workgroup_storage_size={} max_multiview_view_count={}",
        device.features(),
        limits.max_texture_dimension_2d,
        limits.max_texture_array_layers,
        limits.max_buffer_size,
        limits.max_bind_groups,
        limits.max_storage_buffers_per_shader_stage,
        limits.max_uniform_buffers_per_shader_stage,
        limits.max_compute_workgroup_storage_size,
        limits.max_multiview_view_count,
    );
}
