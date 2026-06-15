//! Headless [`GpuContext::new_headless`] constructor.

use std::sync::Arc;

use super::super::super::adapter::device::{request_device_for_adapter, try_gpu_profiler};
use super::super::super::adapter::features::adapter_render_features_intersection;
use super::super::super::adapter::msaa_support::MsaaSupport;
use super::super::super::limits::GpuLimits;
use super::super::super::sync::device_health::GpuDeviceHealth;
use super::super::super::sync::mapped_buffer_health::GpuMappedBufferHealth;
use super::super::{GpuContext, GpuError};
use super::shared::{
    GpuContextParts, GpuRuntimeHandles, adapter_info_field_or_unreported, assemble_context,
    log_device_capability_summary, select_headless_adapter_with_fallback,
};
use crate::config::GraphicsApiSetting;
use crate::gpu::flight_recorder::GpuFlightRecorder;
use crate::gpu::submission_state::GpuSubmissionState;

impl GpuContext {
    /// Builds a GPU stack with **no surface** for headless offscreen rendering (CI / golden tests).
    ///
    /// `--headless` means no window and no swapchain; adapter selection follows normal wgpu rules
    /// (`Backends::all()`, no forced fallback). Developer machines typically use a discrete or
    /// integrated GPU; CI runners with only Mesa lavapipe installed still pick the software Vulkan
    /// ICD automatically.
    ///
    /// The synthesized [`wgpu::SurfaceConfiguration`] has `format = Rgba8UnormSrgb` and the
    /// requested extent so the material system and render graph compile pipelines unchanged.
    ///
    /// `max_frame_latency` populates
    /// [`wgpu::SurfaceConfiguration::desired_maximum_frame_latency`] for parity with the
    /// windowed path; headless rendering has no swapchain so the value mostly affects internal
    /// frame-resource allocation.
    ///
    /// `graphics_api` follows the same startup-only first-attempt and auto-fallback policy as the
    /// windowed constructor.
    pub async fn new_headless(
        width: u32,
        height: u32,
        max_frame_latency: u32,
        gpu_validation_layers: bool,
        power_preference: wgpu::PowerPreference,
        graphics_api: GraphicsApiSetting,
    ) -> Result<Self, GpuError> {
        logger::info!(
            "GPU startup request (headless): graphics_api={} validation={} power_preference={:?} extent={}x{} max_frame_latency={}",
            graphics_api.as_persist_str(),
            gpu_validation_layers,
            power_preference,
            width,
            height,
            max_frame_latency,
        );
        let selection = select_headless_adapter_with_fallback(
            graphics_api,
            gpu_validation_layers,
            power_preference,
        )
        .await?;
        let selected_graphics_api = selection.graphics_api;
        let active_backends = selection.active_backends;
        let instance_flags = selection.instance_flags;
        let adapter = selection.adapter;

        let mapped_buffer_health = Arc::new(GpuMappedBufferHealth::new());
        let device_health = Arc::new(GpuDeviceHealth::new());
        let flight_recorder = Arc::new(GpuFlightRecorder::new());
        let required_features = adapter_render_features_intersection(&adapter);
        let (device, queue) = request_device_for_adapter(
            &adapter,
            required_features,
            Arc::clone(&mapped_buffer_health),
            Arc::clone(&device_health),
            Arc::clone(&flight_recorder),
        )
        .await?;

        let limits = GpuLimits::try_new(device.as_ref(), &adapter)?;

        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let config = headless_surface_config(width, height, max_frame_latency, format);
        let adapter_info = adapter.get_info();
        let depth_stencil_format = crate::gpu::main_forward_depth_stencil_format(required_features);
        let msaa = MsaaSupport::discover(
            &adapter,
            format,
            depth_stencil_format,
            required_features,
            "GPU (headless)",
        );
        log_headless_gpu_selection_summary(
            &adapter_info,
            selected_graphics_api,
            active_backends,
            instance_flags,
            &config,
            &msaa,
        );
        log_device_capability_summary("GPU (headless)", device.as_ref());
        let gpu_profiler = try_gpu_profiler(
            &adapter,
            device.as_ref(),
            &queue,
            "GPU profiler unavailable (headless): adapter lacks TIMESTAMP_QUERY; \
             Tracy GPU timeline will be empty (CPU spans still work)",
        );
        let runtime = GpuRuntimeHandles::new(
            Arc::clone(&device),
            Arc::new(queue),
            Arc::clone(&mapped_buffer_health),
            Arc::clone(&device_health),
            Arc::clone(&flight_recorder),
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
            adapter_info,
            msaa,
            limits,
            device,
            queue: runtime.queue,
            gpu_queue_access_gate: runtime.gpu_queue_access_gate,
            mapped_buffer_health,
            device_health,
            flight_recorder,
            surface: None,
            config,
            supported_present_modes: Vec::new(),
            window: None,
        }))
    }
}

fn log_headless_gpu_selection_summary(
    adapter_info: &wgpu::AdapterInfo,
    graphics_api: GraphicsApiSetting,
    active_backends: wgpu::Backends,
    instance_flags: wgpu::InstanceFlags,
    config: &wgpu::SurfaceConfiguration,
    msaa: &MsaaSupport,
) {
    logger::info!(
        "GPU (headless): adapter={} type={:?} backend={:?} vendor={:#010x} device={:#010x} pci_bus_id={} driver={} driver_info={} graphics_api={} active_backends={:?} extent={}x{} format={:?} instance_flags={:?} \
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
        graphics_api.as_persist_str(),
        active_backends,
        config.width,
        config.height,
        config.format,
        instance_flags,
        &msaa.desktop,
        msaa.desktop_max(),
        &msaa.stereo,
        msaa.stereo_max(),
    );
}

/// Builds the synthetic surface config used by the headless offscreen path.
fn headless_surface_config(
    width: u32,
    height: u32,
    max_frame_latency: u32,
    format: wgpu::TextureFormat,
) -> wgpu::SurfaceConfiguration {
    wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        format,
        width: width.max(1),
        height: height.max(1),
        present_mode: wgpu::PresentMode::AutoNoVsync,
        desired_maximum_frame_latency: max_frame_latency,
        alpha_mode: wgpu::CompositeAlphaMode::Opaque,
        view_formats: Vec::new(),
    }
}
