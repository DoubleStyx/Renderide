//! [`GpuContext`] constructors: window-backed, headless, and OpenXR-bootstrap variants.

use std::sync::{Arc, Mutex};

use winit::window::Window;

use super::super::adapter::device::{
    install_uncaptured_error_handler, request_device_for_adapter, try_gpu_profiler,
};
use super::super::adapter::features::adapter_render_features_intersection;
use super::super::adapter::msaa_support::MsaaSupport;
use super::super::adapter::selection::{build_wgpu_instance, select_adapter};
use super::super::frame_cpu_gpu_timing::{FrameCpuGpuTiming, FrameCpuGpuTimingHandle};
use super::super::limits::GpuLimits;
use super::{GpuContext, GpuError, PrimaryOffscreenTargets};
use crate::config::VsyncMode;
use crate::gpu::submission_state::GpuSubmissionState;

/// Runtime handles derived from a queue and shared by all GPU construction paths.
struct GpuRuntimeHandles {
    /// Shared queue handle stored on [`GpuContext`].
    queue: Arc<wgpu::Queue>,
    /// Driver-thread submit gate paired with [`Self::queue`].
    gpu_queue_access_gate: super::super::GpuQueueAccessGate,
    /// Dedicated submit/present worker.
    driver_thread: super::super::driver_thread::DriverThread,
    /// CPU/GPU frame timing accumulator.
    frame_timing: FrameCpuGpuTimingHandle,
    /// Latest flattened GPU pass timings for the HUD.
    latest_gpu_pass_timings: Arc<Mutex<Vec<crate::profiling::GpuPassEntry>>>,
}

impl GpuRuntimeHandles {
    /// Builds the driver-thread and timing handles for a queue.
    fn new(queue: Arc<wgpu::Queue>) -> Result<Self, GpuError> {
        let gpu_queue_access_gate = super::super::GpuQueueAccessGate::new();
        let driver_thread = super::super::driver_thread::DriverThread::new(
            Arc::clone(&queue),
            gpu_queue_access_gate.clone(),
        )
        .map_err(GpuError::DriverThreadSpawn)?;
        Ok(Self {
            queue,
            gpu_queue_access_gate,
            driver_thread,
            frame_timing: Arc::new(Mutex::new(FrameCpuGpuTiming::default())),
            latest_gpu_pass_timings: Arc::new(Mutex::new(Vec::new())),
        })
    }
}

/// Inputs that differ between the three [`GpuContext`] construction paths.
struct GpuContextParts {
    /// Submission, timing, and profiling state.
    submission: GpuSubmissionState,
    /// Adapter metadata captured at construction.
    adapter_info: wgpu::AdapterInfo,
    /// MSAA support lists for desktop and stereo paths.
    msaa: MsaaSupport,
    /// Effective limits and derived caps.
    limits: Arc<GpuLimits>,
    /// Logical device.
    device: Arc<wgpu::Device>,
    /// Submission queue.
    queue: Arc<wgpu::Queue>,
    /// Shared write-texture/submit gate.
    gpu_queue_access_gate: super::super::GpuQueueAccessGate,
    /// Optional window-backed surface.
    surface: Option<wgpu::Surface<'static>>,
    /// Active surface/offscreen configuration.
    config: wgpu::SurfaceConfiguration,
    /// Surface present modes.
    supported_present_modes: Vec<wgpu::PresentMode>,
    /// Optional window owner.
    window: Option<Arc<Window>>,
}

/// Builds the common [`GpuContext`] field set once all path-specific resources are ready.
fn assemble_context(parts: GpuContextParts) -> GpuContext {
    GpuContext {
        submission: parts.submission,
        adapter_info: parts.adapter_info,
        msaa_supported_sample_counts: parts.msaa.desktop,
        msaa_supported_sample_counts_stereo: parts.msaa.stereo,
        swapchain_msaa_effective: 1,
        swapchain_msaa_requested_stereo: 1,
        swapchain_msaa_effective_stereo: 1,
        limits: parts.limits,
        device: parts.device,
        queue: parts.queue,
        gpu_queue_access_gate: parts.gpu_queue_access_gate,
        surface: parts.surface,
        config: parts.config,
        supported_present_modes: parts.supported_present_modes,
        window: parts.window,
        depth_attachment: None,
        depth_extent_px: (0, 0),
        primary_offscreen: Option::<PrimaryOffscreenTargets>::None,
    }
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
    /// [`VsyncMode::resolve_present_mode`] (so e.g. [`VsyncMode::On`] picks `Mailbox` when
    /// available rather than the deeper-queue plain `Fifo`).
    ///
    /// `max_frame_latency` is the initial value for
    /// [`wgpu::SurfaceConfiguration::desired_maximum_frame_latency`]. Pass the resolved value
    /// from [`crate::config::RenderingSettings::resolved_max_frame_latency`]. The default of `2`
    /// allows CPU recording for frame N+1 to overlap with GPU work for frame N; lowering to `1`
    /// reduces input latency at the cost of stalls inside [`wgpu::Surface::get_current_texture`].
    /// The setting is live-tunable via [`crate::gpu::GpuContext::set_max_frame_latency`].
    pub async fn new(
        window: Arc<Window>,
        vsync: VsyncMode,
        max_frame_latency: u32,
        gpu_validation_layers: bool,
        power_preference: wgpu::PowerPreference,
    ) -> Result<Self, GpuError> {
        let (instance, instance_flags) = build_wgpu_instance(gpu_validation_layers);

        // `Arc<Window>` is `Into<SurfaceTarget<'static>>`, so the returned `Surface` is
        // already `'static` — no `transmute` is required to extend the borrow.
        let surface_safe: wgpu::Surface<'static> = instance
            .create_surface(window.clone())
            .map_err(|e| GpuError::Surface(format!("{e:?}")))?;

        let adapter = select_adapter(&instance, Some(&surface_safe), power_preference).await?;

        let required_features = adapter_render_features_intersection(&adapter);
        let (device, queue) = request_device_for_adapter(&adapter, required_features).await?;

        let limits = GpuLimits::try_new(device.as_ref(), &adapter)?;
        let size = window.inner_size();
        let supported_present_modes = surface_safe.get_capabilities(&adapter).present_modes;
        let mut config = surface_safe
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .ok_or(GpuError::SurfaceUnsupported)?;
        config.present_mode = vsync.resolve_present_mode(&supported_present_modes);
        config.desired_maximum_frame_latency = max_frame_latency;
        surface_safe.configure(&device, &config);

        let adapter_info = adapter.get_info();
        let depth_stencil_format = crate::gpu::main_forward_depth_stencil_format(required_features);
        let msaa = MsaaSupport::discover(
            &adapter,
            config.format,
            depth_stencil_format,
            required_features,
            "GPU",
        );
        logger::info!(
            "GPU: adapter={} backend={:?} vsync={:?} present_mode={:?} \
             supported_present_modes={:?} desired_maximum_frame_latency={} instance_flags={:?} \
             msaa_supported_sample_counts={:?} msaa_max_sample_count={} \
             msaa_supported_sample_counts_stereo={:?} msaa_max_sample_count_stereo={}",
            adapter_info.name,
            adapter_info.backend,
            vsync,
            config.present_mode,
            supported_present_modes,
            config.desired_maximum_frame_latency,
            instance_flags,
            &msaa.desktop,
            msaa.desktop_max(),
            &msaa.stereo,
            msaa.stereo_max()
        );

        let gpu_profiler = try_gpu_profiler(
            &adapter,
            device.as_ref(),
            &queue,
            "GPU profiler unavailable: adapter lacks TIMESTAMP_QUERY; \
             Tracy GPU timeline will be empty (CPU spans still work)",
        );
        let runtime = GpuRuntimeHandles::new(Arc::new(queue))?;
        let submission = GpuSubmissionState::new(
            runtime.driver_thread,
            runtime.frame_timing,
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
            surface: Some(surface_safe),
            config,
            supported_present_modes,
            window: Some(window),
        }))
    }

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
    /// frame-resource allocation. Pass the resolved value from
    /// [`crate::config::RenderingSettings::resolved_max_frame_latency`].
    pub async fn new_headless(
        width: u32,
        height: u32,
        max_frame_latency: u32,
        gpu_validation_layers: bool,
        power_preference: wgpu::PowerPreference,
    ) -> Result<Self, GpuError> {
        let (instance, instance_flags) = build_wgpu_instance(gpu_validation_layers);

        let adapter = select_adapter(&instance, None, power_preference).await?;

        let required_features = adapter_render_features_intersection(&adapter);
        let (device, queue) = request_device_for_adapter(&adapter, required_features).await?;

        let limits = GpuLimits::try_new(device.as_ref(), &adapter)?;

        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format,
            width: width.max(1),
            height: height.max(1),
            present_mode: wgpu::PresentMode::AutoNoVsync,
            desired_maximum_frame_latency: max_frame_latency,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            view_formats: Vec::new(),
        };
        let adapter_info = adapter.get_info();
        let depth_stencil_format = crate::gpu::main_forward_depth_stencil_format(required_features);
        let msaa = MsaaSupport::discover(
            &adapter,
            format,
            depth_stencil_format,
            required_features,
            "GPU (headless)",
        );
        logger::info!(
            "GPU (headless): adapter={} backend={:?} extent={}x{} format={:?} instance_flags={:?} \
             msaa_supported_sample_counts={:?} msaa_max_sample_count={} \
             msaa_supported_sample_counts_stereo={:?} msaa_max_sample_count_stereo={}",
            adapter_info.name,
            adapter_info.backend,
            config.width,
            config.height,
            config.format,
            instance_flags,
            &msaa.desktop,
            msaa.desktop_max(),
            &msaa.stereo,
            msaa.stereo_max(),
        );
        let gpu_profiler = try_gpu_profiler(
            &adapter,
            device.as_ref(),
            &queue,
            "GPU profiler unavailable (headless): adapter lacks TIMESTAMP_QUERY; \
             Tracy GPU timeline will be empty (CPU spans still work)",
        );
        let runtime = GpuRuntimeHandles::new(Arc::new(queue))?;
        let submission = GpuSubmissionState::new(
            runtime.driver_thread,
            runtime.frame_timing,
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
            surface: None,
            config,
            supported_present_modes: Vec::new(),
            window: None,
        }))
    }

    /// Builds GPU state using an existing wgpu instance/device from OpenXR bootstrap (mirror window).
    ///
    /// The mirror surface uses the same capability-aware [`VsyncMode`] mapping and
    /// `max_frame_latency` semantics as the desktop constructor so windowed presentation
    /// behaves consistently across desktop and VR startup paths.
    pub fn new_from_openxr_bootstrap(
        instance: &wgpu::Instance,
        adapter: &wgpu::Adapter,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        window: Arc<Window>,
        vsync: VsyncMode,
        max_frame_latency: u32,
    ) -> Result<Self, GpuError> {
        install_uncaptured_error_handler(device.as_ref());
        // `Arc<Window>` is `Into<SurfaceTarget<'static>>`, so the returned `Surface` is
        // already `'static` — no `transmute` is required to extend the borrow.
        let surface_safe: wgpu::Surface<'static> = instance
            .create_surface(window.clone())
            .map_err(|e| GpuError::Surface(format!("{e:?}")))?;
        let size = window.inner_size();
        let supported_present_modes = surface_safe.get_capabilities(adapter).present_modes;
        let mut config = surface_safe
            .get_default_config(adapter, size.width.max(1), size.height.max(1))
            .ok_or(GpuError::SurfaceUnsupported)?;
        config.present_mode = vsync.resolve_present_mode(&supported_present_modes);
        config.desired_maximum_frame_latency = max_frame_latency;
        surface_safe.configure(&device, &config);
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
            "GPU (OpenXR path): adapter={} backend={:?} vsync={:?} present_mode={:?} \
             supported_present_modes={:?} desired_maximum_frame_latency={} \
             msaa_supported_sample_counts={:?} msaa_max_sample_count={} \
             msaa_supported_sample_counts_stereo={:?} msaa_max_sample_count_stereo={}",
            adapter_info.name,
            adapter_info.backend,
            vsync,
            config.present_mode,
            supported_present_modes,
            config.desired_maximum_frame_latency,
            &msaa.desktop,
            msaa.desktop_max(),
            &msaa.stereo,
            msaa.stereo_max()
        );
        let gpu_profiler = try_gpu_profiler(
            adapter,
            device.as_ref(),
            queue.as_ref(),
            "GPU profiler unavailable (OpenXR path): adapter lacks \
             TIMESTAMP_QUERY; Tracy GPU timeline will be empty",
        );
        let runtime = GpuRuntimeHandles::new(queue)?;
        let submission = GpuSubmissionState::new(
            runtime.driver_thread,
            runtime.frame_timing,
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
            surface: Some(surface_safe),
            config,
            supported_present_modes,
            window: Some(window),
        }))
    }
}
