//! [`GpuContext`] constructors: window-backed, headless, and OpenXR-bootstrap variants.

use std::sync::{Arc, Mutex};

use winit::window::Window;

use super::super::frame_cpu_gpu_timing::FrameCpuGpuTiming;
use super::super::instance_limits::instance_flags_for_gpu_init;
use super::super::limits::GpuLimits;
use super::{
    adapter_render_features_intersection, install_uncaptured_error_handler,
    msaa_supported_sample_counts, msaa_supported_sample_counts_stereo, request_device_for_adapter,
    GpuContext, GpuError,
};

impl GpuContext {
    /// Asynchronously builds GPU state for `window`.
    ///
    /// `gpu_validation_layers` selects whether to request backend validation before `WGPU_*` env
    /// overrides; see [`crate::gpu::instance_flags_for_gpu_init`].
    pub async fn new(
        window: Arc<Window>,
        vsync: bool,
        gpu_validation_layers: bool,
    ) -> Result<Self, GpuError> {
        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_desc.backends = wgpu::Backends::all();
        let instance_flags = instance_flags_for_gpu_init(gpu_validation_layers);
        instance_desc.flags = instance_flags;
        let instance = wgpu::Instance::new(instance_desc);

        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| GpuError::Surface(format!("{e:?}")))?;

        // SAFETY: the window `Arc` is kept alive for the lifetime of `GpuContext` (stored in
        // `Self`), so extending the surface's borrow to `'static` is sound — the underlying
        // handle outlives every use of `surface_safe`.
        let surface_safe: wgpu::Surface<'static> = unsafe { std::mem::transmute(surface) };

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface_safe),
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| GpuError::Adapter(format!("{e:?}")))?;

        let required_features = adapter_render_features_intersection(&adapter);
        let (device, queue) = request_device_for_adapter(&adapter, required_features).await?;

        let limits = GpuLimits::try_new(device.as_ref(), &adapter)?;
        let size = window.inner_size();
        let mut config = surface_safe
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .ok_or(GpuError::SurfaceUnsupported)?;
        config.present_mode = if vsync {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        };
        surface_safe.configure(&device, &config);

        let adapter_info = adapter.get_info();
        let depth_stencil_format =
            crate::render_graph::main_forward_depth_stencil_format(required_features);
        let msaa_supported_sample_counts =
            msaa_supported_sample_counts(&adapter, config.format, depth_stencil_format);
        let msaa_max = msaa_supported_sample_counts.last().copied().unwrap_or(1);
        let msaa_supported_sample_counts_stereo = msaa_supported_sample_counts_stereo(
            &adapter,
            config.format,
            depth_stencil_format,
            required_features,
        );
        let msaa_max_stereo = msaa_supported_sample_counts_stereo
            .last()
            .copied()
            .unwrap_or(1);
        logger::info!(
            "GPU: adapter={} backend={:?} present_mode={:?} instance_flags={:?} \
             msaa_supported_sample_counts={:?} msaa_max_sample_count={} \
             msaa_supported_sample_counts_stereo={:?} msaa_max_sample_count_stereo={}",
            adapter_info.name,
            adapter_info.backend,
            config.present_mode,
            instance_flags,
            msaa_supported_sample_counts,
            msaa_max,
            msaa_supported_sample_counts_stereo,
            msaa_max_stereo
        );

        let gpu_profiler =
            crate::profiling::GpuProfilerHandle::try_new(&adapter, device.as_ref(), &queue);
        if cfg!(feature = "tracy") && gpu_profiler.is_none() {
            logger::warn!(
                "GPU profiler unavailable: adapter lacks TIMESTAMP_QUERY; \
                 Tracy GPU timeline will be empty (CPU spans still work)"
            );
        }
        let queue = Arc::new(queue);
        let write_texture_submit_gate = super::super::WriteTextureSubmitGate::new();
        let driver_thread = super::super::driver_thread::DriverThread::new(
            Arc::clone(&queue),
            write_texture_submit_gate.clone(),
        );
        Ok(Self {
            driver_thread,
            adapter_info,
            msaa_supported_sample_counts,
            msaa_supported_sample_counts_stereo,
            swapchain_msaa_effective: 1,
            swapchain_msaa_requested_stereo: 1,
            swapchain_msaa_effective_stereo: 1,
            limits,
            device,
            queue,
            write_texture_submit_gate,
            surface: Some(surface_safe),
            config,
            window: Some(window),
            depth_attachment: None,
            depth_extent_px: (0, 0),
            primary_offscreen: None,
            frame_timing: Arc::new(Mutex::new(FrameCpuGpuTiming::default())),
            gpu_profiler,
            latest_gpu_pass_timings: Arc::new(Mutex::new(Vec::new())),
        })
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
    pub async fn new_headless(
        width: u32,
        height: u32,
        gpu_validation_layers: bool,
    ) -> Result<Self, GpuError> {
        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_desc.backends = wgpu::Backends::all();
        let instance_flags = instance_flags_for_gpu_init(gpu_validation_layers);
        instance_desc.flags = instance_flags;
        let instance = wgpu::Instance::new(instance_desc);

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| {
                GpuError::Adapter(format!(
                    "no Vulkan adapter found ({e:?}). \
                     Install drivers for your GPU, or for software rendering install \
                     `mesa-vulkan-drivers` / `vulkan-swrast` (lavapipe) and verify a Vulkan ICD is present."
                ))
            })?;

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
            desired_maximum_frame_latency: 1,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            view_formats: Vec::new(),
        };
        let adapter_info = adapter.get_info();
        let depth_stencil_format =
            crate::render_graph::main_forward_depth_stencil_format(required_features);
        let msaa_supported_sample_counts =
            msaa_supported_sample_counts(&adapter, format, depth_stencil_format);
        let msaa_supported_sample_counts_stereo = msaa_supported_sample_counts_stereo(
            &adapter,
            format,
            depth_stencil_format,
            required_features,
        );
        logger::info!(
            "GPU (headless): adapter={} backend={:?} extent={}x{} format={:?} instance_flags={:?}",
            adapter_info.name,
            adapter_info.backend,
            config.width,
            config.height,
            config.format,
            instance_flags,
        );
        let gpu_profiler =
            crate::profiling::GpuProfilerHandle::try_new(&adapter, device.as_ref(), &queue);
        if cfg!(feature = "tracy") && gpu_profiler.is_none() {
            logger::warn!(
                "GPU profiler unavailable (headless): adapter lacks TIMESTAMP_QUERY; \
                 Tracy GPU timeline will be empty (CPU spans still work)"
            );
        }
        let queue = Arc::new(queue);
        let write_texture_submit_gate = super::super::WriteTextureSubmitGate::new();
        let driver_thread = super::super::driver_thread::DriverThread::new(
            Arc::clone(&queue),
            write_texture_submit_gate.clone(),
        );
        Ok(Self {
            driver_thread,
            adapter_info,
            msaa_supported_sample_counts,
            msaa_supported_sample_counts_stereo,
            swapchain_msaa_effective: 1,
            swapchain_msaa_requested_stereo: 1,
            swapchain_msaa_effective_stereo: 1,
            limits,
            device,
            queue,
            write_texture_submit_gate,
            surface: None,
            config,
            window: None,
            depth_attachment: None,
            depth_extent_px: (0, 0),
            primary_offscreen: None,
            frame_timing: Arc::new(Mutex::new(FrameCpuGpuTiming::default())),
            gpu_profiler,
            latest_gpu_pass_timings: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Builds GPU state using an existing wgpu instance/device from OpenXR bootstrap (mirror window).
    pub fn new_from_openxr_bootstrap(
        instance: &wgpu::Instance,
        adapter: &wgpu::Adapter,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        window: Arc<Window>,
        vsync: bool,
    ) -> Result<Self, GpuError> {
        install_uncaptured_error_handler(device.as_ref());
        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| GpuError::Surface(format!("{e:?}")))?;
        // SAFETY: the window `Arc` is owned by `GpuContext` for the rest of the process, so the
        // borrow attached to `surface` outlives every access — the `'static` cast is sound.
        let surface_safe: wgpu::Surface<'static> = unsafe { std::mem::transmute(surface) };
        let size = window.inner_size();
        let mut config = surface_safe
            .get_default_config(adapter, size.width.max(1), size.height.max(1))
            .ok_or(GpuError::SurfaceUnsupported)?;
        config.present_mode = if vsync {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        };
        surface_safe.configure(&device, &config);
        let adapter_info = adapter.get_info();
        let limits = GpuLimits::try_new(device.as_ref(), adapter)?;
        let depth_stencil_format =
            crate::render_graph::main_forward_depth_stencil_format(device.features());
        let msaa_supported_sample_counts =
            msaa_supported_sample_counts(adapter, config.format, depth_stencil_format);
        let msaa_max = msaa_supported_sample_counts.last().copied().unwrap_or(1);
        let msaa_supported_sample_counts_stereo = msaa_supported_sample_counts_stereo(
            adapter,
            config.format,
            depth_stencil_format,
            device.features(),
        );
        let msaa_max_stereo = msaa_supported_sample_counts_stereo
            .last()
            .copied()
            .unwrap_or(1);
        logger::info!(
            "GPU (OpenXR path): adapter={} backend={:?} present_mode={:?} \
             msaa_supported_sample_counts={:?} msaa_max_sample_count={} \
             msaa_supported_sample_counts_stereo={:?} msaa_max_sample_count_stereo={}",
            adapter_info.name,
            adapter_info.backend,
            config.present_mode,
            msaa_supported_sample_counts,
            msaa_max,
            msaa_supported_sample_counts_stereo,
            msaa_max_stereo
        );
        let gpu_profiler =
            crate::profiling::GpuProfilerHandle::try_new(adapter, device.as_ref(), queue.as_ref());
        if cfg!(feature = "tracy") && gpu_profiler.is_none() {
            logger::warn!(
                "GPU profiler unavailable (OpenXR path): adapter lacks \
                 TIMESTAMP_QUERY; Tracy GPU timeline will be empty"
            );
        }
        let write_texture_submit_gate = super::super::WriteTextureSubmitGate::new();
        let driver_thread = super::super::driver_thread::DriverThread::new(
            Arc::clone(&queue),
            write_texture_submit_gate.clone(),
        );
        Ok(Self {
            driver_thread,
            adapter_info,
            msaa_supported_sample_counts,
            msaa_supported_sample_counts_stereo,
            swapchain_msaa_effective: 1,
            swapchain_msaa_requested_stereo: 1,
            swapchain_msaa_effective_stereo: 1,
            limits,
            device,
            queue,
            write_texture_submit_gate,
            surface: Some(surface_safe),
            config,
            window: Some(window),
            depth_attachment: None,
            depth_extent_px: (0, 0),
            primary_offscreen: None,
            frame_timing: Arc::new(Mutex::new(FrameCpuGpuTiming::default())),
            gpu_profiler,
            latest_gpu_pass_timings: Arc::new(Mutex::new(Vec::new())),
        })
    }
}
