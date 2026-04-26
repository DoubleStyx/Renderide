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

/// Lower scores rank earlier. Stable across systems so Vulkan ICD reordering does not flip the
/// chosen adapter.
///
/// [`wgpu::PowerPreference::None`] is treated as [`wgpu::PowerPreference::HighPerformance`] so that
/// callers without an explicit preference still get the discrete GPU on hybrid systems — matches
/// Renderide's `[debug] power_preference` default.
fn power_preference_score(
    device_type: wgpu::DeviceType,
    power_preference: wgpu::PowerPreference,
) -> u8 {
    use wgpu::DeviceType::*;
    let prefer_low_power = power_preference == wgpu::PowerPreference::LowPower;
    match device_type {
        DiscreteGpu => {
            if prefer_low_power {
                1
            } else {
                0
            }
        }
        IntegratedGpu => {
            if prefer_low_power {
                0
            } else {
                1
            }
        }
        VirtualGpu => 2,
        Cpu => 3,
        Other => 4,
    }
}

/// Returns the index of the best compatible adapter, or [`None`] if none pass `is_compatible`.
///
/// Ranking uses [`power_preference_score`]; ties break on enumeration order so the result is
/// deterministic given the same adapter list.
fn pick_adapter_index<F>(
    adapters: &[wgpu::Adapter],
    is_compatible: F,
    power_preference: wgpu::PowerPreference,
) -> Option<usize>
where
    F: Fn(&wgpu::Adapter) -> bool,
{
    adapters
        .iter()
        .enumerate()
        .filter(|(_, a)| is_compatible(a))
        .min_by_key(|(i, a)| {
            (
                power_preference_score(a.get_info().device_type, power_preference),
                *i,
            )
        })
        .map(|(i, _)| i)
}

/// Logs every enumerated adapter at info level so users can see what wgpu found and why one was chosen.
fn log_adapter_candidates(adapters: &[wgpu::Adapter]) {
    if adapters.is_empty() {
        logger::warn!("wgpu adapter candidates: <none enumerated>");
        return;
    }
    for a in adapters {
        let info = a.get_info();
        logger::info!(
            "wgpu adapter candidate: {} type={:?} backend={:?} vendor=0x{:04x} device=0x{:04x}",
            info.name,
            info.device_type,
            info.backend,
            info.vendor,
            info.device,
        );
    }
}

/// Builds the [`wgpu::Instance`] used by both windowed and headless paths and returns the
/// derived [`wgpu::InstanceFlags`] for logging.
fn build_wgpu_instance(gpu_validation_layers: bool) -> (wgpu::Instance, wgpu::InstanceFlags) {
    let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
    instance_desc.backends = wgpu::Backends::all();
    let instance_flags = instance_flags_for_gpu_init(gpu_validation_layers);
    instance_desc.flags = instance_flags;
    (wgpu::Instance::new(instance_desc), instance_flags)
}

/// Enumerates adapters, logs all candidates, and returns the best match for `power_preference`.
///
/// When `surface` is [`Some`], adapters that cannot present to it are filtered out. Errors are
/// returned as [`GpuError::Adapter`] with messages distinguishing the windowed and headless paths.
async fn select_adapter(
    instance: &wgpu::Instance,
    surface: Option<&wgpu::Surface<'_>>,
    power_preference: wgpu::PowerPreference,
) -> Result<wgpu::Adapter, GpuError> {
    let adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;
    log_adapter_candidates(&adapters);
    let chosen = match surface {
        Some(s) => pick_adapter_index(&adapters, |a| a.is_surface_supported(s), power_preference),
        None => pick_adapter_index(&adapters, |_| true, power_preference),
    }
    .ok_or_else(|| {
        match surface {
        Some(_) => GpuError::Adapter(format!(
            "no surface-compatible adapter found among {} candidate(s)",
            adapters.len()
        )),
        None => GpuError::Adapter(
            "no Vulkan adapter found. \
             Install drivers for your GPU, or for software rendering install \
             `mesa-vulkan-drivers` / `vulkan-swrast` (lavapipe) and verify a Vulkan ICD is present."
                .into(),
        ),
    }
    })?;
    let adapter = adapters
        .into_iter()
        .nth(chosen)
        .ok_or_else(|| GpuError::Adapter("adapter index out of range".into()))?;
    let info = adapter.get_info();
    let label = if surface.is_some() {
        "wgpu adapter selected"
    } else {
        "wgpu adapter selected (headless)"
    };
    logger::info!(
        "{label}: {} type={:?} backend={:?} (preference={:?})",
        info.name,
        info.device_type,
        info.backend,
        power_preference,
    );
    Ok(adapter)
}

impl GpuContext {
    /// Asynchronously builds GPU state for `window`.
    ///
    /// `gpu_validation_layers` selects whether to request backend validation before `WGPU_*` env
    /// overrides; see [`crate::gpu::instance_flags_for_gpu_init`]. `power_preference` is sourced
    /// from [`crate::config::DebugSettings::power_preference`] and used to rank enumerated
    /// adapters (discrete first when [`wgpu::PowerPreference::HighPerformance`], integrated first
    /// when [`wgpu::PowerPreference::LowPower`]).
    pub async fn new(
        window: Arc<Window>,
        vsync: bool,
        gpu_validation_layers: bool,
        power_preference: wgpu::PowerPreference,
    ) -> Result<Self, GpuError> {
        let (instance, instance_flags) = build_wgpu_instance(gpu_validation_layers);

        // wgpu infers `Surface<'static>` directly from an `Arc<Window>` source, so no lifetime
        // extension is needed; the `Arc` is then stored alongside the surface in `GpuContext`.
        let surface_safe: wgpu::Surface<'static> = instance
            .create_surface(window.clone())
            .map_err(|e| GpuError::Surface(format!("{e:?}")))?;

        let adapter = select_adapter(&instance, Some(&surface_safe), power_preference).await?;

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
        // wgpu infers `Surface<'static>` directly from an `Arc<Window>` source, so no lifetime
        // extension is needed; the `Arc` is then stored alongside the surface in `GpuContext`.
        let surface_safe: wgpu::Surface<'static> = instance
            .create_surface(window.clone())
            .map_err(|e| GpuError::Surface(format!("{e:?}")))?;
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

#[cfg(test)]
mod power_preference_score_tests {
    use super::power_preference_score;
    use wgpu::{DeviceType, PowerPreference};

    #[test]
    fn high_performance_prefers_discrete_over_integrated() {
        assert!(
            power_preference_score(DeviceType::DiscreteGpu, PowerPreference::HighPerformance)
                < power_preference_score(
                    DeviceType::IntegratedGpu,
                    PowerPreference::HighPerformance,
                )
        );
    }

    #[test]
    fn low_power_prefers_integrated_over_discrete() {
        assert!(
            power_preference_score(DeviceType::IntegratedGpu, PowerPreference::LowPower)
                < power_preference_score(DeviceType::DiscreteGpu, PowerPreference::LowPower)
        );
    }

    #[test]
    fn cpu_and_other_rank_below_real_gpus() {
        for pref in [PowerPreference::HighPerformance, PowerPreference::LowPower] {
            let cpu = power_preference_score(DeviceType::Cpu, pref);
            let other = power_preference_score(DeviceType::Other, pref);
            let discrete = power_preference_score(DeviceType::DiscreteGpu, pref);
            let integrated = power_preference_score(DeviceType::IntegratedGpu, pref);
            assert!(cpu > discrete && cpu > integrated, "Cpu rank too high");
            assert!(
                other > discrete && other > integrated,
                "Other rank too high"
            );
        }
    }

    #[test]
    fn virtual_gpu_ranks_below_real_gpus_but_above_cpu() {
        for pref in [PowerPreference::HighPerformance, PowerPreference::LowPower] {
            let virt = power_preference_score(DeviceType::VirtualGpu, pref);
            let cpu = power_preference_score(DeviceType::Cpu, pref);
            let discrete = power_preference_score(DeviceType::DiscreteGpu, pref);
            assert!(virt > discrete);
            assert!(virt < cpu);
        }
    }
}
