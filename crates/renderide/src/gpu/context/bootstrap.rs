//! Desktop and OpenXR-mirror GPU bootstrap: instance, adapter, device, surface, and MSAA tier capture.

use std::sync::{Arc, Mutex};

use winit::window::Window;

use crate::gpu::instance_limits::{instance_flags_for_gpu_init, required_limits_for_adapter};
use crate::gpu::limits::GpuLimits;

use super::msaa_tiers::{msaa_supported_sample_counts, msaa_supported_sample_counts_stereo};
use super::GpuError;

/// Fields assembled by bootstrap code for [`super::GpuContext`], excluding per-frame timing state.
pub(super) struct GpuBootstrapParts {
    pub(super) adapter_info: wgpu::AdapterInfo,
    pub(super) msaa_supported_sample_counts: Vec<u32>,
    pub(super) msaa_supported_sample_counts_stereo: Vec<u32>,
    pub(super) limits: Arc<GpuLimits>,
    pub(super) device: Arc<wgpu::Device>,
    pub(super) queue: Arc<Mutex<wgpu::Queue>>,
    pub(super) surface: wgpu::Surface<'static>,
    pub(super) config: wgpu::SurfaceConfiguration,
}

fn surface_into_static(
    instance: &wgpu::Instance,
    window: Arc<Window>,
) -> Result<wgpu::Surface<'static>, GpuError> {
    let surface = instance
        .create_surface(window)
        .map_err(|e| GpuError::Surface(format!("{e:?}")))?;
    // SAFETY: `GpuContext` is stored next to the owning `Arc<Window>` in the app; the surface must not
    // outlive that window. wgpu ties the surface lifetime to the borrow used at `create_surface`; we
    // erase it to `'static` so `GpuContext` can move independently of local borrows.
    let surface_safe: wgpu::Surface<'static> = unsafe { std::mem::transmute(surface) };
    Ok(surface_safe)
}

/// Asynchronously builds adapter, device, surface, and swapchain for the desktop path.
pub(super) async fn bootstrap_desktop(
    window: Arc<Window>,
    vsync: bool,
    gpu_validation_layers: bool,
) -> Result<GpuBootstrapParts, GpuError> {
    let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
    instance_desc.backends = wgpu::Backends::all();
    let instance_flags = instance_flags_for_gpu_init(gpu_validation_layers);
    instance_desc.flags = instance_flags;
    let instance = wgpu::Instance::new(instance_desc);

    let surface_safe = surface_into_static(&instance, window.clone())?;

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface_safe),
            force_fallback_adapter: false,
        })
        .await
        .map_err(|e| GpuError::Adapter(format!("{e:?}")))?;

    let compression = wgpu::Features::TEXTURE_COMPRESSION_BC
        | wgpu::Features::TEXTURE_COMPRESSION_ETC2
        | wgpu::Features::TEXTURE_COMPRESSION_ASTC;
    // FLOAT32_FILTERABLE: without it, Rgba32Float is unfilterable and cannot bind to embedded
    // material layouts that use filterable float texture + Filtering samplers.
    let optional_float32_filterable = wgpu::Features::FLOAT32_FILTERABLE;
    // TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES: without it, wgpu restricts format capabilities
    // (including MSAA sample counts) to the WebGPU baseline, which is much narrower than what
    // the GPU actually supports. Enabling this unlocks the hardware-reported feature set.
    let adapter_format_features = wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
    // DEPTH32FLOAT_STENCIL8: enables `Depth32FloatStencil8`, used as the main forward depth
    // attachment when the adapter supports it (see `main_forward_depth_stencil_format`).
    let optional_depth32_stencil8 = wgpu::Features::DEPTH32FLOAT_STENCIL8;
    // MULTISAMPLE_ARRAY: allows creating multisampled 2D array textures and views. Required for
    // stereo MSAA in the OpenXR multiview path; harmless for the desktop swapchain path.
    let multisample_array = wgpu::Features::MULTISAMPLE_ARRAY;
    let required_features = adapter.features()
        & (compression
            | optional_float32_filterable
            | adapter_format_features
            | optional_depth32_stencil8
            | multisample_array);

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("renderide-skeleton"),
            required_features,
            required_limits: required_limits_for_adapter(&adapter),
            ..Default::default()
        })
        .await
        .map_err(|e| GpuError::Device(format!("{e:?}")))?;

    let device = Arc::new(device);
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
    let msaa_supported_sample_counts = msaa_supported_sample_counts(&adapter, config.format);
    let msaa_max = msaa_supported_sample_counts.last().copied().unwrap_or(1);
    let msaa_supported_sample_counts_stereo =
        msaa_supported_sample_counts_stereo(&adapter, config.format, required_features);
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

    Ok(GpuBootstrapParts {
        adapter_info,
        msaa_supported_sample_counts,
        msaa_supported_sample_counts_stereo,
        limits,
        device,
        queue: Arc::new(Mutex::new(queue)),
        surface: surface_safe,
        config,
    })
}

/// Builds surface and swapchain state using an existing wgpu instance/device from OpenXR bootstrap (mirror window).
pub(super) fn bootstrap_openxr_mirror(
    instance: &wgpu::Instance,
    adapter: &wgpu::Adapter,
    device: Arc<wgpu::Device>,
    queue: Arc<Mutex<wgpu::Queue>>,
    window: Arc<Window>,
    vsync: bool,
) -> Result<GpuBootstrapParts, GpuError> {
    let size = window.inner_size();
    let surface_safe = surface_into_static(instance, window.clone())?;
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
    let msaa_supported_sample_counts = msaa_supported_sample_counts(adapter, config.format);
    let msaa_max = msaa_supported_sample_counts.last().copied().unwrap_or(1);
    let msaa_supported_sample_counts_stereo =
        msaa_supported_sample_counts_stereo(adapter, config.format, device.features());
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
    Ok(GpuBootstrapParts {
        adapter_info,
        msaa_supported_sample_counts,
        msaa_supported_sample_counts_stereo,
        limits,
        device,
        queue,
        surface: surface_safe,
        config,
    })
}
