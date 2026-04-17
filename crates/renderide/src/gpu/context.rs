//! [`GpuContext`]: instance, surface, device, and swapchain state.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::frame_cpu_gpu_timing::{
    make_gpu_done_callback, FrameCpuGpuTiming, FrameCpuGpuTimingHandle,
};
use super::instance_limits::{instance_flags_for_gpu_init, required_limits_for_adapter};
use super::limits::{GpuLimits, GpuLimitsError};
use thiserror::Error;
use winit::dpi::PhysicalSize;
use winit::window::Window;

/// Sorted list of MSAA sample counts `2`, `4`, and `8` supported for **both** `color` and
/// [`wgpu::TextureFormat::Depth32Float`] on `adapter`.
///
/// Per-format support is not uniform: e.g. [`wgpu::TextureFormat::Rgba8UnormSrgb`] may allow 4× but
/// not 2× on some drivers; callers must use [`clamp_msaa_request_to_supported`] before creating textures.
fn msaa_supported_sample_counts(adapter: &wgpu::Adapter, color: wgpu::TextureFormat) -> Vec<u32> {
    let color_f = adapter.get_texture_format_features(color);
    let depth_f = adapter.get_texture_format_features(wgpu::TextureFormat::Depth32Float);
    let mut out: Vec<u32> = [2u32, 4, 8]
        .into_iter()
        .filter(|&n| {
            color_f.flags.sample_count_supported(n) && depth_f.flags.sample_count_supported(n)
        })
        .collect();
    out.sort_unstable();
    out
}

/// Sorted list of MSAA sample counts supported for **2D array** color + [`wgpu::TextureFormat::Depth32Float`]
/// on `adapter`, when the device exposes both [`wgpu::Features::MULTISAMPLE_ARRAY`] and
/// [`wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`].
///
/// Returns an empty vector when either feature is missing; callers treat this as "stereo MSAA off"
/// and silently fall back to `sample_count = 1` via [`clamp_msaa_request_to_supported`]. Upstream
/// per-format support for array multisampling currently tracks the same tiers as `MULTISAMPLE_RESOLVE`,
/// so intersecting the regular `sample_count_supported` is sufficient when the device feature is on.
fn msaa_supported_sample_counts_stereo(
    adapter: &wgpu::Adapter,
    color: wgpu::TextureFormat,
    features: wgpu::Features,
) -> Vec<u32> {
    let required = wgpu::Features::MULTISAMPLE_ARRAY
        | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
    if !features.contains(required) {
        return Vec::new();
    }
    msaa_supported_sample_counts(adapter, color)
}

/// Maps a user-requested MSAA level to a **device-valid** sample count for the current surface format.
///
/// - `requested` ≤ 1 → `1` (off).
/// - Otherwise picks the **smallest** supported count ≥ `requested` when possible (e.g. 2× requested
///   but only 4× is valid → 4×). If `requested` exceeds all tiers, uses the **largest** supported count.
fn clamp_msaa_request_to_supported(requested: u32, supported: &[u32]) -> u32 {
    if requested <= 1 {
        return 1;
    }
    if supported.is_empty() {
        return 1;
    }
    if let Some(&n) = supported.iter().find(|&&n| n >= requested) {
        return n;
    }
    supported.last().copied().unwrap_or(1)
}

/// GPU stack for presentation and future render passes.
pub struct GpuContext {
    /// Adapter metadata from construction (for diagnostics).
    adapter_info: wgpu::AdapterInfo,
    /// MSAA tiers supported for the configured surface color format and [`wgpu::TextureFormat::Depth32Float`]
    /// (sorted ascending: 2, 4, …). Empty means MSAA is unavailable.
    msaa_supported_sample_counts: Vec<u32>,
    /// MSAA tiers supported for **2D array** color + [`wgpu::TextureFormat::Depth32Float`] on the OpenXR
    /// path (sorted ascending). Empty when the adapter lacks
    /// [`wgpu::Features::MULTISAMPLE_ARRAY`] / [`wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`],
    /// which silently clamps the stereo request to `1` (MSAA off).
    msaa_supported_sample_counts_stereo: Vec<u32>,
    /// Effective swapchain MSAA sample count this frame (1 = off), set via [`Self::set_swapchain_msaa_requested`].
    swapchain_msaa_effective: u32,
    /// Requested stereo MSAA (from settings) before clamping; set each XR frame by the runtime.
    swapchain_msaa_requested_stereo: u32,
    /// Effective stereo MSAA sample count (1 = off), set via [`Self::set_swapchain_msaa_requested_stereo`].
    swapchain_msaa_effective_stereo: u32,
    /// Effective limits and derived caps for this device (shared across backend and uploads).
    limits: Arc<GpuLimits>,
    device: Arc<wgpu::Device>,
    queue: Arc<Mutex<wgpu::Queue>>,
    /// Kept as `'static` so the context can move independently of the window borrow; the window
    /// must outlive this value (owned alongside it in the app handler).
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    /// Depth target matching [`Self::config`] extent; recreated after resize.
    depth_attachment: Option<(wgpu::Texture, wgpu::TextureView)>,
    depth_extent_px: (u32, u32),
    /// Debug HUD: wall-clock CPU (tick start → last submit) and GPU (last submit → idle) timing.
    frame_timing: FrameCpuGpuTimingHandle,
}

/// GPU initialization or resize failure.
#[derive(Debug, Error)]
pub enum GpuError {
    /// No suitable adapter was found.
    #[error("request_adapter failed: {0}")]
    Adapter(String),
    /// Device creation failed.
    #[error("request_device failed: {0}")]
    Device(String),
    /// Surface could not be created from the window.
    #[error("create_surface failed: {0}")]
    Surface(String),
    /// No default surface configuration for this adapter.
    #[error("surface unsupported")]
    SurfaceUnsupported,
    /// Device reports limits below Renderide minimums.
    #[error("GPU limits: {0}")]
    Limits(#[from] GpuLimitsError),
}

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

        let surface_safe: wgpu::Surface<'static> = unsafe { std::mem::transmute(surface) };

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

        Ok(Self {
            adapter_info,
            msaa_supported_sample_counts,
            msaa_supported_sample_counts_stereo,
            swapchain_msaa_effective: 1,
            swapchain_msaa_requested_stereo: 1,
            swapchain_msaa_effective_stereo: 1,
            limits,
            device,
            queue: Arc::new(Mutex::new(queue)),
            surface: surface_safe,
            config,
            depth_attachment: None,
            depth_extent_px: (0, 0),
            frame_timing: Arc::new(Mutex::new(FrameCpuGpuTiming::default())),
        })
    }

    /// Builds GPU state using an existing wgpu instance/device from OpenXR bootstrap (mirror window).
    pub fn new_from_openxr_bootstrap(
        instance: &wgpu::Instance,
        adapter: &wgpu::Adapter,
        device: Arc<wgpu::Device>,
        queue: Arc<Mutex<wgpu::Queue>>,
        window: Arc<Window>,
        vsync: bool,
    ) -> Result<Self, GpuError> {
        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| GpuError::Surface(format!("{e:?}")))?;
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
        Ok(Self {
            adapter_info,
            msaa_supported_sample_counts,
            msaa_supported_sample_counts_stereo,
            swapchain_msaa_effective: 1,
            swapchain_msaa_requested_stereo: 1,
            swapchain_msaa_effective_stereo: 1,
            limits,
            device,
            queue,
            surface: surface_safe,
            config,
            depth_attachment: None,
            depth_extent_px: (0, 0),
            frame_timing: Arc::new(Mutex::new(FrameCpuGpuTiming::default())),
        })
    }

    /// Updates vertical sync / present mode and reconfigures the surface (hot-reload from settings).
    pub fn set_vsync(&mut self, vsync: bool) {
        let mode = if vsync {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        };
        if self.config.present_mode == mode {
            return;
        }
        self.config.present_mode = mode;
        self.surface.configure(&self.device, &self.config);
        logger::info!(
            "Present mode set to {:?} (vsync={})",
            self.config.present_mode,
            vsync
        );
    }

    /// Current swapchain configuration extent.
    pub fn size(&self) -> PhysicalSize<u32> {
        PhysicalSize::new(self.config.width, self.config.height)
    }

    /// Swapchain pixel size `(width, height)`.
    pub fn surface_extent_px(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Reconfigures the swapchain after resize or after [`wgpu::CurrentSurfaceTexture::Lost`] /
    /// [`wgpu::CurrentSurfaceTexture::Outdated`].
    pub fn reconfigure(&mut self, width: u32, height: u32) {
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.surface.configure(&self.device, &self.config);
        self.depth_attachment = None;
        self.depth_extent_px = (0, 0);
    }

    /// Borrows the configured surface for acquire/submit.
    pub fn surface(&self) -> &wgpu::Surface<'static> {
        &self.surface
    }

    /// Centralized device limits and derived caps ([`GpuLimits`]).
    pub fn limits(&self) -> &Arc<GpuLimits> {
        &self.limits
    }

    /// WGPU device for buffer/texture/pipeline creation.
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    /// Shared handle also passed to [`crate::runtime::RendererRuntime`] for uploads.
    pub fn queue(&self) -> &Arc<Mutex<wgpu::Queue>> {
        &self.queue
    }

    /// Submits render work for this frame; records last submit and GPU idle for the debug HUD timing HUD.
    pub fn submit_tracked_frame_commands(&self, cmd: wgpu::CommandBuffer) {
        self.submit_tracked_inner(cmd);
    }

    /// Same as [`Self::submit_tracked_frame_commands`] but uses an already-locked queue (e.g. debug HUD overlay encode).
    pub fn submit_tracked_frame_commands_with_queue(
        &self,
        queue: &mut wgpu::Queue,
        cmd: wgpu::CommandBuffer,
    ) {
        self.submit_tracked_inner_with_queue(queue, cmd);
    }

    fn submit_tracked_inner(&self, cmd: wgpu::CommandBuffer) {
        let track = {
            let mut ft = self.frame_timing.lock().unwrap_or_else(|e| e.into_inner());
            ft.on_before_tracked_submit()
        };
        if let Some((gen, seq)) = track {
            let submit_at = Instant::now();
            let q = self.queue.lock().unwrap_or_else(|e| e.into_inner());
            q.submit(std::iter::once(cmd));
            let handle = Arc::clone(&self.frame_timing);
            let cb = make_gpu_done_callback(handle, gen, seq, submit_at);
            q.on_submitted_work_done(cb);
        } else {
            self.queue
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .submit(std::iter::once(cmd));
        }
    }

    fn submit_tracked_inner_with_queue(&self, queue: &mut wgpu::Queue, cmd: wgpu::CommandBuffer) {
        let track = {
            let mut ft = self.frame_timing.lock().unwrap_or_else(|e| e.into_inner());
            ft.on_before_tracked_submit()
        };
        if let Some((gen, seq)) = track {
            let submit_at = Instant::now();
            queue.submit(std::iter::once(cmd));
            let handle = Arc::clone(&self.frame_timing);
            let cb = make_gpu_done_callback(handle, gen, seq, submit_at);
            queue.on_submitted_work_done(cb);
        } else {
            queue.submit(std::iter::once(cmd));
        }
    }

    /// Call at the start of each winit frame tick (same instant as [`crate::runtime::RendererRuntime::tick_frame_wall_clock_begin`]).
    pub fn begin_frame_timing(&self, frame_start: Instant) {
        self.frame_timing
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .begin_frame(frame_start);
    }

    /// Call after all tracked queue submits for this tick (before reading HUD metrics).
    ///
    /// Finalizes CPU-until-submit for this tick. GPU idle time for the HUD comes from
    /// [`super::frame_cpu_gpu_timing::FrameCpuGpuTiming::last_completed_gpu_idle_ms`], which is
    /// updated asynchronously when [`wgpu::Queue::on_submitted_work_done`] runs—no blocking poll here.
    pub fn end_frame_timing(&self) {
        let mut ft = self.frame_timing.lock().unwrap_or_else(|e| e.into_inner());
        ft.end_frame();
    }

    /// CPU time for this tick and the **latest completed** GPU submit→idle ms (may lag; see
    /// [`super::frame_cpu_gpu_timing::FrameCpuGpuTiming::last_completed_gpu_idle_ms`]).
    pub fn frame_cpu_gpu_ms_for_hud(&self) -> (Option<f64>, Option<f64>) {
        let ft = self.frame_timing.lock().unwrap_or_else(|e| e.into_inner());
        (ft.cpu_until_submit_ms, ft.last_completed_gpu_idle_ms)
    }

    /// Swapchain color format from the active surface configuration.
    pub fn config_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    /// WGPU adapter description captured at init ([`Self::new`]).
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    /// Process-local GPU memory from wgpu’s allocator when the active backend supports
    /// [`wgpu::Device::generate_allocator_report`].
    ///
    /// Returns `(allocated_bytes, reserved_bytes)`, or `(None, None)` when the backend does not report.
    /// The **Stats** debug HUD tab uses these totals every capture; the **GPU memory** tab uses a
    /// throttled full [`wgpu::AllocatorReport`] via [`crate::runtime::RendererRuntime`].
    pub fn gpu_allocator_bytes(&self) -> (Option<u64>, Option<u64>) {
        self.device
            .generate_allocator_report()
            .map(|r| (Some(r.total_allocated_bytes), Some(r.total_reserved_bytes)))
            .unwrap_or((None, None))
    }

    /// Swapchain present mode (vsync policy).
    pub fn present_mode(&self) -> wgpu::PresentMode {
        self.config.present_mode
    }

    /// Adapter-reported maximum MSAA sample count for the swapchain color format and depth.
    pub fn msaa_max_sample_count(&self) -> u32 {
        self.msaa_supported_sample_counts
            .last()
            .copied()
            .unwrap_or(1)
    }

    /// Adapter-reported maximum MSAA sample count for **2D array** color + depth (stereo / OpenXR path).
    ///
    /// Returns `1` when the device lacks [`wgpu::Features::MULTISAMPLE_ARRAY`] or
    /// [`wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`], in which case the stereo forward
    /// path silently falls back to no MSAA.
    pub fn msaa_max_sample_count_stereo(&self) -> u32 {
        self.msaa_supported_sample_counts_stereo
            .last()
            .copied()
            .unwrap_or(1)
    }

    /// Effective MSAA sample count for the main window this frame (after [`Self::set_swapchain_msaa_requested`]).
    pub fn swapchain_msaa_effective(&self) -> u32 {
        self.swapchain_msaa_effective
    }

    /// Effective stereo MSAA sample count for the OpenXR path this frame (after
    /// [`Self::set_swapchain_msaa_requested_stereo`]). `1` = off.
    pub fn swapchain_msaa_effective_stereo(&self) -> u32 {
        self.swapchain_msaa_effective_stereo
    }

    /// Sets requested MSAA for the desktop swapchain path; values are rounded to a **format-valid**
    /// tier ([`Self::msaa_supported_sample_counts`]), not merely capped by the maximum tier.
    ///
    /// Call each frame before graph execution (from [`crate::config::RenderingSettings::msaa`]).
    pub fn set_swapchain_msaa_requested(&mut self, requested: u32) {
        self.swapchain_msaa_effective =
            clamp_msaa_request_to_supported(requested, &self.msaa_supported_sample_counts);
    }

    /// Sets requested MSAA for the OpenXR stereo path; clamps to a format-valid tier against the
    /// stereo supported list. When `MULTISAMPLE_ARRAY` is unavailable the stereo list is empty and
    /// the effective count silently becomes `1`.
    ///
    /// Call each XR frame before graph execution (from [`crate::config::RenderingSettings::msaa`]).
    pub fn set_swapchain_msaa_requested_stereo(&mut self, requested: u32) {
        let requested = requested.max(1);
        let effective =
            clamp_msaa_request_to_supported(requested, &self.msaa_supported_sample_counts_stereo);
        if self.swapchain_msaa_requested_stereo != requested
            || self.swapchain_msaa_effective_stereo != effective
        {
            if requested > 1 && effective != requested {
                logger::info!(
                    "VR MSAA clamped: requested {}× → effective {}× (supported={:?})",
                    requested,
                    effective,
                    self.msaa_supported_sample_counts_stereo
                );
            }
            self.swapchain_msaa_requested_stereo = requested;
            self.swapchain_msaa_effective_stereo = effective;
        }
    }

    /// Ensures a [`wgpu::TextureFormat::Depth32Float`] attachment exists for the current surface extent.
    ///
    /// Call after [`Self::reconfigure`] or when the swapchain size may have changed.
    ///
    /// Returns an error string only if the depth attachment could not be read after allocation (defensive).
    pub fn ensure_depth_view(&mut self) -> Result<&wgpu::TextureView, &'static str> {
        self.ensure_depth_target().map(|(_, v)| v)
    }

    /// Ensures the main depth attachment exists and returns both the texture and its default view.
    ///
    /// Returns an error string only if the depth attachment could not be read after allocation (defensive).
    pub fn ensure_depth_target(
        &mut self,
    ) -> Result<(&wgpu::Texture, &wgpu::TextureView), &'static str> {
        let w = self.config.width.max(1);
        let h = self.config.height.max(1);
        let needs_recreate = self.depth_extent_px != (w, h) || self.depth_attachment.is_none();
        if needs_recreate {
            let max_dim = self.limits.wgpu.max_texture_dimension_2d;
            if w > max_dim || h > max_dim {
                logger::warn!(
                    "depth attachment extent {}×{} exceeds max_texture_dimension_2d ({max_dim}); creation may fail validation",
                    w,
                    h
                );
            }
            let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("renderide-depth"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            self.depth_extent_px = (w, h);
            self.depth_attachment = Some((tex, view));
        }
        self.depth_attachment
            .as_ref()
            .map(|(t, v)| (t, v))
            .ok_or("depth attachment missing after ensure_depth_target")
    }

    /// Acquires the next frame, reconfiguring once on [`wgpu::CurrentSurfaceTexture::Lost`] or
    /// [`wgpu::CurrentSurfaceTexture::Outdated`].
    pub fn acquire_with_recovery(
        &mut self,
        window: &Window,
    ) -> Result<wgpu::SurfaceTexture, wgpu::CurrentSurfaceTexture> {
        match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => Ok(t),
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                logger::info!("surface Lost or Outdated — reconfiguring");
                let s = window.inner_size();
                self.reconfigure(s.width, s.height);
                match self.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(t)
                    | wgpu::CurrentSurfaceTexture::Suboptimal(t) => Ok(t),
                    other => Err(other),
                }
            }
            other => Err(other),
        }
    }
}

#[cfg(test)]
mod msaa_clamp_tests {
    use super::clamp_msaa_request_to_supported;

    #[test]
    fn clamp_off_stays_off() {
        assert_eq!(clamp_msaa_request_to_supported(0, &[2, 4, 8]), 1);
        assert_eq!(clamp_msaa_request_to_supported(1, &[2, 4, 8]), 1);
    }

    #[test]
    fn clamp_upgrades_when_two_missing() {
        // Same situation as Rgba8UnormSrgb on some Vulkan drivers: only 4+ is valid.
        assert_eq!(clamp_msaa_request_to_supported(2, &[4, 8]), 4);
        assert_eq!(clamp_msaa_request_to_supported(3, &[4, 8]), 4);
    }

    #[test]
    fn clamp_exact_tier_preserved() {
        assert_eq!(clamp_msaa_request_to_supported(4, &[2, 4, 8]), 4);
    }

    #[test]
    fn clamp_falls_back_to_max_when_above_all_tiers() {
        assert_eq!(clamp_msaa_request_to_supported(16, &[4, 8]), 8);
    }

    #[test]
    fn clamp_empty_supported_means_off() {
        assert_eq!(clamp_msaa_request_to_supported(4, &[]), 1);
    }

    #[test]
    fn clamp_empty_stereo_tiers_forces_off_even_for_valid_desktop_requests() {
        // Models the case where the device lacks MULTISAMPLE_ARRAY /
        // TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES: `msaa_supported_sample_counts_stereo` returns
        // an empty `Vec`, and any MSAA request must silently collapse to 1x for the stereo path.
        for r in [2u32, 3, 4, 8, 16] {
            assert_eq!(clamp_msaa_request_to_supported(r, &[]), 1);
        }
    }
}

#[cfg(test)]
mod msaa_stereo_feature_gate_tests {
    /// Mirrors the gate used inside `msaa_supported_sample_counts_stereo`. When either feature is
    /// missing the stereo supported list must be empty regardless of per-format sample counts, so
    /// [`clamp_msaa_request_to_supported`](super::clamp_msaa_request_to_supported) can silently
    /// fall back to 1x via the empty-slice rule (see `clamp_empty_supported_means_off`).
    fn stereo_feature_gate_passes(features: wgpu::Features) -> bool {
        let required = wgpu::Features::MULTISAMPLE_ARRAY
            | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        features.contains(required)
    }

    #[test]
    fn gate_requires_both_features() {
        assert!(!stereo_feature_gate_passes(wgpu::Features::empty()));
        assert!(!stereo_feature_gate_passes(
            wgpu::Features::MULTISAMPLE_ARRAY
        ));
        assert!(!stereo_feature_gate_passes(
            wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
        ));
    }

    #[test]
    fn gate_passes_when_both_present() {
        let feats = wgpu::Features::MULTISAMPLE_ARRAY
            | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        assert!(stereo_feature_gate_passes(feats));
    }

    #[test]
    fn gate_ignores_unrelated_features() {
        let feats = wgpu::Features::MULTIVIEW | wgpu::Features::FLOAT32_FILTERABLE;
        assert!(!stereo_feature_gate_passes(feats));
    }
}
