//! [`GpuContext`]: instance, surface, device, and swapchain state.

use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Compile-time assertion that `wgpu::Queue` is `Send + Sync`; relied on by the submission path.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<wgpu::Queue>();
};

use super::frame_cpu_gpu_timing::{make_gpu_done_callback, FrameCpuGpuTimingHandle};
use super::instance_limits::required_limits_for_adapter;
use super::limits::{GpuLimits, GpuLimitsError};
use crate::config::VsyncMode;
use thiserror::Error;
use winit::dpi::PhysicalSize;
use winit::window::Window;

/// Sorted list of MSAA sample counts `2`, `4`, and `8` supported for **both** `color` and
/// the forward depth/stencil format on `adapter`.
///
/// Per-format support is not uniform: e.g. [`wgpu::TextureFormat::Rgba8UnormSrgb`] may allow 4× but
/// not 2× on some drivers; callers must use [`clamp_msaa_request_to_supported`] before creating textures.
pub(super) fn msaa_supported_sample_counts(
    adapter: &wgpu::Adapter,
    color: wgpu::TextureFormat,
    depth_stencil: wgpu::TextureFormat,
) -> Vec<u32> {
    let color_f = adapter.get_texture_format_features(color);
    let depth_f = adapter.get_texture_format_features(depth_stencil);
    let mut out: Vec<u32> = [2u32, 4, 8]
        .into_iter()
        .filter(|&n| {
            color_f.flags.sample_count_supported(n) && depth_f.flags.sample_count_supported(n)
        })
        .collect();
    out.sort_unstable();
    out
}

/// Sorted list of MSAA sample counts supported for **2D array** color + the forward depth/stencil format
/// on `adapter`, when the device exposes both [`wgpu::Features::MULTISAMPLE_ARRAY`] and
/// [`wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`].
///
/// Returns an empty vector when either feature is missing; callers treat this as "stereo MSAA off"
/// and silently fall back to `sample_count = 1` via [`clamp_msaa_request_to_supported`]. Upstream
/// per-format support for array multisampling currently tracks the same tiers as `MULTISAMPLE_RESOLVE`,
/// so intersecting the regular `sample_count_supported` is sufficient when the device feature is on.
pub(super) fn msaa_supported_sample_counts_stereo(
    adapter: &wgpu::Adapter,
    color: wgpu::TextureFormat,
    depth_stencil: wgpu::TextureFormat,
    features: wgpu::Features,
) -> Vec<u32> {
    let required = wgpu::Features::MULTISAMPLE_ARRAY
        | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
    if !features.contains(required) {
        return Vec::new();
    }
    msaa_supported_sample_counts(adapter, color, depth_stencil)
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

/// Intersects [`wgpu::Adapter::features`] with the feature bits Renderide requires for rendering.
///
/// When the `tracy` Cargo feature is active, also requests the subset of
/// `TIMESTAMP_QUERY | TIMESTAMP_QUERY_INSIDE_ENCODERS` that the adapter supports. Pass-level
/// queries only need `TIMESTAMP_QUERY`; `TIMESTAMP_QUERY_INSIDE_ENCODERS` additionally enables
/// encoder-level queries. Either feature being absent is gracefully tolerated:
/// [`crate::profiling::GpuProfilerHandle::try_new`] returns [`None`] only when
/// `TIMESTAMP_QUERY` itself is missing.
pub(super) fn adapter_render_features_intersection(adapter: &wgpu::Adapter) -> wgpu::Features {
    let compression = wgpu::Features::TEXTURE_COMPRESSION_BC
        | wgpu::Features::TEXTURE_COMPRESSION_ETC2
        | wgpu::Features::TEXTURE_COMPRESSION_ASTC;
    let optional_float32_filterable = wgpu::Features::FLOAT32_FILTERABLE;
    let adapter_format_features = wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
    let optional_depth32_stencil8 = wgpu::Features::DEPTH32FLOAT_STENCIL8;
    let multisample_array = wgpu::Features::MULTISAMPLE_ARRAY;
    let timestamp = crate::profiling::timestamp_query_features_if_supported(adapter);
    adapter.features()
        & (compression
            | optional_float32_filterable
            | adapter_format_features
            | optional_depth32_stencil8
            | multisample_array)
        | timestamp
}

pub(super) async fn request_device_for_adapter(
    adapter: &wgpu::Adapter,
    required_features: wgpu::Features,
) -> Result<(Arc<wgpu::Device>, wgpu::Queue), GpuError> {
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("renderide-skeleton"),
            required_features,
            required_limits: required_limits_for_adapter(adapter),
            ..Default::default()
        })
        .await
        .map_err(|e| GpuError::Device(format!("{e:?}")))?;
    install_uncaptured_error_handler(&device);
    Ok((Arc::new(device), queue))
}

/// Installs a non-panicking uncaptured error handler on `device` so stray wgpu validation
/// errors (for example, a [`wgpu::Device::create_view`] on a texture left invalid by a
/// device-lost event) are logged instead of terminating the process via wgpu's default
/// panicking handler. Callers that pass an externally built device (OpenXR bootstrap) must
/// invoke this explicitly so that path gets the same protection as the owned-device paths.
pub(super) fn install_uncaptured_error_handler(device: &wgpu::Device) {
    device.on_uncaptured_error(Arc::new(|err: wgpu::Error| match err {
        wgpu::Error::OutOfMemory { source } => {
            logger::error!("wgpu out-of-memory error: {source}");
        }
        wgpu::Error::Validation {
            description,
            source,
        } => {
            logger::error!("wgpu validation error: {description} ({source})");
        }
        wgpu::Error::Internal {
            description,
            source,
        } => {
            logger::error!("wgpu internal error: {description} ({source})");
        }
    }));
}

/// GPU stack for presentation and future render passes.
pub struct GpuContext {
    /// Dedicated GPU-submission thread. All main-frame `Queue::submit` and
    /// `SurfaceTexture::present` calls flow through this handle; the main tick only records
    /// command buffers and hands a [`super::driver_thread::SubmitBatch`] to the driver.
    ///
    /// Declared **first** so it drops before `queue`, `surface`, and `device`. On drop the
    /// driver pushes a shutdown sentinel, the worker drains remaining batches (dropping any
    /// unpresented [`wgpu::SurfaceTexture`] cleanly), and the thread joins — after which
    /// the queue and surface are safe to tear down.
    driver_thread: super::driver_thread::DriverThread,
    /// Adapter metadata from construction (for diagnostics).
    adapter_info: wgpu::AdapterInfo,
    /// MSAA tiers supported for the configured surface color format and forward depth/stencil format.
    /// (sorted ascending: 2, 4, …). Empty means MSAA is unavailable.
    msaa_supported_sample_counts: Vec<u32>,
    /// MSAA tiers supported for **2D array** color + forward depth/stencil format on the OpenXR
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
    queue: Arc<wgpu::Queue>,
    /// Gate that serialises operations that may access the Vulkan queue shared by wgpu and
    /// OpenXR. See [`super::GpuQueueAccessGate`] for details.
    gpu_queue_access_gate: super::GpuQueueAccessGate,
    /// Kept as `'static` so the context can move independently of the window borrow; the window
    /// must outlive this value (owned alongside it in the app handler). [`None`] in headless mode
    /// (see [`Self::new_headless`]).
    surface: Option<wgpu::Surface<'static>>,
    /// Surface configuration. In headless mode this is synthesized to describe the offscreen color
    /// format and target extent so [`Self::config_format`] / [`Self::surface_extent_px`] still
    /// return useful values.
    config: wgpu::SurfaceConfiguration,
    /// Surface-advertised present modes captured at init from
    /// [`wgpu::SurfaceCapabilities::present_modes`]. Drives the low-latency fallback chain in
    /// [`VsyncMode::resolve_present_mode`] when [`Self::set_present_mode`] reconfigures the
    /// swapchain at runtime. Empty in headless mode (no surface, no caps to query).
    supported_present_modes: Vec<wgpu::PresentMode>,
    /// Window the surface was created from, kept so swapchain Lost/Outdated recovery can call
    /// [`Window::inner_size`] without threading `&Window` through every render-path signature.
    /// [`None`] in headless mode (no winit window exists).
    window: Option<Arc<Window>>,
    /// Depth target matching [`Self::config`] extent; recreated after resize.
    depth_attachment: Option<(wgpu::Texture, wgpu::TextureView)>,
    depth_extent_px: (u32, u32),
    /// Headless primary color/depth target (lazy). Allocated on the first call to
    /// [`Self::ensure_primary_offscreen_targets`] when [`Self::is_headless`] is true so the
    /// headless `render_frame` substitution can render the main view to a persistent
    /// offscreen RT and the headless driver can copy it back to a PNG. The wrapping `Arc` lets
    /// callers obtain an owned handle that does not borrow from [`GpuContext`], avoiding the
    /// `&mut GpuContext` aliasing that would otherwise prevent passing `gpu` to the backend
    /// after substituting view targets.
    primary_offscreen: Option<PrimaryOffscreenTargets>,
    /// Debug HUD: wall-clock CPU (tick start → last submit) and GPU (last submit → idle) timing.
    frame_timing: FrameCpuGpuTimingHandle,
    /// GPU timestamp profiler for the Tracy timeline. [`None`] when the `tracy` feature is off
    /// or when the adapter lacks [`wgpu::Features::TIMESTAMP_QUERY`]. Pass-level queries work as
    /// long as that one feature is present; encoder-level queries additionally require
    /// [`wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS`].
    gpu_profiler: Option<crate::profiling::GpuProfilerHandle>,
    /// Flattened per-pass GPU timings from the most recently drained profiling frame.
    ///
    /// Written by [`Self::end_gpu_profiler_frame`] and polled by the debug HUD. Empty when no
    /// profiling frame has completed yet (GPU results lag recording by 1-2 frames).
    latest_gpu_pass_timings: Arc<Mutex<Vec<crate::profiling::GpuPassEntry>>>,
}

/// Persistent offscreen color + depth pair owned by [`GpuContext`] in headless mode.
///
/// The render graph treats these as a host render-texture (an `OffscreenRt` view) when
/// `render_frame` substitutes the main `Swapchain` view in headless mode. The headless
/// driver then `copy_texture_to_buffer` against [`PrimaryOffscreenTargets::color_texture`]
/// to read back the pixels and write a PNG.
pub struct PrimaryOffscreenTargets {
    /// Color attachment ([`wgpu::TextureFormat::Rgba8UnormSrgb`] + `RENDER_ATTACHMENT | COPY_SRC`).
    pub color_texture: wgpu::Texture,
    /// Default view of [`Self::color_texture`] for render passes.
    pub color_view: wgpu::TextureView,
    /// Depth-stencil texture matching the main forward pass format.
    pub depth_texture: wgpu::Texture,
    /// Default view of [`Self::depth_texture`] for render passes.
    pub depth_view: wgpu::TextureView,
    /// Pixel extent (width, height) shared by both attachments.
    pub extent_px: (u32, u32),
    /// Color format reused by the render graph when binding pipelines.
    pub color_format: wgpu::TextureFormat,
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
    /// Dedicated renderer-driver thread could not be spawned.
    #[error("driver thread spawn failed: {0}")]
    DriverThreadSpawn(#[source] std::io::Error),
    /// No default surface configuration for this adapter.
    #[error("surface unsupported")]
    SurfaceUnsupported,
    /// Device reports limits below Renderide minimums.
    #[error("GPU limits: {0}")]
    Limits(#[from] GpuLimitsError),
}

impl GpuContext {
    /// Updates the swapchain present mode and reconfigures the surface (hot-reload from settings).
    ///
    /// Resolves [`VsyncMode`] against the surface's actual capabilities via
    /// [`VsyncMode::resolve_present_mode`] so the result is guaranteed to be one of the variants
    /// the swapchain advertises (no risk of `surface.configure` rejecting an unsupported mode).
    /// Early-returns when the resolved mode matches the active configuration, so per-frame calls
    /// from the runtime are cheap.
    pub fn set_present_mode(&mut self, mode: VsyncMode) {
        let resolved = mode.resolve_present_mode(&self.supported_present_modes);
        if self.config.present_mode == resolved {
            return;
        }
        self.config.present_mode = resolved;
        if let Some(surface) = self.surface.as_ref() {
            surface.configure(&self.device, &self.config);
        }
        logger::info!(
            "Present mode set to {:?} (vsync={:?})",
            self.config.present_mode,
            mode
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
        profiling::scope!("gpu::reconfigure_surface");
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        if let Some(surface) = self.surface.as_ref() {
            surface.configure(&self.device, &self.config);
        }
        self.depth_attachment = None;
        self.depth_extent_px = (0, 0);
    }

    /// Borrows the configured surface for acquire/submit; [`None`] in headless mode.
    pub fn surface(&self) -> Option<&wgpu::Surface<'static>> {
        self.surface.as_ref()
    }

    /// Whether this context drives a real swapchain surface (vs. headless offscreen primary target).
    pub fn is_headless(&self) -> bool {
        self.surface.is_none()
    }

    /// Live `inner_size` of the window stored inside this context, if windowed.
    ///
    /// Re-queries the window each call so callers handling `WindowEvent::ScaleFactorChanged` can
    /// pick up the new logical size without holding a separate `Arc<Window>`. Returns [`None`] in
    /// headless mode.
    pub fn window_inner_size(&self) -> Option<(u32, u32)> {
        self.window.as_ref().map(|w| {
            let s = w.inner_size();
            (s.width, s.height)
        })
    }

    /// Returns the lazy-allocated primary offscreen color/depth pair owned by this context.
    ///
    /// Returns [`None`] when the context is windowed (it has a real swapchain instead). On the
    /// first call in headless mode, allocates the persistent textures matching `config.width ×
    /// config.height` and the configured color format. Subsequent calls return the same handles
    /// until the context is dropped.
    ///
    /// `render_frame` calls this when `window.is_none()` to substitute the main `Swapchain`
    /// view with a `FrameViewTarget::OffscreenRt` backed by these textures.
    pub fn primary_offscreen_targets(&mut self) -> Option<&PrimaryOffscreenTargets> {
        if !self.is_headless() {
            return None;
        }
        if self.primary_offscreen.is_none() {
            let max_dim = self.limits.max_texture_dimension_2d();
            let req_w = self.config.width.max(1);
            let req_h = self.config.height.max(1);
            let width = req_w.min(max_dim);
            let height = req_h.min(max_dim);
            if (width, height) != (req_w, req_h) {
                logger::warn!(
                    "headless primary offscreen: {req_w}x{req_h} exceeds max_texture_dimension_2d={max_dim}; clamped to {width}x{height}",
                );
            }
            let color_format = self.config.format;
            let color_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("renderide-headless-primary-color"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: color_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());
            let depth_format =
                crate::render_graph::main_forward_depth_stencil_format(self.device.features());
            let depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("renderide-headless-primary-depth"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: depth_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.primary_offscreen = Some(PrimaryOffscreenTargets {
                color_texture,
                color_view,
                depth_texture,
                depth_view,
                extent_px: (width, height),
                color_format,
            });
        }
        self.primary_offscreen.as_ref()
    }

    /// Returns the persistent headless color texture for PNG readback.
    ///
    /// Returns [`None`] in windowed mode and also when the headless offscreen has not yet been
    /// allocated (call [`Self::primary_offscreen_targets`] first or run a render tick).
    /// Unlike [`Self::primary_offscreen_targets`], this getter takes `&self` so it does not
    /// conflict with concurrent mutable borrows on `gpu` during readback.
    pub fn headless_color_texture(&self) -> Option<&wgpu::Texture> {
        self.primary_offscreen.as_ref().map(|t| &t.color_texture)
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
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }

    /// Gate acquired around short operations that may access the Vulkan queue shared by wgpu and
    /// OpenXR. The driver thread, texture upload path, and OpenXR frame submission all use this
    /// handle.
    pub fn gpu_queue_access_gate(&self) -> &super::GpuQueueAccessGate {
        &self.gpu_queue_access_gate
    }

    /// Submits a single command buffer for this frame through the driver thread, tracked for
    /// the debug HUD frame timing HUD. No surface is presented on this path; the older callers
    /// (VR mirror eye-to-staging blit) will migrate to [`Self::submit_frame_batch`] in a
    /// follow-up.
    pub fn submit_tracked_frame_commands(&self, cmd: wgpu::CommandBuffer) {
        self.submit_frame_batch_inner(vec![cmd], None, None, Vec::new());
    }

    /// Same as [`Self::submit_tracked_frame_commands`] but accepts an externally-held
    /// [`wgpu::Queue`] reference. Retained for API compatibility with the pre-driver-thread
    /// call sites — the reference is ignored because submit now always runs on the driver
    /// thread with its own cloned [`Arc<wgpu::Queue>`].
    pub fn submit_tracked_frame_commands_with_queue(
        &self,
        _queue: &wgpu::Queue,
        cmd: wgpu::CommandBuffer,
    ) {
        self.submit_frame_batch_inner(vec![cmd], None, None, Vec::new());
    }

    /// Submits multiple command buffers through the driver thread in a single
    /// [`wgpu::Queue::submit`] call, tracked for frame timing. No surface is presented on
    /// this path — for swapchain frames use [`Self::submit_frame_batch`] with a
    /// [`wgpu::SurfaceTexture`].
    ///
    /// All `Queue::write_buffer` calls on the main thread must have occurred before this
    /// call so they are visible to GPU commands in the same submit (wgpu guarantees this
    /// ordering regardless of which thread performs the submit).
    pub fn submit_tracked_frame_commands_batch(
        &self,
        _queue: &wgpu::Queue,
        cmds: impl IntoIterator<Item = wgpu::CommandBuffer>,
    ) {
        self.submit_frame_batch_inner(cmds.into_iter().collect(), None, None, Vec::new());
    }

    /// Hands a finished frame off to the driver thread for submit + present.
    ///
    /// The surface texture is optional: pass `Some` for the main swapchain frame (the
    /// driver calls [`wgpu::SurfaceTexture::present`] after submit), `None` for frames
    /// that render to an offscreen target only. `wait` is an opaque oneshot used by
    /// synchronous callers (headless tests) that need to block until the driver has
    /// finished with this batch.
    pub fn submit_frame_batch(
        &self,
        cmds: Vec<wgpu::CommandBuffer>,
        surface_texture: Option<wgpu::SurfaceTexture>,
        wait: Option<super::driver_thread::SubmitWait>,
    ) {
        self.submit_frame_batch_inner(cmds, surface_texture, wait, Vec::new());
    }

    /// Same as [`Self::submit_frame_batch`] but attaches extra `on_submitted_work_done`
    /// callbacks that fire after the driver has submitted this batch to the queue.
    ///
    /// Use this to schedule main-thread work (e.g. `map_async` for Hi-Z readback) that
    /// depends on the submit having completed without paying a driver-ring flush.
    pub fn submit_frame_batch_with_callbacks(
        &self,
        cmds: Vec<wgpu::CommandBuffer>,
        surface_texture: Option<wgpu::SurfaceTexture>,
        wait: Option<super::driver_thread::SubmitWait>,
        extra_on_submitted_work_done: Vec<Box<dyn FnOnce() + Send + 'static>>,
    ) {
        self.submit_frame_batch_inner(cmds, surface_texture, wait, extra_on_submitted_work_done);
    }

    /// Internal helper that builds the [`super::driver_thread::SubmitBatch`] (including the
    /// frame-timing callback) and pushes it into the driver thread's ring. Blocks when the
    /// ring is full — that block is the frame-pacing backpressure.
    fn submit_frame_batch_inner(
        &self,
        command_buffers: Vec<wgpu::CommandBuffer>,
        surface_texture: Option<wgpu::SurfaceTexture>,
        wait: Option<super::driver_thread::SubmitWait>,
        mut extra_on_submitted_work_done: Vec<Box<dyn FnOnce() + Send + 'static>>,
    ) {
        let track = {
            let mut ft = self.frame_timing.lock().unwrap_or_else(|e| e.into_inner());
            ft.on_before_tracked_submit()
        };
        let mut on_submitted_work_done: Vec<Box<dyn FnOnce() + Send + 'static>> = Vec::new();
        if let Some((gen, seq)) = track {
            let submit_at = Instant::now();
            let handle = Arc::clone(&self.frame_timing);
            on_submitted_work_done.push(Box::new(make_gpu_done_callback(
                handle, gen, seq, submit_at,
            )));
        }
        on_submitted_work_done.append(&mut extra_on_submitted_work_done);
        let frame_seq = track.map(|(_, seq)| seq as u64).unwrap_or(0);
        let batch = super::driver_thread::SubmitBatch {
            command_buffers,
            surface_texture,
            on_submitted_work_done,
            wait,
            frame_seq,
        };
        self.driver_thread.submit(batch);
    }

    /// Drains any driver-thread error captured since the last check, leaving the slot empty.
    ///
    /// Call once per tick from the frame epilogue; route the returned error through the
    /// existing device-recovery path (same as a swapchain `SurfaceError::Lost`).
    pub fn take_driver_error(&self) -> Option<super::driver_thread::DriverError> {
        self.driver_thread.take_pending_error()
    }

    /// Blocks until the driver thread has processed every previously-submitted batch.
    ///
    /// Used by the headless readback path to establish ordering between the rendered
    /// frame's submit (which runs on the driver thread) and the readback copy (which
    /// runs on the main thread). Most code paths never need this.
    pub fn flush_driver(&self) {
        self.driver_thread.flush();
    }

    /// Blocks only until the most recently submitted surface-carrying batch has reached
    /// [`wgpu::SurfaceTexture::present`] on the driver thread.
    ///
    /// Call this right before [`wgpu::Surface::get_current_texture`] to honour wgpu's
    /// "only one outstanding surface texture" rule without flushing the whole ring.
    /// Unlike [`Self::flush_driver`] this permits non-surface work (submits without a
    /// swapchain texture, [`wgpu::Queue::on_submitted_work_done`] callbacks) to remain
    /// pipelined alongside the next frame's CPU recording.
    pub fn wait_for_previous_present(&self) {
        self.driver_thread.wait_for_previous_present();
    }

    /// Call at the start of each winit frame tick (same instant as [`crate::runtime::RendererRuntime::tick_frame_wall_clock_begin`]).
    pub fn begin_frame_timing(&self, frame_start: Instant) {
        profiling::scope!("gpu::begin_frame_timing");
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
        profiling::scope!("gpu::end_frame_timing");
        let mut ft = self.frame_timing.lock().unwrap_or_else(|e| e.into_inner());
        ft.end_frame();
    }

    /// Mutable reference to the GPU profiler, when one is active.
    ///
    /// Returns [`None`] when the `tracy` feature is off, or when the adapter lacks the required
    /// timestamp-query features (see [`crate::profiling::GpuProfilerHandle::try_new`]).
    pub fn gpu_profiler_mut(&mut self) -> Option<&mut crate::profiling::GpuProfilerHandle> {
        self.gpu_profiler.as_mut()
    }

    /// Temporarily removes the GPU profiler handle from [`GpuContext`] and returns it.
    ///
    /// Use this when code must hold a borrowed reference into `GpuContext` (e.g. a
    /// `ResolvedView` that borrows `depth_texture`) while also needing to drive the profiler
    /// inside a nested loop. Pair every call with [`Self::restore_gpu_profiler`].
    ///
    /// Returns [`None`] when no profiler is active (feature off or adapter unsupported).
    pub fn take_gpu_profiler(&mut self) -> Option<crate::profiling::GpuProfilerHandle> {
        self.gpu_profiler.take()
    }

    /// Restores a profiler handle previously removed by [`Self::take_gpu_profiler`].
    ///
    /// If `profiler` is [`None`], this is a no-op.
    pub fn restore_gpu_profiler(&mut self, profiler: Option<crate::profiling::GpuProfilerHandle>) {
        if self.gpu_profiler.is_none() {
            self.gpu_profiler = profiler;
        }
    }

    /// Ends the GPU profiling frame and drains completed query results into Tracy.
    ///
    /// Call once per render tick after all command encoders for the tick have been submitted
    /// (e.g. from [`crate::app::renderide_app::RenderideApp::frame_tick_epilogue`]).
    /// Does nothing when no GPU profiler is active.
    pub fn end_gpu_profiler_frame(&mut self) {
        profiling::scope!("gpu::drain_gpu_profiler");
        if self.gpu_profiler.is_none() {
            return;
        }
        // `wgpu_profiler::end_frame` calls `map_async` on the same Query Read Buffer that
        // `resolve_queries` just wrote a copy into. The render graph hands those resolve
        // command buffers to the driver thread for an asynchronous `Queue::submit`, so if
        // the driver has not yet drained the ring by the time we reach this point,
        // `map_async` would put the buffer in pending-mapped state before the submit runs
        // and wgpu validation would reject it with "buffer is still mapped". Flushing the
        // driver guarantees every prior submit has completed before we transition the
        // buffer.
        self.driver_thread.flush();
        if let Some(p) = self.gpu_profiler.as_mut() {
            p.end_frame();
            let ts_period = self.queue.get_timestamp_period();
            if let Some(timings) = p.process_finished_frame(ts_period) {
                if let Ok(mut slot) = self.latest_gpu_pass_timings.lock() {
                    *slot = timings;
                }
            }
        }
    }

    /// Returns a shared handle to the latest flattened per-pass GPU timings.
    ///
    /// The debug HUD polls this once per frame. The underlying vector is replaced atomically by
    /// [`Self::end_gpu_profiler_frame`] on the main thread; readers clone the current contents
    /// under a short lock and render them without blocking the renderer.
    pub fn latest_gpu_pass_timings_handle(
        &self,
    ) -> Arc<Mutex<Vec<crate::profiling::GpuPassEntry>>> {
        Arc::clone(&self.latest_gpu_pass_timings)
    }

    /// CPU time for this tick and the **latest completed** GPU submit→idle ms (may lag; see
    /// [`super::frame_cpu_gpu_timing::FrameCpuGpuTiming::last_completed_gpu_idle_ms`]).
    pub fn frame_cpu_gpu_ms_for_hud(&self) -> (Option<f64>, Option<f64>) {
        let ft = self.frame_timing.lock().unwrap_or_else(|e| e.into_inner());
        (ft.cpu_until_submit_ms, ft.last_completed_gpu_idle_ms)
    }

    /// Most recently completed GPU submit→idle interval in **seconds**, for the IPC
    /// [`crate::shared::PerformanceState::render_time`] field consumed by
    /// `FrooxEngine.PerformanceMetrics.RenderTime`.
    ///
    /// Returns [`None`] until the first [`wgpu::Queue::on_submitted_work_done`] callback has run;
    /// callers that need the host-visible "unavailable" sentinel should map [`None`] to `-1.0`,
    /// matching the Renderite.Unity `XRStats.TryGetGPUTimeLastFrame` contract.
    pub fn last_completed_gpu_render_time_seconds(&self) -> Option<f32> {
        let ft = self.frame_timing.lock().unwrap_or_else(|e| e.into_inner());
        ft.last_completed_gpu_idle_ms.map(|ms| (ms / 1000.0) as f32)
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

    /// Ensures a stencil-capable depth attachment exists for the current surface extent.
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
        let depth_stencil_format =
            crate::render_graph::main_forward_depth_stencil_format(self.device.features());
        let needs_recreate = self.depth_extent_px != (w, h)
            || self
                .depth_attachment
                .as_ref()
                .is_none_or(|(tex, _)| tex.format() != depth_stencil_format);
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
                format: depth_stencil_format,
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
    ///
    /// Returns [`wgpu::CurrentSurfaceTexture::Lost`] when this context is headless (no surface).
    /// Uses the stored [`Self::window`] for size queries on recovery so render-path callers do
    /// not have to thread `&Window` through their signatures.
    pub fn acquire_with_recovery(
        &mut self,
    ) -> Result<wgpu::SurfaceTexture, wgpu::CurrentSurfaceTexture> {
        let Some(surface) = self.surface.as_ref() else {
            return Err(wgpu::CurrentSurfaceTexture::Lost);
        };
        match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => Ok(t),
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                logger::info!("surface Lost or Outdated — reconfiguring");
                let size = self.window.as_ref().map(|w| w.inner_size());
                if let Some(s) = size {
                    self.reconfigure(s.width, s.height);
                }
                let Some(surface) = self.surface.as_ref() else {
                    return Err(wgpu::CurrentSurfaceTexture::Lost);
                };
                match surface.get_current_texture() {
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

mod init;
