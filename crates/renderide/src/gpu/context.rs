//! [`GpuContext`]: instance, surface, device, and swapchain state.
//!
//! The struct lives here together with [`GpuError`] and the small core accessors
//! (`limits`, `device`, `queue`, `gpu_queue_access_gate`, `adapter_info`). All other
//! inherent methods are split across thematic submodules:
//!
//! - [`init`] -- three constructors (windowed, headless, OpenXR-bootstrap) + assemble helpers.
//! - [`surface`] -- present mode / max latency / resize / acquire-with-recovery.
//! - [`depth_attachment`] -- main forward depth target ensure/recreate.
//! - [`headless_targets`] -- [`headless_targets::PrimaryOffscreenTargets`] state and accessors.
//! - [`submission`] -- driver-thread submit / present facade.
//! - [`profiler`] -- frame-timing + GPU profiler facade and HUD-facing readouts.
//! - [`msaa_state`] -- supported / effective MSAA tier state for desktop and stereo paths.

use std::sync::Arc;

/// Compile-time assertion that `wgpu::Queue` is `Send + Sync`; relied on by the submission path.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<wgpu::Queue>();
};

use super::limits::{GpuLimits, GpuLimitsError};
use super::submission_state::GpuSubmissionState;
use thiserror::Error;
use winit::window::Window;

mod depth_attachment;
mod headless_targets;
mod init;
mod msaa_state;
mod profiler;
mod submission;
mod surface;

pub use headless_targets::PrimaryOffscreenTargets;

/// GPU stack for presentation and future render passes.
pub struct GpuContext {
    /// Submission, frame timing, and GPU profiling state. All main-frame `Queue::submit` and
    /// `SurfaceTexture::present` calls flow through this bundle; the main tick only records
    /// command buffers and hands a [`super::driver_thread::SubmitBatch`] to the driver.
    ///
    /// Declared **first** so it drops before `queue`, `surface`, and `device`. On drop the
    /// driver pushes a shutdown sentinel, the worker drains remaining batches (dropping any
    /// unpresented [`wgpu::SurfaceTexture`] cleanly), and the thread joins -- after which
    /// the queue and surface are safe to tear down.
    submission: GpuSubmissionState,
    /// Adapter metadata from construction (for diagnostics).
    adapter_info: wgpu::AdapterInfo,
    /// MSAA tiers supported for the configured surface color format and forward depth/stencil format.
    /// (sorted ascending: 2, 4, ...). Empty means MSAA is unavailable.
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
    /// [`crate::config::VsyncMode::resolve_present_mode`] when [`Self::set_present_mode`]
    /// reconfigures the swapchain at runtime. Empty in headless mode (no surface, no caps to query).
    supported_present_modes: Vec<wgpu::PresentMode>,
    /// Window the surface was created from, kept so swapchain Lost/Outdated recovery can call
    /// [`Window::inner_size`] without threading `&Window` through every render-path signature.
    /// [`None`] in headless mode (no winit window exists).
    window: Option<Arc<Window>>,
    /// Depth target matching [`Self::config`] extent; recreated after resize.
    depth_attachment: Option<(wgpu::Texture, wgpu::TextureView)>,
    depth_extent_px: (u32, u32),
    /// Headless primary color/depth target (lazy). Allocated on the first call to
    /// [`Self::primary_offscreen_targets`] when [`Self::is_headless`] is true so the
    /// headless `render_frame` substitution can render the main view to a persistent
    /// offscreen RT and the headless driver can copy it back to a PNG. The wrapping `Arc` lets
    /// callers obtain an owned handle that does not borrow from [`GpuContext`], avoiding the
    /// `&mut GpuContext` aliasing that would otherwise prevent passing `gpu` to the backend
    /// after substituting view targets.
    primary_offscreen: Option<PrimaryOffscreenTargets>,
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

    /// WGPU adapter description captured at init ([`Self::new`]).
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }
}
