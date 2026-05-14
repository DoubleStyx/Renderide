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
//! - [`msaa`] -- [`GpuMsaa`] sub-handle for supported / effective MSAA tier state.
//! - [`mapped_buffer_recovery`] -- mapped staging/readback buffer recovery.

use std::sync::Arc;

/// Compile-time assertion that `wgpu::Queue` is `Send + Sync`; relied on by the submission path.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<wgpu::Queue>();
};

use super::limits::{GpuLimits, GpuLimitsError};
use super::submission_state::GpuSubmissionState;
use super::sync::device_health::GpuDeviceHealth;
use super::sync::mapped_buffer_health::GpuMappedBufferHealth;
use mapped_buffer_recovery::MappedBufferRecoveryFrame;
use thiserror::Error;
use winit::window::Window;

mod depth_attachment;
mod headless_targets;
mod init;
mod mapped_buffer_recovery;
mod msaa;
mod profiler;
mod submission;
mod surface;

pub use headless_targets::PrimaryOffscreenTargets;
pub use msaa::GpuMsaa;

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
    /// Supported / effective MSAA tier state ([`GpuMsaa`]).
    msaa: GpuMsaa,
    /// Effective limits and derived caps for this device (shared across backend and uploads).
    limits: Arc<GpuLimits>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    /// Gate that serialises operations that may access the Vulkan queue shared by wgpu and
    /// OpenXR. See [`super::GpuQueueAccessGate`] for details.
    gpu_queue_access_gate: super::GpuQueueAccessGate,
    /// Per-frame recovery policy for CPU-mapped GPU staging/readback buffers.
    mapped_buffer_recovery: mapped_buffer_recovery::GpuMappedBufferRecovery,
    /// Shared fatal device-loss state set by wgpu callbacks.
    device_health: Arc<GpuDeviceHealth>,
    /// Last device-loss generation observed by the app frame loop.
    seen_device_lost_generation: u64,
    /// Kept as `'static` so the context can move independently of the window borrow; the window
    /// must outlive this value (owned alongside it in the app handler). [`None`] in headless mode
    /// (see [`Self::new_headless`]).
    surface: Option<wgpu::Surface<'static>>,
    /// Whether the active window surface has successfully completed [`wgpu::Surface::configure`].
    ///
    /// wgpu leaves a surface unconfigured when configure reports a validation error. Presentation
    /// paths must check this before calling [`wgpu::Surface::get_current_texture`], because an
    /// unconfigured surface can otherwise be reported as a fatal error by backends that have not
    /// installed a surface error sink yet.
    surface_configured: bool,
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
    /// [`Window::surface_size`] without threading `&Window` through every render-path signature.
    /// [`None`] in headless mode (no winit window exists).
    window: Option<Arc<dyn Window>>,
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
    /// Surface could not be configured for presentation.
    #[error("surface configure failed: {0}")]
    SurfaceConfigure(String),
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

    /// MSAA tier state ([`GpuMsaa`]).
    pub fn msaa(&self) -> &GpuMsaa {
        &self.msaa
    }

    /// Mutable MSAA tier state for setting requested sample counts each frame.
    pub fn msaa_mut(&mut self) -> &mut GpuMsaa {
        &mut self.msaa
    }

    /// Records that mapped staging/readback buffers should be discarded before reuse.
    pub(crate) fn mark_mapped_buffers_invalid(&self, reason: impl AsRef<str>) {
        self.mapped_buffer_recovery
            .mark_mapped_buffers_invalid(reason);
    }

    /// Shared mapped-buffer invalidation generation used by async GPU owners.
    pub(crate) fn mapped_buffer_health(&self) -> Arc<GpuMappedBufferHealth> {
        self.mapped_buffer_recovery.health()
    }

    /// Begins mapped-buffer recovery bookkeeping for a render frame.
    pub(crate) fn begin_mapped_buffer_recovery_frame(&mut self) -> MappedBufferRecoveryFrame {
        self.mapped_buffer_recovery
            .begin_mapped_buffer_recovery_frame()
    }

    /// Observes invalidations reported by wgpu while the current frame is already running.
    pub(crate) fn observe_mapped_buffer_invalidation_during_frame(&mut self) -> bool {
        self.mapped_buffer_recovery
            .observe_mapped_buffer_invalidation_during_frame()
    }

    /// Whether this frame should avoid CPU-mapped staging/readback buffers.
    pub(crate) fn avoid_mapped_buffers_this_frame(&self) -> bool {
        self.mapped_buffer_recovery
            .avoid_mapped_buffers_this_frame()
    }

    /// Current mapped-buffer invalidation generation.
    pub(crate) fn mapped_buffer_invalidation_generation(&self) -> u64 {
        self.mapped_buffer_recovery
            .mapped_buffer_invalidation_generation()
    }

    /// Whether the active `wgpu::Device` has been reported lost.
    pub(crate) fn device_lost(&self) -> bool {
        self.device_health.is_lost()
    }

    /// Returns a newly observed device-loss generation once.
    pub(crate) fn take_device_lost(&mut self) -> Option<u64> {
        let generation = self.device_health.lost_generation();
        if generation == 0 || generation == self.seen_device_lost_generation {
            return None;
        }
        self.seen_device_lost_generation = generation;
        Some(generation)
    }
}
