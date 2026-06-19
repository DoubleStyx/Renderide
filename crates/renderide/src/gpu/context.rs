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
use crate::gpu::flight_recorder::{GpuFlightEventKind, GpuFlightRecorder};
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
pub(crate) use init::WindowDisplayHandle;
pub use msaa::GpuMsaa;
pub use submission::FrameSubmitKind;

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
    /// Recent GPU/XR lifecycle events retained in memory for crash-path dumps.
    flight_recorder: Arc<GpuFlightRecorder>,
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
    /// [`wgpu::SurfaceCapabilities::present_modes`]. Drives explicit-mode fallback chains in
    /// [`crate::config::PresentationModeSetting::resolve_present_mode`] when [`Self::set_present_mode`]
    /// reconfigures the swapchain at runtime. Empty in headless mode (no surface, no caps to query).
    supported_present_modes: Vec<wgpu::PresentMode>,
    /// Window the surface was created from, kept so swapchain Lost/Outdated recovery can call
    /// [`Window::surface_size`] without threading `&Window` through every render-path signature.
    /// [`None`] in headless mode (no winit window exists).
    window: Option<Arc<dyn Window>>,
    /// Depth target matching [`Self::config`] extent; recreated after resize.
    depth_attachment: Option<(wgpu::Texture, wgpu::TextureView)>,
    depth_extent_px: (u32, u32),
    /// Primary final color/depth target (lazy). Allocated on the first call to
    /// [`Self::primary_offscreen_targets`] so the main view can render to a persistent
    /// offscreen RT before desktop presentation blits the final color to the swapchain. The
    /// texture handles are cheap clones, which lets callers build view plans without keeping
    /// a borrow on [`GpuContext`] through render-graph execution.
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
    #[inline]
    pub fn limits(&self) -> &Arc<GpuLimits> {
        &self.limits
    }

    /// WGPU device for buffer/texture/pipeline creation.
    #[inline]
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    /// Shared handle also passed to [`crate::runtime::RendererRuntime`] for uploads.
    #[inline]
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }

    /// Gate acquired around short operations that may access the Vulkan queue shared by wgpu and
    /// OpenXR. The driver thread, texture upload path, and OpenXR frame submission all use this
    /// handle.
    #[inline]
    pub fn gpu_queue_access_gate(&self) -> &super::GpuQueueAccessGate {
        &self.gpu_queue_access_gate
    }

    /// WGPU adapter description captured at init ([`Self::new`]).
    #[inline]
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    /// MSAA tier state ([`GpuMsaa`]).
    #[inline]
    pub fn msaa(&self) -> &GpuMsaa {
        &self.msaa
    }

    /// Mutable MSAA tier state for setting requested sample counts each frame.
    #[inline]
    pub fn msaa_mut(&mut self) -> &mut GpuMsaa {
        &mut self.msaa
    }

    /// Records that mapped staging/readback buffers should be discarded before reuse.
    #[inline]
    pub(crate) fn mark_mapped_buffers_invalid(&self, reason: impl AsRef<str>) {
        self.mapped_buffer_recovery
            .mark_mapped_buffers_invalid(reason);
    }

    /// Shared mapped-buffer invalidation generation used by async GPU owners.
    #[inline]
    pub(crate) fn mapped_buffer_health(&self) -> Arc<GpuMappedBufferHealth> {
        self.mapped_buffer_recovery.health()
    }

    /// Begins mapped-buffer recovery bookkeeping for a render frame.
    #[inline]
    pub(crate) fn begin_mapped_buffer_recovery_frame(&mut self) -> MappedBufferRecoveryFrame {
        self.mapped_buffer_recovery
            .begin_mapped_buffer_recovery_frame()
    }

    /// Observes invalidations reported by wgpu while the current frame is already running.
    #[inline]
    pub(crate) fn observe_mapped_buffer_invalidation_during_frame(&mut self) -> bool {
        self.mapped_buffer_recovery
            .observe_mapped_buffer_invalidation_during_frame()
    }

    /// Whether this frame should avoid CPU-mapped staging/readback buffers.
    #[inline]
    pub(crate) fn avoid_mapped_buffers_this_frame(&self) -> bool {
        self.mapped_buffer_recovery
            .avoid_mapped_buffers_this_frame()
    }

    /// Current mapped-buffer invalidation generation.
    #[inline]
    pub(crate) fn mapped_buffer_invalidation_generation(&self) -> u64 {
        self.mapped_buffer_recovery
            .mapped_buffer_invalidation_generation()
    }

    /// Whether the active `wgpu::Device` has been reported lost.
    #[inline]
    pub(crate) fn device_lost(&self) -> bool {
        self.device_health.is_lost()
    }

    /// Abandons the configured window surface after device loss so wgpu does not unconfigure it.
    ///
    /// wgpu's Vulkan backend validates that every swapchain semaphore has been released when
    /// `Surface::drop` unconfigures the swapchain. If the device is already lost,
    /// `SurfaceTexture::present` can fail before wgpu clears the acquired texture record, leaving
    /// the surface impossible to tear down cleanly inside wgpu. The process is exiting on this
    /// path, so leaking the surface is preferable to a secondary panic that hides the device loss.
    pub(crate) fn abandon_surface_after_device_loss(&mut self) {
        if !self.device_lost() {
            return;
        }
        let Some(surface) = self.surface.take() else {
            self.surface_configured = false;
            return;
        };
        self.surface_configured = false;
        logger::warn!(
            "GPU device lost; abandoning configured wgpu surface during shutdown to avoid Vulkan swapchain unconfigure with a possibly outstanding SurfaceTexture"
        );
        let _surface = std::mem::ManuallyDrop::new(surface);
    }

    /// Records one GPU/XR diagnostic event in the in-memory flight recorder.
    pub(crate) fn record_gpu_flight_event(&self, kind: GpuFlightEventKind) {
        self.flight_recorder.record(kind);
    }

    /// Shared in-memory GPU/XR flight recorder.
    pub(crate) fn gpu_flight_recorder(&self) -> &Arc<GpuFlightRecorder> {
        &self.flight_recorder
    }

    /// Dumps recent GPU/XR events once on a crash path.
    pub(crate) fn dump_gpu_flight_recorder_once(&self, reason: &'static str) -> bool {
        self.flight_recorder.dump_once(reason)
    }

    /// Records that the app has observed a device-loss generation.
    pub(crate) fn record_device_loss_observed(&self, generation: u64) {
        self.record_gpu_flight_event(GpuFlightEventKind::DeviceLossObserved {
            generation,
            adapter_name: self.adapter_info.name.clone(),
            backend: self.adapter_info.backend,
        });
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

impl Drop for GpuContext {
    fn drop(&mut self) {
        self.abandon_surface_after_device_loss();
    }
}
