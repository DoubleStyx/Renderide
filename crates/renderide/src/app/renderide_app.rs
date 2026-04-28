//! Winit [`ApplicationHandler`] state: [`RendererRuntime`], lazily created window and [`GpuContext`],
//! optional [`crate::xr::XrSessionBundle`] (OpenXR GPU path), and the per-frame tick ([`RenderideApp::tick_frame`]). See [`crate::app`] for the
//! high-level flow.
//!
//! ## Frame tick phases
//!
//! [`tick_frame`] runs these **private** stages in order (AAA-style “frame phases” / “tick stages”):
//!
//! 1. [`frame_tick_prologue`] — log level, wall-clock tick markers, GPU frame timing begin, swapchain vsync from settings.
//! 2. [`poll_ipc_and_window`] — drain IPC; apply host output (cursor); per-frame cursor lock when requested.
//! 3. [`RendererRuntime::run_asset_integration`] — one time-sliced mesh/texture upload drain per tick (after IPC, before OpenXR).
//! 4. [`xr_begin_tick`] — OpenXR `wait_frame` / view poses **before** lock-step (must stay before
//!    [`lock_step_exchange`] so [`InputState::vr`] matches the same [`OpenxrFrameTick`] snapshot).
//! 5. [`lock_step_exchange`] — when allowed, [`RendererRuntime::pre_frame`] with winit input + optional VR IPC.
//! 6. Early exits — shutdown, fatal IPC, missing window/GPU (each runs epilogue timing).
//! 7. [`render_views`] — HMD multiview submit if XR+GPU; secondary cameras to render textures;
//!    debug HUD input/time for this frame (must run before [`RendererRuntime::render_frame`] appends the main camera).
//! 8. [`present_and_diagnostics`] — VR mirror blit or clear (with optional Dear ImGui overlay on the desktop surface); OpenXR `end_frame_empty` when needed (desktop world render is in step 7).
//! 9. [`frame_tick_epilogue`] — GPU frame timing end, debug HUD capture, and forwarding the
//!    most recently completed GPU submit→idle interval to the frontend for the next
//!    [`crate::shared::PerformanceState::render_time`].
//!
//! [`tick_phase_trace`] emits `trace!` lines prefixed with [`TICK_TRACE_PREFIX`] for grep/profiling; the same
//! splits are natural boundaries for the `tracing` crate’s spans if added later.

mod frame_driver;

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use logger::LogLevel;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, DeviceEvents};
use winit::window::{Window, WindowId};

use crate::config::VsyncMode;
use crate::frontend::input::{
    apply_device_event, apply_output_state_to_window, apply_per_frame_cursor_lock_when_locked,
    apply_window_event, vr_inputs_for_session, CursorOutputTracking, WindowInputAccumulator,
};
use crate::gpu::GpuContext;
use crate::output_device::head_output_device_wants_openxr;
use crate::present::{
    present_clear_frame, present_clear_frame_overlay_traced, SurfaceAcquireTrace,
    SurfaceSubmitTrace,
};
use crate::render_graph::GraphExecuteError;
use crate::runtime::RendererRuntime;
use crate::shared::{HeadOutputDevice, VRControllerState};
use crate::xr::{synthesize_hand_states, OpenxrFrameTick, XrSessionBundle};
use glam::{Quat, Vec3};

use super::frame_loop;
use super::frame_pacing;
use super::startup::{
    apply_window_title_from_init, effective_output_device_for_gpu, effective_renderer_log_level,
    ExternalShutdownCoordinator, LOG_FLUSH_INTERVAL,
};
use super::window_icon::try_embedded_window_icon;

/// Prefix for per-phase trace lines in [`RenderideApp::tick_frame`] (grep-friendly; no log `target` in this logger).
const TICK_TRACE_PREFIX: &str = "renderide::tick";

/// Emits a trace line naming the current frame phase (see module docs).
fn tick_phase_trace(phase: &'static str) {
    logger::trace!("{} phase={phase}", TICK_TRACE_PREFIX);
}

pub(crate) struct RenderideApp {
    runtime: RendererRuntime,
    /// Initial vsync preference used for [`GpuContext::new`] before live updates from settings.
    initial_vsync: VsyncMode,
    /// GPU validation layers flag for the initial [`GpuContext::new`] (persisted; restart to apply).
    initial_gpu_validation: bool,
    /// GPU power preference resolved from [`crate::config::DebugSettings::power_preference`].
    /// Applied to both the desktop adapter selection and the OpenXR diagnostic log; restart to change.
    initial_power_preference: wgpu::PowerPreference,
    /// Parsed `-LogLevel` from startup, if any. When [`Some`], always overrides [`crate::config::DebugSettings::log_verbose`].
    log_level_cli: Option<LogLevel>,
    /// Copied from host [`crate::shared::RendererInitData::output_device`] when the window is created.
    session_output_device: HeadOutputDevice,
    /// Center-eye pose for host IPC ([`crate::xr::headset_center_pose_from_stereo_views`], Unity-style
    /// [`crate::xr::openxr_pose_to_host_tracking`]), not the GPU rendering basis.
    cached_head_pose: Option<(Vec3, Quat)>,
    /// Controller states from the same XR tick’s [`crate::xr::OpenxrInput::sync_and_sample`] as `cached_head_pose`.
    cached_openxr_controllers: Vec<VRControllerState>,
    window: Option<Arc<Window>>,
    gpu: Option<GpuContext>,
    /// Set by the winit handler; read by [`crate::app::run`] for process exit.
    pub(crate) exit_code: Option<i32>,
    last_log_flush: Option<Instant>,
    input: WindowInputAccumulator,
    /// Host cursor lock transitions (unlock warp parity with Unity mouse driver).
    cursor_output_tracking: CursorOutputTracking,
    /// OpenXR bootstrap plus stereo swapchain/depth and mirror blit when the Vulkan path succeeded.
    xr_session: Option<XrSessionBundle>,
    /// Previous redraw instant for HUD FPS ([`crate::diagnostics::DebugHud`]).
    hud_frame_last: Option<Instant>,
    /// Wall-clock start of the last [`Self::tick_frame`]; anchors desktop FPS caps so the cap
    /// is a true period between frame starts rather than an end-to-start spacing that would
    /// add [`Self::tick_frame`]'s own duration on top of `1/cap`.
    last_frame_start: Option<Instant>,
    /// Wall-clock end of the previous [`Self::tick_frame`]; used only to emit the
    /// [`crate::profiling::plot_event_loop_idle_ms`] Tracy plot at the top of the next tick so
    /// the true inter-frame idle (winit `WaitUntil` parking plus any driver/compositor block)
    /// is visible alongside the frame mark. Not used for pacing decisions.
    previous_tick_end: Option<Instant>,
    /// OS-driven graceful shutdown (Unix signals or Windows Ctrl+C). See [`crate::app::startup`].
    external_shutdown: Option<ExternalShutdownCoordinator>,
    /// Cooperative hang detector heartbeat for the main / winit thread. Pet once per
    /// [`Self::tick_frame`] iteration. [`None`] when the watchdog is disabled by config or
    /// failed to install. See [`crate::diagnostics::Watchdog`].
    main_heartbeat: Option<crate::diagnostics::Heartbeat>,
}

/// Reconfigures the swapchain/depth for the given physical dimensions (shared by resize path and helpers).
fn reconfigure_gpu_for_physical_size(gpu: &mut GpuContext, width: u32, height: u32) {
    profiling::scope!("startup::reconfigure_gpu");
    gpu.reconfigure(width, height);
}

/// Reconfigures using the live window inner size from `gpu.window_inner_size()`.
///
/// Falls back to the cached config size if the GPU context has no window (headless or detached).
/// Used after `WindowEvent::ScaleFactorChanged` and as a recovery fallback after render-graph
/// errors, both of which want the freshest size winit can report.
fn reconfigure_gpu_for_window(gpu: &mut GpuContext) {
    let (w, h) = gpu
        .window_inner_size()
        .unwrap_or_else(|| gpu.surface_extent_px());
    reconfigure_gpu_for_physical_size(gpu, w, h);
}

impl RenderideApp {
    /// Builds initial app state after IPC bootstrap; window and GPU are created on [`ApplicationHandler::resumed`].
    pub(crate) fn new(
        runtime: RendererRuntime,
        initial_vsync: VsyncMode,
        initial_gpu_validation: bool,
        initial_power_preference: wgpu::PowerPreference,
        log_level_cli: Option<LogLevel>,
        external_shutdown: Option<ExternalShutdownCoordinator>,
        main_heartbeat: Option<crate::diagnostics::Heartbeat>,
    ) -> Self {
        Self {
            runtime,
            initial_vsync,
            initial_gpu_validation,
            initial_power_preference,
            log_level_cli,
            session_output_device: HeadOutputDevice::Screen,
            cached_head_pose: None,
            cached_openxr_controllers: Vec::new(),
            window: None,
            gpu: None,
            exit_code: None,
            last_log_flush: None,
            input: WindowInputAccumulator::default(),
            cursor_output_tracking: CursorOutputTracking::default(),
            xr_session: None,
            hud_frame_last: None,
            last_frame_start: None,
            previous_tick_end: None,
            external_shutdown,
            main_heartbeat,
        }
    }

    /// If graceful shutdown was requested (see [`crate::app::startup`]), optionally logs and exits the loop.
    fn check_external_shutdown(&mut self, event_loop: &ActiveEventLoop) -> bool {
        let Some(coord) = self.external_shutdown.as_ref() else {
            return false;
        };
        if !coord.requested.load(Ordering::Relaxed) {
            return false;
        }
        if coord.log_when_checked {
            logger::info!("Graceful shutdown requested; exiting event loop");
        }
        self.exit_code = Some(0);
        event_loop.exit();
        true
    }

    /// Records wall-clock frame start for FPS pacing; called from [`Self::frame_tick_prologue`].
    fn record_frame_tick_start(&mut self, frame_start: Instant) {
        self.last_frame_start = Some(frame_start);
    }

    fn maybe_flush_logs(&mut self) {
        let now = Instant::now();
        let should = self
            .last_log_flush
            .map(|t| now.duration_since(t) >= LOG_FLUSH_INTERVAL)
            .unwrap_or(true);
        if should {
            logger::flush();
            self.last_log_flush = Some(now);
        }
    }

    /// Applies [`effective_renderer_log_level`] from CLI and [`crate::config::DebugSettings::log_verbose`].
    fn sync_log_level_from_settings(&self) {
        let log_verbose = self
            .runtime
            .settings()
            .read()
            .map(|s| s.debug.log_verbose)
            .unwrap_or(false);
        logger::set_max_level(effective_renderer_log_level(
            self.log_level_cli,
            log_verbose,
        ));
    }

    fn ensure_window_gpu(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        profiling::scope!("startup::ensure_window_gpu");

        let attrs = winit::window::Window::default_attributes()
            .with_title("Renderide")
            .with_maximized(true)
            .with_visible(true)
            .with_window_icon(try_embedded_window_icon());

        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                logger::error!("create_window failed: {e}");
                self.exit_code = Some(1);
                event_loop.exit();
                return;
            }
        };

        let output_device = effective_output_device_for_gpu(self.runtime.pending_init());
        self.session_output_device = output_device;

        if let Some(init) = self.runtime.take_pending_init() {
            apply_window_title_from_init(&window, &init);
        }

        let wants_openxr = head_output_device_wants_openxr(output_device);
        if wants_openxr {
            match crate::xr::init_wgpu_openxr(
                self.initial_gpu_validation,
                self.initial_power_preference,
            ) {
                Ok(h) => {
                    match GpuContext::new_from_openxr_bootstrap(
                        &h.wgpu_instance,
                        &h.wgpu_adapter,
                        Arc::clone(&h.device),
                        Arc::clone(&h.queue),
                        Arc::clone(&window),
                        self.initial_vsync,
                    ) {
                        Ok(gpu) => {
                            logger::info!(
                                "GPU initialized (OpenXR Vulkan device + mirror surface)"
                            );
                            self.runtime.attach_gpu(&gpu);
                            self.gpu = Some(gpu);
                            self.xr_session = Some(XrSessionBundle::new(h));
                        }
                        Err(e) => {
                            // No fallback to desktop: the OpenXR runtime selected a Vulkan device
                            // that cannot present to the desktop mirror window, and silently
                            // dropping a half-built OpenXR session has caused a SIGSEGV inside the
                            // runtime in the past. Fail loud so the user can fix the underlying
                            // GPU mismatch (e.g. NVIDIA Optimus offload for the OpenXR runtime).
                            logger::error!(
                                "OpenXR mirror surface failed: {e}. \
                                 Renderer aborting — VR was requested and falling back to desktop \
                                 is unsafe with a partially-initialized OpenXR session."
                            );
                            self.exit_code = Some(2);
                            event_loop.exit();
                        }
                    }
                }
                Err(e) => {
                    logger::error!(
                        "OpenXR init failed: {e}. \
                         Renderer aborting — VR was requested; refusing to silently demote to desktop."
                    );
                    self.exit_code = Some(2);
                    event_loop.exit();
                }
            }
        } else {
            self.init_desktop_gpu(&window, event_loop);
        }

        if self.exit_code.is_some() {
            return;
        }

        self.window = Some(window);
        if let Some(w) = self.window.as_ref() {
            w.set_ime_allowed(true);
            self.input.sync_window_resolution_logical(w.as_ref());
        }
    }

    fn init_desktop_gpu(&mut self, window: &Arc<Window>, event_loop: &ActiveEventLoop) {
        match pollster::block_on(GpuContext::new(
            Arc::clone(window),
            self.initial_vsync,
            self.initial_gpu_validation,
            self.initial_power_preference,
        )) {
            Ok(gpu) => {
                logger::info!("GPU initialized (desktop)");
                self.runtime.attach_gpu(&gpu);
                self.gpu = Some(gpu);
            }
            Err(e) => {
                logger::error!("GPU init failed: {e}");
                self.exit_code = Some(1);
                event_loop.exit();
            }
        }
    }
}

impl ApplicationHandler for RenderideApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        profiling::scope!("app::resumed");
        event_loop.listen_device_events(DeviceEvents::Always);
        self.ensure_window_gpu(event_loop);
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        profiling::scope!("app::device_event");
        apply_device_event(&mut self.input, &event);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = self.window.as_ref() else {
            return;
        };
        if window.id() != window_id {
            return;
        }
        // Outer scope covers the full handling of one winit window event (input dispatch,
        // resize/scale reconfigure, redraw, log flush). Per-event kind sub-scopes emitted
        // inside `apply_window_event` (e.g. `frontend::window_event "cursor_moved"`) nest
        // underneath. The `RedrawRequested` arm additionally opens `app::redraw_requested`
        // so the frame-producing path is distinguishable at a glance.
        profiling::scope!("app::window_event");

        apply_window_event(&mut self.input, window, &event);

        match event {
            WindowEvent::CloseRequested => {
                logger::info!("Window close requested");
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                profiling::scope!("app::window_event_resize");
                if let Some(gpu) = self.gpu.as_mut() {
                    reconfigure_gpu_for_physical_size(gpu, size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                profiling::scope!("app::redraw_requested");
                if let Some(w) = self.window.as_ref() {
                    self.input.sync_window_resolution_logical(w.as_ref());
                }
                self.tick_frame(event_loop);
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                profiling::scope!("app::window_event_scale_factor");
                if let Some(gpu) = self.gpu.as_mut() {
                    reconfigure_gpu_for_window(gpu);
                }
            }
            _ => {}
        }

        {
            profiling::scope!("app::flush_logs");
            self.maybe_flush_logs();
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        profiling::scope!("app::about_to_wait");
        crate::profiling::plot_window_focused(self.input.window_focused);
        if self.check_external_shutdown(event_loop) {
            return;
        }
        if let Some(window) = self.window.as_ref() {
            if self.exit_code.is_none() && !self.runtime.vr_active() {
                let cap = match self.runtime.settings().read() {
                    Ok(s) => {
                        if self.input.window_focused {
                            s.display.focused_fps_cap
                        } else {
                            s.display.unfocused_fps_cap
                        }
                    }
                    Err(_) => 0,
                };
                crate::profiling::plot_fps_cap_active(cap);
                let now = Instant::now();
                if let Some(deadline) =
                    frame_pacing::next_redraw_wait_until(self.last_frame_start, cap, now)
                {
                    let wait_ms = deadline.saturating_duration_since(now).as_secs_f64() * 1000.0;
                    crate::profiling::plot_event_loop_wait_ms(wait_ms);
                    event_loop.set_control_flow(ControlFlow::WaitUntil(deadline));
                    profiling::scope!("app::flush_logs");
                    self.maybe_flush_logs();
                    return;
                }
                crate::profiling::plot_event_loop_wait_ms(0.0);
            } else {
                crate::profiling::plot_fps_cap_active(0);
                crate::profiling::plot_event_loop_wait_ms(0.0);
            }
            window.request_redraw();
        } else {
            crate::profiling::plot_fps_cap_active(0);
            crate::profiling::plot_event_loop_wait_ms(0.0);
        }
        if self.exit_code.is_none() {
            event_loop.set_control_flow(ControlFlow::Poll);
        }
        {
            profiling::scope!("app::flush_logs");
            self.maybe_flush_logs();
        }
    }
}
