//! Winit [`ApplicationHandler`]: window creation, GPU init, IPC-driven tick, and present.
//!
//! The main window is created maximized via [`winit::window::Window::default_attributes`] and
//! [`with_maximized(true)`](winit::window::WindowAttributes::with_maximized), which winit maps to
//! the appropriate Win32, X11, and Wayland behavior.
//!
//! When the host selects a VR [`HeadOutputDevice`](crate::shared::HeadOutputDevice), the Vulkan
//! device may come from [`crate::xr::init_wgpu_openxr`]; the mirror window uses the same device.
//! Each frame: OpenXR `wait_frame` / `locate_views` run **before** lock-step `pre_frame` so headset
//! pose in [`InputState::vr`](crate::shared::InputState) matches the same `locate_views` snapshot.
//! The desktop window uses the normal render graph when VR is inactive. When `vr_active` and multiview
//! are available, the headset path renders once to the OpenXR array swapchain and ends the frame with a
//! projection layer; the desktop window shows a **blit of the left-eye** HMD output (no second world render).
//! When the HMD path does not run, the window is cleared for that frame.
//!
//! VR **IPC input** (a non-empty [`InputState::vr`](crate::shared::InputState)) is sent whenever
//! [`Self::session_output_device`] is VR-capable so the host can create headset devices. If OpenXR
//! init fails, the app falls back to desktop GPU while still sending VR IPC input when the session
//! device is VR-capable.

mod frame_loop;
mod frame_pacing;

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use logger::{LogComponent, LogLevel};
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, DeviceEvents, EventLoop};
use winit::window::{Window, WindowId};

use crate::config::{load_renderer_settings, log_config_resolve_trace, settings_handle_from};
use crate::connection::{get_connection_parameters, try_claim_renderer_singleton};
use crate::frontend::input::{
    apply_device_event, apply_output_state_to_window, apply_per_frame_cursor_lock_when_locked,
    apply_window_event, vr_inputs_for_session, CursorOutputTracking, WindowInputAccumulator,
};
use crate::frontend::InitState;
use crate::gpu::{GpuContext, VrMirrorBlitResources};
use crate::output_device::head_output_device_wants_openxr;
use crate::present::present_clear_frame;
use crate::render_graph::GraphExecuteError;
use crate::runtime::RendererRuntime;
use crate::shared::{HeadOutputDevice, RendererInitData, VRControllerState};
use glam::{Quat, Vec3};

/// Interval between log flushes when using file logging.
const LOG_FLUSH_INTERVAL: Duration = Duration::from_secs(1);

/// Max time to wait for [`RendererInitData`] after IPC connect before exiting with an error.
const IPC_INIT_WAIT_TIMEOUT: Duration = Duration::from_secs(60);

/// Chooses the process max log level after [`logger::init_for`].
///
/// Precedence: **`-LogLevel`** (if present) always wins. If absent, [`crate::config::DebugSettings::log_verbose`]
/// selects [`LogLevel::Trace`] when true and [`LogLevel::Debug`] when false.
fn effective_renderer_log_level(cli: Option<LogLevel>, log_verbose: bool) -> LogLevel {
    if let Some(level) = cli {
        level
    } else if log_verbose {
        LogLevel::Trace
    } else {
        LogLevel::Debug
    }
}

/// Runs the winit event loop until exit or window close.
pub fn run() -> Option<i32> {
    if let Err(e) = try_claim_renderer_singleton() {
        eprintln!("{e}");
        return Some(1);
    }

    let timestamp = logger::log_filename_timestamp();
    let log_level_cli = logger::parse_log_level_from_args();
    let initial_log_level = log_level_cli.unwrap_or(LogLevel::Debug);
    let log_path =
        match logger::init_for(LogComponent::Renderer, &timestamp, initial_log_level, false) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Failed to initialize logging: {e}");
                return Some(1);
            }
        };

    logger::info!("Logging to {}", log_path.display());

    // Vulkan validation (spirv-val, VK_LAYER_KHRONOS_validation) and other native code often use
    // stdout and/or stderr; forward both so messages reach the log file regardless of wgpu flags
    // or `VK_INSTANCE_LAYERS` (see `native_stdio`).
    crate::native_stdio::ensure_stdio_forwarded_to_logger();

    let config_load = load_renderer_settings();
    logger::set_max_level(effective_renderer_log_level(
        log_level_cli,
        config_load.settings.debug.log_verbose,
    ));
    log_config_resolve_trace(&config_load.resolve);
    let settings_handle = settings_handle_from(&config_load);
    let initial_vsync = config_load.settings.rendering.vsync;
    let initial_gpu_validation = config_load.settings.debug.gpu_validation_layers;

    let log_path_hook = log_path.clone();
    std::panic::set_hook(Box::new(move |info| {
        let report = logger::panic_report(info);
        logger::append_panic_report_to_file(&log_path_hook, &report);
        crate::native_stdio::try_write_preserved_stderr(report.as_bytes());
    }));

    let params = get_connection_parameters();
    let mut runtime = RendererRuntime::new(
        params.clone(),
        settings_handle,
        config_load.save_path.clone(),
    );
    if let Err(e) = runtime.connect_ipc() {
        if params.is_some() {
            logger::error!("IPC connect failed: {e}");
            return Some(1);
        }
    }

    if params.is_some() && runtime.is_ipc_connected() {
        logger::info!("IPC connected (Primary/Background)");
        if wait_for_renderer_init_data(&mut runtime).is_err() {
            return Some(1);
        }
    } else if params.is_some() {
        logger::warn!("IPC params present but connection state unexpected");
    } else {
        logger::info!("Standalone mode (no -QueueName/-QueueCapacity; desktop GPU, no host init)");
    }

    let event_loop = match EventLoop::new() {
        Ok(el) => el,
        Err(e) => {
            logger::error!("EventLoop::new failed: {e}");
            return Some(1);
        }
    };

    let mut app = RenderideApp {
        runtime,
        initial_vsync,
        initial_gpu_validation,
        log_level_cli,
        session_output_device: HeadOutputDevice::screen,
        cached_head_pose: None,
        cached_openxr_controllers: Vec::new(),
        window: None,
        gpu: None,
        exit_code: None,
        last_log_flush: None,
        input: WindowInputAccumulator::default(),
        cursor_output_tracking: CursorOutputTracking::default(),
        xr_handles: None,
        xr_swapchain: None,
        xr_stereo_depth: None,
        vr_mirror_blit: VrMirrorBlitResources::new(),
        #[cfg(feature = "debug-hud")]
        hud_frame_last: None,
        last_frame_end: None,
    };

    let _ = event_loop.run_app(&mut app);
    app.exit_code
}

/// Blocks until [`RendererInitData`] arrives or IPC fails (non-standalone only).
fn wait_for_renderer_init_data(runtime: &mut RendererRuntime) -> Result<(), ()> {
    let deadline = Instant::now() + IPC_INIT_WAIT_TIMEOUT;
    while runtime.init_state() == InitState::Uninitialized {
        if Instant::now() > deadline {
            logger::error!("Timed out waiting for RendererInitData from host");
            return Err(());
        }
        runtime.poll_ipc();
        if runtime.fatal_error() {
            logger::error!("Fatal IPC error while waiting for RendererInitData");
            return Err(());
        }
        thread::sleep(Duration::from_millis(1));
    }
    Ok(())
}

/// Standalone runs have no host init; IPC runs should have [`RendererInitData`] before the window exists.
fn effective_output_device_for_gpu(pending: Option<&RendererInitData>) -> HeadOutputDevice {
    pending
        .map(|i| i.output_device)
        .unwrap_or(HeadOutputDevice::screen)
}

fn apply_window_title_from_init(window: &Arc<Window>, init: &RendererInitData) {
    if let Some(ref title) = init.window_title {
        window.set_title(title);
    }
}

/// Winit-owned state: [`RendererRuntime`], plus lazily created window and [`GpuContext`].
struct RenderideApp {
    runtime: RendererRuntime,
    /// VSync flag used for the initial [`GpuContext::new`] before live updates from settings.
    initial_vsync: bool,
    /// GPU validation layers flag for the initial [`GpuContext::new`] (persisted; restart to apply).
    initial_gpu_validation: bool,
    /// Parsed `-LogLevel` from startup, if any. When [`Some`], always overrides [`crate::config::DebugSettings::log_verbose`].
    log_level_cli: Option<LogLevel>,
    /// Copied from host [`RendererInitData::output_device`] when the window is created.
    session_output_device: HeadOutputDevice,
    /// Center-eye pose for host IPC ([`crate::xr::headset_center_pose_from_stereo_views`], Unity-style
    /// [`crate::xr::openxr_pose_to_host_tracking`]), not the GPU rendering basis.
    cached_head_pose: Option<(Vec3, Quat)>,
    /// Controller states from the same XR tick’s [`crate::xr::OpenxrInput::sync_and_sample`] as `cached_head_pose`.
    cached_openxr_controllers: Vec<VRControllerState>,
    window: Option<Arc<Window>>,
    gpu: Option<GpuContext>,
    exit_code: Option<i32>,
    last_log_flush: Option<Instant>,
    input: WindowInputAccumulator,
    /// Host cursor lock transitions (unlock warp parity with Unity mouse driver).
    cursor_output_tracking: CursorOutputTracking,
    xr_handles: Option<crate::xr::XrWgpuHandles>,
    xr_swapchain: Option<crate::xr::XrStereoSwapchain>,
    xr_stereo_depth: Option<(wgpu::Texture, wgpu::TextureView)>,
    /// Staging texture and blit pipelines for the VR desktop mirror (left HMD eye).
    vr_mirror_blit: VrMirrorBlitResources,
    /// Previous redraw instant for HUD FPS ([`diagnostics::DebugHud`]).
    #[cfg(feature = "debug-hud")]
    hud_frame_last: Option<Instant>,
    /// Wall-clock end of the last [`Self::tick_frame`] (for desktop FPS caps).
    last_frame_end: Option<Instant>,
}

impl RenderideApp {
    /// Records wall-clock frame end for FPS pacing and forwards to [`RendererRuntime::tick_frame_wall_clock_end`].
    fn record_frame_tick_end(&mut self, frame_start: Instant) {
        self.last_frame_end = Some(Instant::now());
        self.runtime.tick_frame_wall_clock_end(frame_start);
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

        let attrs = winit::window::Window::default_attributes()
            .with_title("Renderide")
            .with_maximized(true)
            .with_visible(true);

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
            match crate::xr::init_wgpu_openxr(self.initial_gpu_validation) {
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
                            self.xr_handles = Some(h);
                        }
                        Err(e) => {
                            logger::warn!(
                                "OpenXR mirror surface failed; falling back to desktop GPU: {e}"
                            );
                            self.init_desktop_gpu(&window, event_loop);
                        }
                    }
                }
                Err(e) => {
                    logger::warn!("OpenXR init failed; falling back to desktop: {e}");
                    self.init_desktop_gpu(&window, event_loop);
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

    fn tick_frame(&mut self, event_loop: &ActiveEventLoop) {
        self.sync_log_level_from_settings();
        let frame_start = Instant::now();
        self.runtime.tick_frame_wall_clock_begin(frame_start);
        #[cfg(feature = "debug-hud")]
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.begin_frame_timing(frame_start);
        }

        self.runtime.poll_ipc();

        if let (Some(window), Some(out)) = (
            self.window.as_ref(),
            self.runtime.take_pending_output_state(),
        ) {
            if let Err(e) = apply_output_state_to_window(
                window.as_ref(),
                &out,
                &mut self.cursor_output_tracking,
            ) {
                logger::debug!("apply_output_state_to_window: {e:?}");
            }
        }

        if let Some(window) = self.window.as_ref() {
            if self.runtime.host_cursor_lock_requested() {
                let lock_pos = self
                    .runtime
                    .last_output_state()
                    .and_then(|s| s.lock_cursor_position);
                if let Err(e) = apply_per_frame_cursor_lock_when_locked(
                    window.as_ref(),
                    &mut self.input,
                    lock_pos,
                ) {
                    logger::debug!("apply_per_frame_cursor_lock_when_locked: {e:?}");
                }
            }
        }

        let xr_tick = self
            .xr_handles
            .as_mut()
            .and_then(|h| frame_loop::begin_openxr_frame_tick(h, &mut self.runtime));

        if let Some(ref tick) = xr_tick {
            crate::xr::OpenxrInput::log_stereo_view_order_once(&tick.views);
            if let Some(handles) = &self.xr_handles {
                if let Some(ref input) = handles.openxr_input {
                    if handles.xr_session.session_running() {
                        match input.sync_and_sample(
                            handles.xr_session.xr_vulkan_session(),
                            handles.xr_session.stage_space(),
                            tick.predicted_display_time,
                        ) {
                            Ok(v) => self.cached_openxr_controllers = v,
                            Err(e) => logger::trace!("OpenXR input sync: {e:?}"),
                        }
                    }
                }
            }
            self.cached_head_pose =
                crate::xr::headset_center_pose_from_stereo_views(tick.views.as_slice());
            if let (Some(v0), Some(v1), Some((ipc_p, ipc_q))) =
                (tick.views.first(), tick.views.get(1), self.cached_head_pose)
            {
                // Raw OpenXR view positions (what the renderer uses for view-projection)
                let rp0 = &v0.pose.position;
                let rp1 = &v1.pose.position;
                let render_center_x = (rp0.x + rp1.x) * 0.5;
                let render_center_y = (rp0.y + rp1.y) * 0.5;
                let render_center_z = (rp0.z + rp1.z) * 0.5;
                logger::debug!(
                    "HEAD POS | render(OpenXR RH): ({:.3},{:.3},{:.3}) | ipc->host(Unity LH): ({:.3},{:.3},{:.3}) | ipc_quat: ({:.4},{:.4},{:.4},{:.4})",
                    render_center_x, render_center_y, render_center_z,
                    ipc_p.x, ipc_p.y, ipc_p.z,
                    ipc_q.x, ipc_q.y, ipc_q.z, ipc_q.w,
                );
            }
        }

        if self.runtime.should_send_begin_frame() {
            let lock = self.runtime.host_cursor_lock_requested();
            let mut inputs = self.input.take_input_state(lock);
            #[cfg(feature = "debug-hud")]
            crate::diagnostics::sanitize_input_state_for_imgui_host(
                &mut inputs,
                self.runtime.debug_hud_last_want_capture_mouse(),
                self.runtime.debug_hud_last_want_capture_keyboard(),
            );
            if let Some(vr) = vr_inputs_for_session(
                self.session_output_device,
                self.cached_head_pose,
                &self.cached_openxr_controllers,
            ) {
                inputs.vr = Some(vr);
            }
            self.runtime.pre_frame(inputs);
        }

        if self.runtime.shutdown_requested() {
            logger::info!("Renderer shutdown requested by host");
            self.exit_code = Some(0);
            event_loop.exit();
            #[cfg(feature = "debug-hud")]
            self.end_frame_timing_and_hud_capture();
            self.record_frame_tick_end(frame_start);
            return;
        }

        if self.runtime.fatal_error() {
            logger::error!("Renderer fatal IPC error");
            self.exit_code = Some(4);
            event_loop.exit();
            #[cfg(feature = "debug-hud")]
            self.end_frame_timing_and_hud_capture();
            self.record_frame_tick_end(frame_start);
            return;
        }

        let Some(window) = self.window.clone() else {
            #[cfg(feature = "debug-hud")]
            self.end_frame_timing_and_hud_capture();
            self.record_frame_tick_end(frame_start);
            return;
        };

        let hmd_projection_ended = match (
            self.gpu.as_mut(),
            self.xr_handles.as_mut(),
            xr_tick.as_ref(),
        ) {
            (Some(gpu), Some(handles), Some(tick)) => frame_loop::try_hmd_multiview_submit(
                gpu,
                handles,
                &mut self.runtime,
                &mut self.xr_swapchain,
                &mut self.xr_stereo_depth,
                &mut self.vr_mirror_blit,
                window.as_ref(),
                tick,
            ),
            _ => false,
        };

        let Some(gpu) = self.gpu.as_mut() else {
            #[cfg(feature = "debug-hud")]
            self.end_frame_timing_and_hud_capture();
            self.record_frame_tick_end(frame_start);
            return;
        };

        if let Ok(s) = self.runtime.settings().read() {
            gpu.set_vsync(s.rendering.vsync);
        }

        #[cfg(feature = "debug-hud")]
        {
            let now = Instant::now();
            let ms = self
                .hud_frame_last
                .map(|t| now.duration_since(t).as_secs_f64() * 1000.0)
                .unwrap_or(16.67);
            self.hud_frame_last = Some(now);
            let hud_in =
                crate::diagnostics::DebugHudInput::from_winit(window.as_ref(), &self.input);
            self.runtime.set_debug_hud_frame_data(hud_in, ms);
        }

        // VR: desktop shows a blit of the left HMD eye (`VrMirrorBlitResources`); no second world pass.
        // Debug HUD overlay is not drawn on this path (see `frame_graph::compiled` for non-VR HUD).
        if self.runtime.host_camera.vr_active {
            if hmd_projection_ended {
                if let Err(e) = frame_loop::present_vr_mirror_blit(
                    gpu,
                    window.as_ref(),
                    &mut self.vr_mirror_blit,
                ) {
                    logger::debug!("VR mirror blit failed: {e:?}");
                    if let Err(pe) = present_clear_frame(gpu, window.as_ref()) {
                        logger::warn!("present_clear_frame after mirror blit: {pe:?}");
                    }
                }
            } else if let Err(e) = present_clear_frame(gpu, window.as_ref()) {
                logger::debug!("VR mirror clear (no HMD frame): {e:?}");
            }
        } else if let Err(e) =
            frame_loop::execute_mirror_frame_graph(&mut self.runtime, gpu, window.as_ref())
        {
            Self::handle_frame_graph_error(gpu, window.as_ref(), e);
        }

        if let (Some(handles), Some(tick)) = (self.xr_handles.as_mut(), xr_tick) {
            if !hmd_projection_ended {
                if let Err(e) = handles
                    .xr_session
                    .end_frame_empty(tick.predicted_display_time)
                {
                    logger::debug!("OpenXR end_frame_empty: {e:?}");
                }
            }
        }

        #[cfg(feature = "debug-hud")]
        self.end_frame_timing_and_hud_capture();
        self.record_frame_tick_end(frame_start);
    }

    /// Finalizes [`GpuContext`] frame timing and refreshes debug HUD snapshots for the tick.
    #[cfg(feature = "debug-hud")]
    fn end_frame_timing_and_hud_capture(&mut self) {
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.end_frame_timing();
            self.runtime.capture_debug_hud_after_frame_end(gpu);
        }
    }

    fn handle_frame_graph_error(gpu: &mut GpuContext, window: &Window, e: GraphExecuteError) {
        match e {
            GraphExecuteError::NoFrameGraph => {
                if let Err(pe) = present_clear_frame(gpu, window) {
                    logger::warn!("present fallback failed: {pe:?}");
                    let s = window.inner_size();
                    gpu.reconfigure(s.width, s.height);
                }
            }
            _ => {
                logger::warn!("frame graph failed: {e:?}");
                let s = window.inner_size();
                gpu.reconfigure(s.width, s.height);
            }
        }
    }
}

impl ApplicationHandler for RenderideApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.listen_device_events(DeviceEvents::Always);
        self.ensure_window_gpu(event_loop);
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
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

        apply_window_event(&mut self.input, window, &event);

        match event {
            WindowEvent::CloseRequested => {
                logger::info!("Window close requested");
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.reconfigure(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(w) = self.window.as_ref() {
                    self.input.sync_window_resolution_logical(w.as_ref());
                }
                self.tick_frame(event_loop);
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                let s = window.inner_size();
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.reconfigure(s.width, s.height);
                }
            }
            _ => {}
        }

        self.maybe_flush_logs();
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            if self.exit_code.is_none() && !self.runtime.host_camera.vr_active {
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
                let now = Instant::now();
                if let Some(deadline) =
                    frame_pacing::next_redraw_wait_until(self.last_frame_end, cap, now)
                {
                    event_loop.set_control_flow(ControlFlow::WaitUntil(deadline));
                    self.maybe_flush_logs();
                    return;
                }
            }
            window.request_redraw();
        }
        if self.exit_code.is_none() {
            event_loop.set_control_flow(ControlFlow::Poll);
        }
        self.maybe_flush_logs();
    }
}

#[cfg(test)]
mod effective_log_level_tests {
    use super::effective_renderer_log_level;
    use logger::LogLevel;

    #[test]
    fn cli_always_overrides_log_verbose() {
        assert_eq!(
            effective_renderer_log_level(Some(LogLevel::Warn), true),
            LogLevel::Warn
        );
    }

    #[test]
    fn no_cli_uses_trace_when_log_verbose() {
        assert_eq!(effective_renderer_log_level(None, true), LogLevel::Trace);
    }

    #[test]
    fn no_cli_uses_debug_when_not_log_verbose() {
        assert_eq!(effective_renderer_log_level(None, false), LogLevel::Debug);
    }
}
