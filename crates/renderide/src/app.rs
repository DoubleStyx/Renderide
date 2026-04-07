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
//! The mirror uses the normal render graph. When `vr_active` and multiview are available, the headset path renders
//! once to the OpenXR array swapchain and ends the frame with a projection layer; otherwise the
//! mirror window is rendered and the frame ends empty.
//!
//! VR **IPC input** (a non-empty [`InputState::vr`](crate::shared::InputState)) is sent whenever
//! [`Self::session_output_device`] is VR-capable so the host can create headset devices. If OpenXR
//! init fails, the app falls back to desktop GPU while still sending VR IPC input when the session
//! device is VR-capable.

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use logger::{LogComponent, LogLevel};
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, DeviceEvents, EventLoop};
use winit::window::{Window, WindowId};

use crate::config::{load_renderer_settings, log_config_resolve_trace, settings_handle_from};
use crate::connection::{get_connection_parameters, try_claim_renderer_singleton};
use crate::frontend::input::{
    apply_device_event, apply_output_state_to_window, apply_per_frame_cursor_lock_when_locked,
    apply_window_event, vr_inputs_for_session, CursorOutputTracking, WindowInputAccumulator,
};
use crate::frontend::InitState;
use crate::gpu::GpuContext;
use crate::output_device::head_output_device_wants_openxr;
use crate::present::present_clear_frame;
use crate::render_graph::ExternalFrameTargets;
use crate::render_graph::GraphExecuteError;
use crate::runtime::RendererRuntime;
use crate::shared::{HeadOutputDevice, RendererInitData, VRControllerState};
use glam::{Quat, Vec3};
use openxr as xr;

/// Cached OpenXR frame state after a single `wait_frame` (no second wait per tick).
struct OpenxrFrameTick {
    predicted_display_time: xr::Time,
    should_render: bool,
    views: Vec<xr::View>,
}

/// Interval between log flushes when using file logging.
const LOG_FLUSH_INTERVAL: Duration = Duration::from_secs(1);

/// Max time to wait for [`RendererInitData`] after IPC connect before exiting with an error.
const IPC_INIT_WAIT_TIMEOUT: Duration = Duration::from_secs(60);

/// Runs the winit event loop until exit or window close.
pub fn run() -> Option<i32> {
    if let Err(e) = try_claim_renderer_singleton() {
        eprintln!("{e}");
        return Some(1);
    }

    let timestamp = logger::log_filename_timestamp();
    let log_level = logger::parse_log_level_from_args().unwrap_or(LogLevel::Info);
    let log_path = match logger::init_for(LogComponent::Renderer, &timestamp, log_level, false) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to initialize logging: {e}");
            return Some(1);
        }
    };

    logger::info!("Logging to {}", log_path.display());

    let config_load = load_renderer_settings();
    log_config_resolve_trace(&config_load.resolve);
    let settings_handle = settings_handle_from(&config_load);
    let initial_vsync = config_load.settings.rendering.vsync;

    let default_hook = std::panic::take_hook();
    let log_path_hook = log_path.clone();
    std::panic::set_hook(Box::new(move |info| {
        logger::log_panic(&log_path_hook, info);
        default_hook(info);
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
        #[cfg(feature = "debug-hud")]
        hud_frame_last: None,
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
    /// Previous redraw instant for HUD FPS ([`diagnostics::DebugHud`]).
    #[cfg(feature = "debug-hud")]
    hud_frame_last: Option<Instant>,
}

impl RenderideApp {
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
            match crate::xr::init_wgpu_openxr() {
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
        match pollster::block_on(GpuContext::new(Arc::clone(window), self.initial_vsync)) {
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

        let xr_tick = self.openxr_begin_frame_and_stereo_matrices();

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
            if let (Some(v0), Some((p, q))) = (tick.views.first(), self.cached_head_pose) {
                let r = &v0.pose.position;
                logger::trace!(
                    "headset IPC: raw OpenXR views[0] pos=({:.3},{:.3},{:.3}) packed center pos=({:.3},{:.3},{:.3}) quat=({:.4},{:.4},{:.4},{:.4})",
                    r.x,
                    r.y,
                    r.z,
                    p.x,
                    p.y,
                    p.z,
                    q.x,
                    q.y,
                    q.z,
                    q.w
                );
            }
        }

        if self.runtime.should_send_begin_frame() {
            let lock = self.runtime.host_cursor_lock_requested();
            let mut inputs = self.input.take_input_state(lock);
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
            return;
        }

        if self.runtime.fatal_error() {
            logger::error!("Renderer fatal IPC error");
            self.exit_code = Some(4);
            event_loop.exit();
            return;
        }

        let Some(window) = self.window.clone() else {
            return;
        };

        let hmd_projection_ended = if let Some(ref tick) = xr_tick {
            self.try_openxr_hmd_multiview_submit(window.as_ref(), tick)
        } else {
            false
        };

        let Some(gpu) = self.gpu.as_mut() else {
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

        if !hmd_projection_ended {
            if let Err(e) = self.runtime.execute_frame_graph(gpu, window.as_ref()) {
                Self::handle_frame_graph_error(gpu, window.as_ref(), e);
            }
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
    }

    /// Single `wait_frame` + `locate_views` for stereo uniforms; used for both mirror and HMD paths.
    fn openxr_begin_frame_and_stereo_matrices(&mut self) -> Option<OpenxrFrameTick> {
        let handles = self.xr_handles.as_mut()?;
        let _ = handles.xr_session.poll_events();
        let fs = handles.xr_session.wait_frame().ok()??;
        let views = if fs.should_render {
            handles
                .xr_session
                .locate_views(fs.predicted_display_time)
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        if views.len() >= 2 {
            let near = self.runtime.host_camera.near_clip;
            let far = self.runtime.host_camera.far_clip;
            // Left from views[1], right from views[0]: matches `end_frame_projection` layer 0/1 order.
            let l = crate::xr::view_projection_from_xr_view(&views[1], near, far);
            let r = crate::xr::view_projection_from_xr_view(&views[0], near, far);
            self.runtime.set_stereo_view_proj(Some((l, r)));
        }
        Some(OpenxrFrameTick {
            predicted_display_time: fs.predicted_display_time,
            should_render: fs.should_render,
            views,
        })
    }

    /// Renders to the OpenXR stereo swapchain and calls [`crate::xr::session::XrSessionState::end_frame_projection`].
    ///
    /// Uses the same [`xr::FrameState`] as [`Self::openxr_begin_frame_and_stereo_matrices`] — no second `wait_frame`.
    fn try_openxr_hmd_multiview_submit(&mut self, window: &Window, tick: &OpenxrFrameTick) -> bool {
        let Some(gpu) = self.gpu.as_mut() else {
            return false;
        };
        let Some(handles) = self.xr_handles.as_mut() else {
            return false;
        };
        if !handles.xr_session.session_running() {
            return false;
        }
        if !self.runtime.host_camera.vr_active {
            return false;
        }
        if !gpu.device().features().contains(wgpu::Features::MULTIVIEW) {
            return false;
        }
        if !tick.should_render || tick.views.len() < 2 {
            return false;
        }
        if self.xr_swapchain.is_none() {
            let sys_id = handles.xr_system_id;
            let session = handles.xr_session.xr_vulkan_session();
            let inst = handles.xr_session.xr_instance();
            let dev = handles.device.as_ref();
            let res = unsafe { crate::xr::XrStereoSwapchain::new(session, inst, sys_id, dev) };
            match res {
                Ok(sc) => {
                    logger::info!(
                        "OpenXR swapchain {}×{} (stereo array)",
                        sc.resolution.0,
                        sc.resolution.1
                    );
                    self.xr_swapchain = Some(sc);
                }
                Err(e) => {
                    logger::debug!("OpenXR swapchain not created: {e}");
                    return false;
                }
            }
        }
        let sc = match self.xr_swapchain.as_mut() {
            Some(s) => s,
            None => return false,
        };
        let image_index = match sc.handle.acquire_image() {
            Ok(i) => i as usize,
            Err(_) => return false,
        };
        if sc.handle.wait_image(xr::Duration::INFINITE).is_err() {
            return false;
        }
        let Some(color_view) = sc.color_view_for_image(image_index) else {
            let _ = sc.handle.release_image();
            return false;
        };
        let extent = sc.resolution;
        let need_new_depth = self
            .xr_stereo_depth
            .as_ref()
            .map(|(tex, _)| {
                tex.size().width != extent.0
                    || tex.size().height != extent.1
                    || tex.size().depth_or_array_layers != crate::xr::XR_VIEW_COUNT
            })
            .unwrap_or(true);
        if need_new_depth {
            let (dt, dv) = crate::xr::create_stereo_depth_texture(gpu.device().as_ref(), extent);
            self.xr_stereo_depth = Some((dt, dv));
        }
        let depth_pair = &self
            .xr_stereo_depth
            .as_ref()
            .expect("xr_stereo_depth set above when missing")
            .1;
        let ext = ExternalFrameTargets {
            color_view,
            depth_view: depth_pair,
            extent_px: extent,
            surface_format: crate::xr::XR_COLOR_FORMAT,
        };
        let rect = xr::Rect2Di {
            offset: xr::Offset2Di { x: 0, y: 0 },
            extent: xr::Extent2Di {
                width: extent.0 as i32,
                height: extent.1 as i32,
            },
        };
        let views_ref = tick.views.as_slice();
        if self
            .runtime
            .execute_frame_graph_external_multiview(gpu, window, ext)
            .is_err()
        {
            let _ = sc.handle.release_image();
            return false;
        }
        if sc.handle.release_image().is_err() {
            return false;
        }
        if handles
            .xr_session
            .end_frame_projection(tick.predicted_display_time, &sc.handle, views_ref, rect)
            .is_err()
        {
            return false;
        }
        true
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
            window.request_redraw();
        }
        if self.exit_code.is_none() {
            event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
        }
        self.maybe_flush_logs();
    }
}
