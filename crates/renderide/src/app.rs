//! Application entry point: event loop, window lifecycle, and winit integration.
//!
//! Owns the RenderideApp handler that bridges winit events to the session, GPU, and render loop.
//! Swapchain recovery ([`wgpu::SurfaceError`], suboptimal acquire) is handled in [`recover_from_surface_error`]
//! and [`acquire_surface_texture_with_recovery`], with resize delegating to [`crate::gpu::reconfigure_surface_for_window`].

use std::path::Path;
use std::time::{Duration, Instant};

use logger::LogLevel;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalPosition;
use winit::event::{DeviceEvent, ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, DeviceEvents, EventLoop};
use winit::window::{CursorGrabMode, Window, WindowAttributes};

use crate::gpu::GpuState;
use crate::input::{WindowInputState, winit_key_to_renderite_key};
use crate::render::{RenderLoop, RenderTarget, RenderingContext, set_context};
use crate::session::Session;

/// Target frame interval when focused (240 Hz). Throttles redraws when using WaitUntil.
const FOCUSED_TARGET_INTERVAL: Duration = Duration::from_micros(1_000_000 / 240);
/// Target frame interval when unfocused (60 Hz).
const UNFOCUSED_TARGET_INTERVAL: Duration = Duration::from_micros(1_000_000 / 60);
/// Interval between log flushes.
const LOG_FLUSH_INTERVAL: Duration = Duration::from_secs(1);

/// Path to Renderide.log in the logs folder at repo root (two levels up from crates/renderide).
fn renderide_log_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap_or_else(|| Path::new("."))
        .join("logs")
        .join("Renderide.log")
}

/// Runs the Renderide application: initializes logging, panic hook, session, and event loop.
/// Returns the exit code if the session requested one, otherwise runs until the window is closed.
pub fn run() -> Option<i32> {
    let path = renderide_log_path();
    if let Err(e) = logger::init(
        &path,
        logger::parse_log_level_from_args().unwrap_or(LogLevel::Trace),
        true,
    ) {
        eprintln!("Failed to initialize logging to {}: {}", path.display(), e);
        return Some(1);
    }
    logger::info!("Logging to {}", path.display());

    let default_hook = std::panic::take_hook();
    let log_path = path.clone();
    std::panic::set_hook(Box::new(move |info| {
        logger::log_panic(&log_path, info);
        default_hook(info);
    }));

    let event_loop = match EventLoop::new() {
        Ok(el) => el,
        Err(e) => {
            logger::error!("EventLoop::new failed: {}", e);
            return Some(1);
        }
    };
    let mut app = RenderideApp::new();

    if let Err(e) = app.session.init() {
        logger::error!("Session init failed: {}", e);
        return Some(1);
    }

    let _ = event_loop.run_app(&mut app);

    app.exit_code
}

/// Interval (frames) between CPU/GPU bottleneck diagnostic logs.
const DIAGNOSTIC_LOG_INTERVAL: u32 = 60;

/// Applies a consistent policy to [`wgpu::SurfaceError`]: reconfigure the swapchain when the
/// platform indicates it is stale or unknown, request another redraw to retry, or only log for
/// transient errors such as [`wgpu::SurfaceError::Timeout`].
fn recover_from_surface_error(
    gpu: &mut GpuState,
    window: Option<&Window>,
    err: &wgpu::SurfaceError,
    context: &str,
) {
    let Some(window) = window else {
        logger::warn!("{}: {} (no window to reconfigure)", context, err);
        return;
    };
    match err {
        wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated => {
            logger::info!("{}: {} — reconfiguring swapchain", context, err);
            crate::gpu::reconfigure_surface_for_window(gpu, window, None);
            window.request_redraw();
        }
        wgpu::SurfaceError::Timeout => {
            logger::debug!("{}: {}", context, err);
        }
        wgpu::SurfaceError::OutOfMemory => {
            logger::error!("{}: {} — reconfiguring swapchain", context, err);
            crate::gpu::reconfigure_surface_for_window(gpu, window, None);
            window.request_redraw();
        }
        wgpu::SurfaceError::Other => {
            logger::warn!("{}: {} — reconfiguring swapchain", context, err);
            crate::gpu::reconfigure_surface_for_window(gpu, window, None);
            window.request_redraw();
        }
    }
}

/// Acquires the next swapchain texture. On [`wgpu::SurfaceError::Lost`] or
/// [`wgpu::SurfaceError::Outdated`], reconfigures once then retries acquire; other errors are
/// handled by [`recover_from_surface_error`] without a second acquire attempt.
fn acquire_surface_texture_with_recovery(
    gpu: &mut GpuState,
    window: &Window,
) -> Result<wgpu::SurfaceTexture, wgpu::SurfaceError> {
    match gpu.surface.get_current_texture() {
        Ok(texture) => Ok(texture),
        Err(e @ (wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated)) => {
            logger::info!(
                "Surface acquire: {} — reconfiguring swapchain and retrying once",
                e
            );
            crate::gpu::reconfigure_surface_for_window(gpu, window, None);
            match gpu.surface.get_current_texture() {
                Ok(texture) => Ok(texture),
                Err(e2) => {
                    recover_from_surface_error(
                        gpu,
                        Some(window),
                        &e2,
                        "Surface acquire (after retry)",
                    );
                    Err(e2)
                }
            }
        }
        Err(e) => {
            recover_from_surface_error(gpu, Some(window), &e, "Surface acquire");
            Err(e)
        }
    }
}

/// Accumulates frame timings for CPU/GPU bottleneck diagnosis.
///
/// Logs every `DIAGNOSTIC_LOG_INTERVAL` frames to `logs/Renderide.log` with average CPU
/// breakdown (session update, collect draw batches, render, present) and GPU mesh pass time.
/// Compares total CPU frame time vs GPU mesh pass time to infer bottleneck.
struct FrameDiagnostic {
    /// Number of frames accumulated since last log.
    frame_count: u32,
    /// Total microseconds in session update (phase Update).
    cpu_session_update_us: u64,
    /// Total microseconds in collect_draw_batches.
    cpu_collect_draw_batches_us: u64,
    /// Total microseconds in render_frame (CPU side).
    cpu_render_frame_us: u64,
    /// Total microseconds in present.
    cpu_present_us: u64,
    /// Total microseconds per frame.
    total_frame_us: u64,
}

impl FrameDiagnostic {
    fn new() -> Self {
        Self {
            frame_count: 0,
            cpu_session_update_us: 0,
            cpu_collect_draw_batches_us: 0,
            cpu_render_frame_us: 0,
            cpu_present_us: 0,
            total_frame_us: 0,
        }
    }

    fn add_frame(
        &mut self,
        session_us: u64,
        collect_us: u64,
        render_us: u64,
        present_us: u64,
        total_us: u64,
    ) {
        self.frame_count += 1;
        self.cpu_session_update_us += session_us;
        self.cpu_collect_draw_batches_us += collect_us;
        self.cpu_render_frame_us += render_us;
        self.cpu_present_us += present_us;
        self.total_frame_us += total_us;
    }

    fn log_and_reset(&mut self, gpu_mesh_pass_ms: Option<f64>) {
        if self.frame_count == 0 {
            return;
        }
        let n = self.frame_count as f64;
        let cpu_session_ms = self.cpu_session_update_us as f64 / 1000.0 / n;
        let cpu_collect_ms = self.cpu_collect_draw_batches_us as f64 / 1000.0 / n;
        let cpu_render_ms = self.cpu_render_frame_us as f64 / 1000.0 / n;
        let cpu_present_ms = self.cpu_present_us as f64 / 1000.0 / n;
        let cpu_total_ms = self.total_frame_us as f64 / 1000.0 / n;
        let bottleneck = match gpu_mesh_pass_ms {
            Some(gpu_ms) if gpu_ms > cpu_total_ms => "GPU",
            Some(_) => "CPU",
            None => "CPU (GPU timing unavailable)",
        };
        logger::info!(
            "[frame diag] frames={} CPU: session={:.2}ms collect={:.2}ms render={:.2}ms present={:.2}ms total={:.2}ms | GPU mesh_pass={:.2}ms | Bottleneck: {}",
            self.frame_count,
            cpu_session_ms,
            cpu_collect_ms,
            cpu_render_ms,
            cpu_present_ms,
            cpu_total_ms,
            gpu_mesh_pass_ms.unwrap_or(0.0),
            bottleneck
        );
        *self = Self::new();
    }
}

/// Application handler that owns the session, window, GPU state, and render loop.
struct RenderideApp {
    session: Session,
    window: Option<Window>,
    gpu: Option<GpuState>,
    render_loop: Option<RenderLoop>,
    exit_code: Option<i32>,
    input: WindowInputState,
    last_unfocused_redraw: Option<Instant>,
    last_log_flush: Option<Instant>,
    frame_diagnostic: FrameDiagnostic,
}

impl RenderideApp {
    fn new() -> Self {
        Self {
            session: Session::new(),
            window: None,
            gpu: None,
            render_loop: None,
            exit_code: None,
            input: WindowInputState::default(),
            last_unfocused_redraw: None,
            last_log_flush: None,
            frame_diagnostic: FrameDiagnostic::new(),
        }
    }

    /// Flushes logs if LOG_FLUSH_INTERVAL has passed since last flush.
    fn maybe_flush_logs(&mut self) {
        let now = Instant::now();
        let should_flush = self
            .last_log_flush
            .map(|t| now.duration_since(t) >= LOG_FLUSH_INTERVAL)
            .unwrap_or(true);
        if should_flush {
            logger::flush();
            self.last_log_flush = Some(now);
        }
    }

    /// Runs one frame: session update, GPU init (if needed), render, and render tasks.
    /// Returns `Some(exit_code)` if the session requested exit, otherwise `None`.
    fn run_frame(&mut self) -> Option<i32> {
        let frame_start = Instant::now();

        // Phase 1: Update — session update and command processing.
        if let Some(code) = self.session.update() {
            self.exit_code = Some(code);
            return Some(code);
        }
        let session_us = frame_start.elapsed().as_micros() as u64;

        if let (Some(window), None) = (&self.window, &self.gpu) {
            match pollster::block_on(crate::gpu::init_gpu(window)) {
                Ok(g) => {
                    logger::info!(
                        "GPU initialized: ray_tracing_available={}",
                        g.ray_tracing_available
                    );
                    self.render_loop = Some(RenderLoop::new(&g.device, &g.config));
                    self.gpu = Some(g);
                }
                Err(e) => {
                    logger::error!("GPU initialization failed: {}", e);
                    self.exit_code = Some(1);
                    return Some(1);
                }
            }
        }
        if let (Some(ref mut gpu), Some(ref mut render_loop)) =
            (self.gpu.as_mut(), self.render_loop.as_mut())
        {
            render_loop.drain_pending_camera_task_readbacks(&gpu.device, &mut self.session);
            // Phase 2: RenderToScreen — user view (main window).
            set_context(RenderingContext::user_view);
            for asset_id in self.session.drain_pending_mesh_unloads() {
                gpu.mesh_buffer_cache.remove(&asset_id);
                gpu.skinned_bind_group_cache
                    .retain(|(_, mid), _| *mid != asset_id);
                if let Some(ref mut accel) = gpu.accel_cache {
                    crate::gpu::remove_blas(accel, asset_id);
                }
            }
            for material_id in self.session.drain_pending_material_unloads() {
                render_loop.evict_material(material_id);
            }
            let draw_batches = self.session.collect_draw_batches();
            let window = self.window.as_ref();
            let t1 = Instant::now();
            let (render_result, collect_us, render_us) = match window {
                None => {
                    logger::warn!("GPU active without window; skipping main view render");
                    let collect_us = t1.elapsed().as_micros() as u64;
                    (Err(wgpu::SurfaceError::Other), collect_us, 0u64)
                }
                Some(w) => match acquire_surface_texture_with_recovery(gpu, w) {
                    Err(e) => {
                        let collect_us = t1.elapsed().as_micros() as u64;
                        (Err(e), collect_us, 0u64)
                    }
                    Ok(output) => {
                        let target = RenderTarget::from_surface_texture(output);
                        let viewport = target.dimensions();
                        let pre_collected = crate::render::prepare_mesh_draws_for_view(
                            gpu,
                            &self.session,
                            &draw_batches,
                            viewport,
                        );
                        let collect_us = t1.elapsed().as_micros() as u64;
                        let t2 = Instant::now();
                        let rendered = render_loop.render_frame(
                            gpu,
                            &self.session,
                            &draw_batches,
                            target,
                            Some(&pre_collected),
                        );
                        let render_us = t2.elapsed().as_micros() as u64;
                        if let Err(ref e) = rendered {
                            recover_from_surface_error(gpu, window, e, "Main view render");
                        }
                        (rendered, collect_us, render_us)
                    }
                },
            };

            let t3 = Instant::now();
            if let Ok(target) = render_result
                && let Some(surface_texture) = target.into_surface_texture()
            {
                let suboptimal = surface_texture.suboptimal;
                surface_texture.present();
                if suboptimal && let Some(w) = window {
                    logger::debug!(
                        "Swapchain suboptimal after present; reconfiguring for next frame"
                    );
                    crate::gpu::reconfigure_surface_for_window(gpu, w, None);
                    w.request_redraw();
                }
            }
            let present_us = t3.elapsed().as_micros() as u64;

            self.session.send_lights_consumed_for_rendered_spaces();

            let total_us = frame_start.elapsed().as_micros() as u64;
            self.frame_diagnostic
                .add_frame(session_us, collect_us, render_us, present_us, total_us);
            if self.frame_diagnostic.frame_count >= DIAGNOSTIC_LOG_INTERVAL {
                let gpu_ms = render_loop.last_gpu_mesh_pass_ms();
                self.frame_diagnostic.log_and_reset(gpu_ms);
            }
        }

        // Phase 3: RenderToAsset — offscreen camera tasks.
        set_context(RenderingContext::render_to_asset);
        self.session
            .process_render_tasks(self.gpu.as_mut(), self.render_loop.as_mut());

        self.maybe_flush_logs();
        None
    }
}

impl ApplicationHandler for RenderideApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.listen_device_events(DeviceEvents::Always);
        if self.window.is_none() {
            let attrs = WindowAttributes::default().with_title("Renderide");
            match event_loop.create_window(attrs) {
                Ok(w) => self.window = Some(w),
                Err(e) => logger::error!("Failed to create window: {}", e),
            }
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            self.input.mouse_delta.x += delta.0 as f32;
            self.input.mouse_delta.y -= delta.1 as f32;
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                if let Some(ref window) = self.window {
                    let size = window.inner_size();
                    self.input.window_resolution = (size.width, size.height);
                    let center =
                        nalgebra::Vector2::new((size.width / 2) as f32, (size.height / 2) as f32);
                    let lock = self.session.cursor_lock_requested();

                    if lock {
                        let _ = window
                            .set_cursor_grab(CursorGrabMode::Locked)
                            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
                        window.set_cursor_visible(false);
                        let center_phys = PhysicalPosition::new(size.width / 2, size.height / 2);
                        let _ = window.set_cursor_position(center_phys);
                        self.input.window_position = center;
                    } else {
                        let _ = window.set_cursor_grab(CursorGrabMode::None);
                        window.set_cursor_visible(true);
                        if !self.input.window_focused {
                            self.input.window_position = center;
                        }
                    }
                }

                let mut input = self.input.take_input_state();
                if let Some(ref mut m) = input.mouse {
                    m.is_active = m.is_active || self.session.cursor_lock_requested();
                }
                self.session.set_pending_input(input);
                if self.run_frame().is_some() {
                    event_loop.exit();
                }
            }
            WindowEvent::Resized(size) => {
                self.input.window_resolution = (size.width, size.height);
                if let (Some(ref mut gpu), Some(window)) = (self.gpu.as_mut(), self.window.as_ref())
                {
                    crate::gpu::reconfigure_surface_for_window(
                        gpu,
                        window,
                        Some((size.width, size.height)),
                    );
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.input.window_position.x = position.x as f32;
                self.input.window_position.y = position.y as f32;
            }
            WindowEvent::CursorEntered { .. } => self.input.mouse_active = true,
            WindowEvent::CursorLeft { .. } => self.input.mouse_active = false,
            WindowEvent::Focused(focused) => self.input.window_focused = focused,
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                match button {
                    MouseButton::Left => self.input.left_held = pressed,
                    MouseButton::Right => self.input.right_held = pressed,
                    MouseButton::Middle => self.input.middle_held = pressed,
                    MouseButton::Back => self.input.button4_held = pressed,
                    MouseButton::Forward => self.input.button5_held = pressed,
                    MouseButton::Other(_) => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                const SCROLL_SCALE: f32 = 120.0;
                match delta {
                    MouseScrollDelta::LineDelta(x, y) => {
                        self.input.scroll_delta.x += x * SCROLL_SCALE;
                        self.input.scroll_delta.y += y * SCROLL_SCALE;
                    }
                    MouseScrollDelta::PixelDelta(p) => {
                        self.input.scroll_delta.x += p.x as f32;
                        self.input.scroll_delta.y += p.y as f32;
                    }
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.repeat {
                    return;
                }
                if let Some(key) = winit_key_to_renderite_key(event.physical_key) {
                    match event.state {
                        ElementState::Pressed => {
                            if !self.input.held_keys.contains(&key) {
                                self.input.held_keys.push(key);
                            }
                        }
                        ElementState::Released => {
                            self.input.held_keys.retain(|held| *held != key);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(ref window) = self.window {
            if self.input.window_focused {
                self.last_unfocused_redraw = None;
                window.request_redraw();
                event_loop.set_control_flow(ControlFlow::WaitUntil(
                    Instant::now() + FOCUSED_TARGET_INTERVAL,
                ));
            } else {
                event_loop.set_control_flow(ControlFlow::WaitUntil(
                    Instant::now() + UNFOCUSED_TARGET_INTERVAL,
                ));

                let now = Instant::now();
                let should_redraw = self
                    .last_unfocused_redraw
                    .map(|t| now.duration_since(t) >= UNFOCUSED_TARGET_INTERVAL)
                    .unwrap_or(true);
                if should_redraw {
                    self.last_unfocused_redraw = Some(now);
                    let mut input = self.input.take_input_state();
                    if let Some(ref mut m) = input.mouse {
                        m.is_active = m.is_active || self.session.cursor_lock_requested();
                    }
                    self.session.set_pending_input(input);
                    if self.run_frame().is_some() {
                        event_loop.exit();
                    }
                }
            }
        }
    }
}
