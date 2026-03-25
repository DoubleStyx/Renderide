//! Application entry point: event loop, window lifecycle, and winit integration.
//!
//! The main window is created maximized via [`winit::window::WindowAttributes::with_maximized`], which winit maps
//! to the appropriate Win32, X11, and Wayland behavior.
//!
//! [`ApplicationHandler::about_to_wait`] schedules redraws using caps from `configuration.ini`
//! (`[display]` `focused_fps` / `unfocused_fps` via [`crate::config::AppConfig`]): it uses
//! [`ControlFlow::WaitUntil`] with the matching frame interval, or [`ControlFlow::Poll`] when
//! `focused_fps = 0` (fully uncapped while focused). The `RedrawRequested` event applies the
//! same caps using a shared `last_redraw` timestamp so Windows `WM_PAINT` storms cannot bypass the
//! throttle. [`crate::config::RenderConfig::vsync`] selects tear-reducing swapchain presentation
//! vs `wgpu::PresentMode::AutoNoVsync` without changing that FPS pacing.
//!
//! Owns the RenderideApp handler that bridges winit events to the session, GPU, and render loop.
//! Swapchain recovery ([`wgpu::SurfaceError`], suboptimal acquire) is handled in [`recover_from_surface_error`]
//! and [`acquire_surface_texture_with_recovery`], with resize delegating to [`crate::gpu::reconfigure_surface_for_window`].
//!
//! ## Frame-rate and HUD configuration
//!
//! At startup the app reads `configuration.ini` (next to the exe or in the cwd) via
//! [`crate::config::AppConfig::load`] and [`crate::config::RenderConfig::load`].  The following
//! keys are honoured (see [`crate::config`] for every `[camera]` / `[rendering]` field):
//!
//! | Section     | Key              | Default | Description                                      |
//! |-------------|------------------|---------|--------------------------------------------------|
//! | `[display]` | `focused_fps`    | 240     | Max FPS when window is focused (0 = uncapped).   |
//! | `[display]` | `unfocused_fps`  | 60      | Max FPS when window is unfocused (0 = uncapped). |
//! | `[display]` | `vsync`          | false   | Swapchain vsync (also in [`RenderConfig`]).      |
//! | `[camera]`  | `near_clip`, `far_clip`, `desktop_fov` | per `RenderConfig` | Host overwrites when connected. |
//! | `[rendering]` | `ray_traced_shadows_enabled` | false | PBR ray-query / TLAS shadows (opt-in; needs RT GPU). |
//! | `[hud]`     | `show_hud`       | true    | Show/hide the debug HUD overlay.                 |

use std::path::Path;
use std::time::{Duration, Instant};

use logger::LogLevel;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalPosition;
use winit::event::{DeviceEvent, ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, DeviceEvents, EventLoop};
use winit::window::{CursorGrabMode, Window, WindowAttributes};

use crate::config::AppConfig;
use crate::diagnostics::{
    DebugHud, GpuAllocatorSnapshot, HostCpuMemorySnapshot, LiveFrameDiagnostics,
};
use crate::gpu::GpuState;
use crate::input::{WindowInputState, winit_key_to_renderite_key};
use crate::render::{MainViewFrameInput, RenderLoop, RenderTarget, RenderingContext, set_context};
use crate::session::Session;
use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};

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
///
/// Default max log level is [`LogLevel::Info`] when `-LogLevel` is omitted; pass `-LogLevel debug`
/// or `trace` for frame diagnostics and per-frame render traces.
/// Returns the exit code if the session requested one, otherwise runs until the window is closed.
pub fn run() -> Option<i32> {
    let path = renderide_log_path();
    if let Err(e) = logger::init(
        &path,
        logger::parse_log_level_from_args().unwrap_or(LogLevel::Info),
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
/// Emits one line every `DIAGNOSTIC_LOG_INTERVAL` frames at **debug** level with average CPU
/// breakdown (session update, collect draw batches, render, present) and GPU mesh pass time.
/// Compares total CPU frame time vs GPU mesh pass time to infer bottleneck. Use `-LogLevel debug`
/// (or `trace`) to record these lines in `logs/Renderide.log`.
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
        logger::debug!(
            "[frame diag] frames={} CPU: session={:.2}ms collect+prep={:.2}ms render={:.2}ms present={:.2}ms total={:.2}ms | GPU mesh_pass={:.2}ms | Bottleneck: {}",
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
    /// Timestamp of the last rendered frame, shared by both focused and unfocused throttle paths.
    last_redraw: Option<Instant>,
    /// Wall-clock start of the previous `run_frame()` call, used for actual FPS tracking.
    last_frame_wall_start: Option<Instant>,
    last_log_flush: Option<Instant>,
    frame_diagnostic: FrameDiagnostic,
    debug_hud: Option<DebugHud>,
    /// Settings loaded from `configuration.ini` at startup.
    app_config: AppConfig,
    /// Lazily created host metrics source for the debug HUD (`sysinfo`).
    sysinfo_system: Option<System>,
}

impl RenderideApp {
    fn new() -> Self {
        let app_config = AppConfig::load();
        logger::info!(
            "AppConfig: focused_fps={} unfocused_fps={} show_hud={}",
            app_config.focused_fps,
            app_config.unfocused_fps,
            app_config.show_hud,
        );
        Self {
            session: Session::new(),
            window: None,
            gpu: None,
            render_loop: None,
            exit_code: None,
            input: WindowInputState::default(),
            last_redraw: None,
            last_frame_wall_start: None,
            last_log_flush: None,
            frame_diagnostic: FrameDiagnostic::new(),
            debug_hud: None,
            app_config,
            sysinfo_system: None,
        }
    }

    /// Returns the target frame interval for the **focused** state.
    ///
    /// `focused_fps = 0` → [`Duration::ZERO`] which is used to set [`ControlFlow::Poll`]
    /// (fully uncapped).
    fn focused_interval(&self) -> Duration {
        if self.app_config.focused_fps == 0 {
            Duration::ZERO
        } else {
            Duration::from_micros(1_000_000 / self.app_config.focused_fps as u64)
        }
    }

    /// Returns the target frame interval for the **unfocused** (tabbed-out) state.
    ///
    /// `unfocused_fps = 0` → 1 ms (effectively uncapped but still uses `WaitUntil` to
    /// avoid a pure spin loop draining the CPU).
    fn unfocused_interval(&self) -> Duration {
        if self.app_config.unfocused_fps == 0 {
            Duration::from_millis(1)
        } else {
            Duration::from_micros(1_000_000 / self.app_config.unfocused_fps as u64)
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

    /// Runs one frame on the winit event-loop thread.
    ///
    /// Phases:
    /// 1. **Update** — [`Session::update`](crate::session::Session::update) with
    ///    [`WindowInputState`](crate::input::WindowInputState) (IPC / commands; input sampled only when
    ///    `frame_start_data` is sent).
    /// 2. **Main view** — [`MainViewFrameInput::from_session`] (draw batches), swapchain acquire,
    ///    [`crate::render::prepare_mesh_draws_for_view`], [`RenderLoop::render_frame`], present.
    /// 3. **Render-to-asset** — [`Session::process_render_tasks`] (offscreen camera tasks).
    ///
    /// Returns `Some(exit_code)` if the session requested exit, otherwise `None`.
    fn run_frame(&mut self) -> Option<i32> {
        let frame_start = Instant::now();
        let wall_interval_us = self
            .last_frame_wall_start
            .map(|t| frame_start.duration_since(t).as_micros() as u64)
            .unwrap_or(0);
        self.last_frame_wall_start = Some(frame_start);

        // Phase 1: Update — session update and command processing.
        if let Some(code) = self.session.update(&mut self.input) {
            self.exit_code = Some(code);
            return Some(code);
        }
        let session_us = frame_start.elapsed().as_micros() as u64;

        if let Some(ref mut gpu) = self.gpu {
            gpu.set_present_mode_for_vsync(self.session.render_config().vsync);
        }

        if let (Some(window), None) = (&self.window, &self.gpu) {
            match pollster::block_on(crate::gpu::init_gpu(
                window,
                self.session.render_config().vsync,
                self.session.render_config().gpu_validation_layers,
                self.session.render_config().ray_tracing_enabled,
                self.session.render_config().use_opengl,
                self.session.render_config().use_dx12,
            )) {
                Ok(g) => {
                    logger::info!(
                        "GPU initialized: ray_tracing_available={}",
                        g.ray_tracing_available
                    );
                    // Only create the HUD if show_hud = true in configuration.ini.
                    self.debug_hud = if self.app_config.show_hud {
                        match DebugHud::new(&g.device, &g.queue, g.config.format) {
                            Ok(hud) => Some(hud),
                            Err(e) => {
                                logger::warn!("Debug HUD init failed: {}", e);
                                None
                            }
                        }
                    } else {
                        logger::info!(
                            "Debug HUD disabled via configuration.ini (show_hud = false)"
                        );
                        None
                    };
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
            for tex_id in self.session.drain_pending_texture_unloads() {
                gpu.drop_texture2d(tex_id);
            }
            for material_id in self.session.drain_pending_material_unloads() {
                render_loop.evict_material(material_id);
            }
            let t1 = Instant::now();
            let main_view_input = MainViewFrameInput::from_session(&mut self.session);
            // Time IPC batch collection separately from mesh culling/GPU upload.
            let ipc_us = t1.elapsed().as_micros() as u64;

            let window = self.window.as_ref();
            let (render_result, collect_us, ipc_collect_us, mesh_prep_us, render_us, prep_stats) =
                match window {
                    None => {
                        logger::warn!("GPU active without window; skipping main view render");
                        let collect_us = t1.elapsed().as_micros() as u64;
                        (
                            Err(wgpu::SurfaceError::Other),
                            collect_us,
                            ipc_us,
                            0u64,
                            0u64,
                            crate::render::pass::MeshDrawPrepStats::default(),
                        )
                    }
                    Some(w) => match acquire_surface_texture_with_recovery(gpu, w) {
                        Err(e) => {
                            let collect_us = t1.elapsed().as_micros() as u64;
                            (
                                Err(e),
                                collect_us,
                                ipc_us,
                                collect_us.saturating_sub(ipc_us),
                                0u64,
                                crate::render::pass::MeshDrawPrepStats::default(),
                            )
                        }
                        Ok(output) => {
                            let target = RenderTarget::from_surface_texture(output);
                            let viewport = target.dimensions();
                            let t_prep = Instant::now();
                            let pre_collected = crate::render::prepare_mesh_draws_for_view(
                                gpu,
                                &self.session,
                                &main_view_input.draw_batches,
                                viewport,
                            );
                            let mesh_prep_us = t_prep.elapsed().as_micros() as u64;
                            let collect_us = t1.elapsed().as_micros() as u64;
                            let t2 = Instant::now();
                            let rendered = render_loop.render_frame(
                                gpu,
                                &self.session,
                                &main_view_input.draw_batches,
                                target,
                                Some(&pre_collected),
                            );
                            let render_us = t2.elapsed().as_micros() as u64;
                            if let Err(ref e) = rendered {
                                recover_from_surface_error(gpu, window, e, "Main view render");
                            }
                            (
                                rendered,
                                collect_us,
                                ipc_us,
                                mesh_prep_us,
                                render_us,
                                pre_collected.prep_stats,
                            )
                        }
                    },
                };

            let t3 = Instant::now();
            if let Ok(target) = render_result {
                // HUD rendering is skipped when show_hud = false (debug_hud will be None).
                if let Some(debug_hud) = self.debug_hud.as_mut()
                    && let Err(e) = debug_hud.render(&gpu.device, &gpu.queue, &target, &self.input)
                {
                    logger::warn!("Debug HUD render failed: {}", e);
                }
                if let Some(surface_texture) = target.into_surface_texture() {
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

            let batch_count = main_view_input.draw_batches.len();
            let overlay_batch_count = main_view_input
                .draw_batches
                .iter()
                .filter(|b| b.is_overlay)
                .count();
            let total_draws_in_batches: usize = main_view_input
                .draw_batches
                .iter()
                .map(|b| b.draws.len())
                .sum();
            let overlay_draws_in_batches: usize = main_view_input
                .draw_batches
                .iter()
                .filter(|b| b.is_overlay)
                .map(|b| b.draws.len())
                .sum();
            if self.sysinfo_system.is_none() && sysinfo::IS_SUPPORTED_SYSTEM {
                self.sysinfo_system = Some(System::new_with_specifics(
                    RefreshKind::nothing()
                        .with_cpu(CpuRefreshKind::everything())
                        .with_memory(MemoryRefreshKind::everything()),
                ));
            }
            let host = if let Some(ref mut sys) = self.sysinfo_system {
                sys.refresh_cpu_usage();
                sys.refresh_memory();
                let cpu_model = sys
                    .cpus()
                    .first()
                    .map(|c| c.brand().to_string())
                    .unwrap_or_default();
                HostCpuMemorySnapshot {
                    cpu_model,
                    logical_cpus: sys.cpus().len(),
                    cpu_usage_percent: sys.global_cpu_usage(),
                    ram_total_bytes: sys.total_memory(),
                    ram_used_bytes: sys.used_memory(),
                }
            } else {
                HostCpuMemorySnapshot {
                    cpu_model: if sysinfo::IS_SUPPORTED_SYSTEM {
                        String::new()
                    } else {
                        "unsupported platform".into()
                    },
                    ..Default::default()
                }
            };
            let gpu_allocator = gpu
                .device
                .generate_allocator_report()
                .map(|r| GpuAllocatorSnapshot {
                    allocated_bytes: Some(r.total_allocated_bytes),
                    reserved_bytes: Some(r.total_reserved_bytes),
                })
                .unwrap_or_default();
            let rc = self.session.render_config();
            let live_sample = LiveFrameDiagnostics {
                frame_index: self.session.last_frame_index(),
                viewport: (gpu.config.width.max(1), gpu.config.height.max(1)),
                // CPU phase timings
                session_update_us: session_us,
                ipc_collect_us,
                mesh_prep_us,
                collect_us,
                render_us,
                present_us,
                total_us,
                wall_interval_us,
                // GPU timing
                gpu_mesh_pass_ms: render_loop.last_gpu_mesh_pass_ms(),
                // Draw stats
                batch_count,
                overlay_batch_count,
                total_draws_in_batches,
                overlay_draws_in_batches,
                prep_stats,
                mesh_cache_count: gpu.mesh_buffer_cache.len(),
                pending_render_tasks: self.session.pending_render_task_count(),
                pending_camera_task_readbacks: render_loop.pending_camera_task_readback_count(),
                // Lights
                gpu_light_count: gpu.light_count,
                // RT / RTAO
                blas_count: gpu.accel_cache.as_ref().map(|a| a.len()).unwrap_or(0),
                tlas_available: gpu
                    .ray_tracing_state
                    .as_ref()
                    .and_then(|rt| rt.tlas.as_ref())
                    .is_some(),
                ao_radius: rc.ao_radius,
                ao_strength: rc.rtao_strength,
                ao_sample_count: 4,
                // Feature flags
                frustum_culling_enabled: rc.frustum_culling,
                rtao_enabled: rc.rtao_enabled,
                ray_traced_shadows_enabled: rc.ray_traced_shadows_enabled,
                ray_tracing_available: gpu.ray_tracing_available,
                adapter_info: gpu.adapter_info.clone(),
                gpu_allocator,
                host,
            };
            if let Some(debug_hud) = self.debug_hud.as_mut() {
                debug_hud.update(live_sample);
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
            let attrs = WindowAttributes::default()
                .with_title("Renderide")
                .with_maximized(true);
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
                // ── Frame-rate throttle ──────────────────────────────────────────────
                // On Windows the OS generates continuous WM_PAINT messages that become
                // RedrawRequested events, completely bypassing about_to_wait throttling.
                // Both focused and unfocused paths are gated here using their respective
                // last-redraw timestamps.  about_to_wait only controls WaitUntil timing
                // and calls request_redraw() — it never calls run_frame() directly.
                // Pick the cap for the current focus state.
                let interval = if self.input.window_focused {
                    self.focused_interval()
                } else {
                    self.unfocused_interval()
                };
                if !interval.is_zero() {
                    let now = Instant::now();
                    let elapsed = self
                        .last_redraw
                        .map(|t| now.duration_since(t))
                        .unwrap_or(interval); // first frame: always render
                    if elapsed < interval {
                        return;
                    }
                    // Record BEFORE GPU work so the deadline is stable.
                    self.last_redraw = Some(now);
                }
                // ────────────────────────────────────────────────────────────────────
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
            // Both focused and unfocused use the same pattern:
            //   - request_redraw() only when the target interval has elapsed
            //   - WaitUntil the next deadline so the OS can sleep the thread
            // RedrawRequested does the actual throttle gate via last_redraw.
            // Using a single shared timestamp avoids resets when focus changes.
            let interval = if self.input.window_focused {
                self.focused_interval()
            } else {
                self.unfocused_interval()
            };
            if interval.is_zero() {
                // focused_fps = 0: fully uncapped.
                window.request_redraw();
                event_loop.set_control_flow(ControlFlow::Poll);
            } else {
                let now = Instant::now();
                let elapsed = self
                    .last_redraw
                    .map(|t| now.duration_since(t))
                    .unwrap_or(interval);
                if elapsed >= interval {
                    window.request_redraw();
                }
                let next_frame = self.last_redraw.map(|t| t + interval).unwrap_or(now);
                event_loop.set_control_flow(ControlFlow::WaitUntil(next_frame));
            }
        }
    }
}

#[cfg(test)]
mod frame_interval_tests {
    use std::time::Duration;

    /// Fallback display refresh in millihertz when the platform does not report a rate (60 Hz).
    const FALLBACK_REFRESH_MILLIHERTZ: u32 = 60_000;
    /// Minimum refresh treated as valid when clamping winit-reported millihertz (10 Hz).
    const MIN_REFRESH_MILLIHERTZ: u32 = 10_000;
    /// Maximum refresh treated as valid when clamping winit-reported millihertz (480 Hz).
    const MAX_REFRESH_MILLIHERTZ: u32 = 480_000;

    /// Converts a display refresh rate in millihertz to a frame interval (e.g. 60 Hz → `60_000`).
    fn interval_from_refresh_millihertz(millihertz: Option<u32>) -> Duration {
        let mhz = millihertz
            .unwrap_or(FALLBACK_REFRESH_MILLIHERTZ)
            .clamp(MIN_REFRESH_MILLIHERTZ, MAX_REFRESH_MILLIHERTZ);
        let nanos = 1_000_000_000_000u64 / u64::from(mhz);
        Duration::from_nanos(nanos.max(1))
    }

    #[test]
    fn sixty_hz_interval_matches_period() {
        let d = interval_from_refresh_millihertz(Some(60_000));
        let expected_ns = 1_000_000_000_000u64 / 60_000;
        assert_eq!(d.as_nanos(), expected_ns as u128);
    }

    #[test]
    fn none_uses_fallback_sixty_hz() {
        let d = interval_from_refresh_millihertz(None);
        let expected_ns = 1_000_000_000_000u64 / 60_000;
        assert_eq!(d.as_nanos(), expected_ns as u128);
    }
}
