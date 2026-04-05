//! Winit [`ApplicationHandler`]: window creation, GPU init, IPC-driven tick, and present.
//!
//! The main window is created maximized via [`winit::window::Window::default_attributes`] and
//! [`with_maximized(true)`](winit::window::WindowAttributes::with_maximized), which winit maps to
//! the appropriate Win32, X11, and Wayland behavior.

use std::sync::Arc;
use std::time::{Duration, Instant};

use logger::{LogComponent, LogLevel};
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, DeviceEvents, EventLoop};
use winit::window::{Window, WindowId};

use crate::config::{load_renderer_settings, log_config_resolve_trace, RendererSettings};
use crate::connection::{get_connection_parameters, try_claim_renderer_singleton};
use crate::frontend::input::{
    apply_device_event, apply_output_state_to_window, apply_window_event, WindowInputAccumulator,
};
use crate::gpu::GpuContext;
use crate::present::present_clear_frame;
use crate::render_graph::GraphExecuteError;
use crate::runtime::RendererRuntime;

/// Interval between log flushes when using file logging.
const LOG_FLUSH_INTERVAL: Duration = Duration::from_secs(1);

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
    let renderer_settings = config_load.settings;

    let default_hook = std::panic::take_hook();
    let log_path_hook = log_path.clone();
    std::panic::set_hook(Box::new(move |info| {
        logger::log_panic(&log_path_hook, info);
        default_hook(info);
    }));

    let params = get_connection_parameters();
    let mut runtime = RendererRuntime::new(params.clone());
    if let Err(e) = runtime.connect_ipc() {
        if params.is_some() {
            logger::error!("IPC connect failed: {e}");
            return Some(1);
        }
    }

    if params.is_some() && runtime.is_ipc_connected() {
        logger::info!("IPC connected (Primary/Background)");
    } else if params.is_some() {
        logger::warn!("IPC params present but connection state unexpected");
    } else {
        logger::info!("Standalone mode (no -QueueName/-QueueCapacity)");
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
        renderer_settings,
        window: None,
        gpu: None,
        exit_code: None,
        last_log_flush: None,
        input: WindowInputAccumulator::default(),
        #[cfg(feature = "debug-hud")]
        hud_frame_last: None,
    };

    let _ = event_loop.run_app(&mut app);
    app.exit_code
}

/// Winit-owned state: [`RendererRuntime`], plus lazily created window and [`GpuContext`].
struct RenderideApp {
    runtime: RendererRuntime,
    /// Loaded once at startup from `config.ini` (or defaults).
    #[allow(dead_code)]
    renderer_settings: RendererSettings,
    window: Option<Arc<Window>>,
    gpu: Option<GpuContext>,
    exit_code: Option<i32>,
    last_log_flush: Option<Instant>,
    input: WindowInputAccumulator,
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

        if let Some(init) = self.runtime.take_pending_init() {
            if let Some(ref title) = init.window_title {
                window.set_title(title);
            }
        }

        match pollster::block_on(GpuContext::new(Arc::clone(&window), false)) {
            Ok(gpu) => {
                logger::info!("GPU initialized");
                self.runtime.attach_gpu(&gpu);
                self.gpu = Some(gpu);
            }
            Err(e) => {
                logger::error!("GPU init failed: {e}");
                self.exit_code = Some(1);
                event_loop.exit();
                return;
            }
        }

        self.window = Some(window);
        if let Some(w) = self.window.as_ref() {
            w.set_ime_allowed(true);
        }
    }

    fn tick_frame(&mut self, event_loop: &ActiveEventLoop) {
        self.runtime.poll_ipc();

        if let (Some(window), Some(out)) = (
            self.window.as_ref(),
            self.runtime.take_pending_output_state(),
        ) {
            if let Err(e) = apply_output_state_to_window(window.as_ref(), &out) {
                logger::debug!("apply_output_state_to_window: {e:?}");
            }
        }

        if self.runtime.should_send_begin_frame() {
            let lock = self.runtime.host_cursor_lock_requested();
            let inputs = self.input.take_input_state(lock);
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

        let Some(window) = self.window.as_ref() else {
            return;
        };
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };

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

        if let Err(e) = self.runtime.execute_frame_graph(gpu, window) {
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

        apply_window_event(&mut self.input, &event);

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
                    let s = w.inner_size();
                    self.input.window_resolution = (s.width, s.height);
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
