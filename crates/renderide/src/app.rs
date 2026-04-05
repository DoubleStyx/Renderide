//! Winit [`ApplicationHandler`]: window creation, GPU init, IPC-driven tick, and present.

use std::sync::Arc;
use std::time::{Duration, Instant};

use logger::{LogComponent, LogLevel};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

use crate::connection::{get_connection_parameters, try_claim_renderer_singleton};
use crate::gpu::GpuContext;
use crate::present::present_clear_frame;
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
        window: None,
        gpu: None,
        exit_code: None,
        last_log_flush: None,
    };

    let _ = event_loop.run_app(&mut app);
    app.exit_code
}

/// Winit-owned state: [`RendererRuntime`], plus lazily created window and [`GpuContext`].
struct RenderideApp {
    runtime: RendererRuntime,
    window: Option<Arc<Window>>,
    gpu: Option<GpuContext>,
    exit_code: Option<i32>,
    last_log_flush: Option<Instant>,
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
                self.runtime
                    .attach_gpu(gpu.device().clone(), Arc::clone(gpu.queue()));
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
    }

    fn tick_frame(&mut self, event_loop: &ActiveEventLoop) {
        self.runtime.pre_frame();
        self.runtime.poll_ipc();

        if self.runtime.shutdown_requested {
            logger::info!("Renderer shutdown requested by host");
            self.exit_code = Some(0);
            event_loop.exit();
            return;
        }

        if self.runtime.fatal_error {
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

        if let Err(e) = present_clear_frame(gpu, window) {
            logger::warn!("present failed: {e:?}");
            let s = window.inner_size();
            gpu.reconfigure(s.width, s.height);
        }
    }
}

impl ApplicationHandler for RenderideApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.ensure_window_gpu(event_loop);
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
