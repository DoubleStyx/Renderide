//! Winit event handling for the app driver.

use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, DeviceEvents};
use winit::window::WindowId;

use crate::frontend::input::{apply_device_event, apply_window_event};

use super::super::exit::ExitReason;
use super::super::frame_clock::{RedrawDecision, RedrawInputs, plan_redraw};
use super::{AppDriver, RenderTarget};

impl AppDriver {
    fn ensure_render_target(&mut self, event_loop: &ActiveEventLoop) {
        if self.target.is_some() {
            return;
        }
        profiling::scope!("startup::ensure_render_target");
        match RenderTarget::create(event_loop, &mut self.runtime, self.startup_gpu) {
            Ok(target) => {
                self.input
                    .sync_window_resolution_logical(target.window().as_ref());
                self.target = Some(target);
            }
            Err(error) => {
                logger::error!("{error}");
                self.request_exit(error.exit_reason(), event_loop);
            }
        }
    }

    fn flush_logs_if_due(&mut self) {
        profiling::scope!("app::flush_logs");
        self.log_flush.flush_if_due();
    }
}

impl ApplicationHandler for AppDriver {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        profiling::scope!("app::resumed");
        if self.exit.is_requested() {
            return;
        }
        event_loop.listen_device_events(DeviceEvents::Always);
        self.ensure_render_target(event_loop);
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        profiling::scope!("app::device_event");
        if self.exit.is_requested() {
            return;
        }
        apply_device_event(&mut self.input, &event);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(target) = self.target.as_ref() else {
            return;
        };
        if target.window().id() != window_id {
            return;
        }

        profiling::scope!("app::window_event");
        apply_window_event(&mut self.input, target.window(), &event);

        match event {
            WindowEvent::CloseRequested => {
                logger::info!("Window close requested");
                self.request_exit(ExitReason::WindowClosed, event_loop);
            }
            WindowEvent::Resized(size) => {
                profiling::scope!("app::window_event_resize");
                if !self.exit.is_requested()
                    && let Some(target) = self.target.as_mut()
                {
                    target.reconfigure_physical_size(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                profiling::scope!("app::redraw_requested");
                if self.exit.is_requested() {
                    self.poll_graceful_shutdown(event_loop);
                    self.flush_logs_if_due();
                    return;
                }
                if let Some(target) = self.target.as_ref() {
                    self.input
                        .sync_window_resolution_logical(target.window().as_ref());
                }
                self.tick_frame(event_loop);
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                profiling::scope!("app::window_event_scale_factor");
                if !self.exit.is_requested()
                    && let Some(target) = self.target.as_mut()
                {
                    target.reconfigure_for_window();
                }
            }
            _ => {}
        }

        self.flush_logs_if_due();
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        profiling::scope!("app::about_to_wait");
        crate::profiling::plot_window_focused(self.input.window_focused);
        if self.exit.is_requested() {
            self.poll_graceful_shutdown(event_loop);
            self.flush_logs_if_due();
            return;
        }
        if self.check_external_shutdown(event_loop) {
            self.flush_logs_if_due();
            return;
        }

        if self
            .runtime
            .run_asset_integration_while_waiting_for_submit(std::time::Instant::now())
        {
            event_loop.set_control_flow(ControlFlow::Poll);
            self.flush_logs_if_due();
            return;
        }

        let (focused_fps_cap, unfocused_fps_cap) = self
            .runtime
            .settings()
            .read()
            .map(|settings| {
                (
                    settings.display.focused_fps_cap,
                    settings.display.unfocused_fps_cap,
                )
            })
            .unwrap_or((0, 0));
        let plan = plan_redraw(RedrawInputs {
            has_window: self.target.is_some(),
            exit_requested: self.exit.is_requested(),
            vr_active: self.runtime.vr_active(),
            window_focused: self.input.window_focused,
            focused_fps_cap,
            unfocused_fps_cap,
            last_frame_start: self.frame_clock.last_frame_start(),
            now: std::time::Instant::now(),
        });

        crate::profiling::plot_fps_cap_active(plan.fps_cap);
        crate::profiling::plot_event_loop_wait_ms(plan.wait_ms);

        match plan.decision {
            RedrawDecision::WaitUntil(deadline) => {
                event_loop.set_control_flow(ControlFlow::WaitUntil(deadline));
                self.flush_logs_if_due();
                return;
            }
            RedrawDecision::RedrawNow => {
                if let Some(target) = self.target.as_ref() {
                    target.window().request_redraw();
                }
            }
            RedrawDecision::Idle => {}
        }

        if !self.exit.is_requested() {
            event_loop.set_control_flow(ControlFlow::Poll);
        }
        self.flush_logs_if_due();
    }
}
