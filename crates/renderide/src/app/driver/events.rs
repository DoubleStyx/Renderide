//! Winit event handling for the app driver.

use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, DeviceEvents};
use winit::window::WindowId;

use crate::config::VsyncMode;
use crate::frontend::input::{apply_device_event, apply_window_event};

use super::super::exit::ExitReason;
use super::super::redraw_plan::{RedrawDecision, RedrawInputs, plan_redraw};
use super::shortcuts::{fullscreen_toggle_shortcut, imgui_visibility_shortcut};
use super::{AppDriver, RenderTarget};

impl AppDriver {
    #[cfg(feature = "tracy")]
    fn begin_about_to_wait_span(&mut self) {
        self.about_to_wait_span.begin_if_empty("app::about_to_wait");
    }

    #[cfg(not(feature = "tracy"))]
    fn begin_about_to_wait_span(&self) {
        self.about_to_wait_span.begin_if_empty("app::about_to_wait");
    }

    #[cfg(feature = "tracy")]
    fn end_about_to_wait_span(&mut self) {
        self.about_to_wait_span.end();
    }

    #[cfg(not(feature = "tracy"))]
    fn end_about_to_wait_span(&self) {
        self.about_to_wait_span.end();
    }

    fn ensure_render_target(&mut self, event_loop: &dyn ActiveEventLoop) {
        if self.target.is_some() {
            return;
        }
        profiling::scope!("startup::ensure_render_target");
        match RenderTarget::create(
            event_loop,
            &mut self.runtime,
            self.startup_gpu,
            &self.display_handle,
        ) {
            Ok(target) => {
                let window = target.window();
                self.input.sync_window_resolution_logical(window.as_ref());
                self.input.set_window_focused(window.has_focus());
                self.input.set_fullscreen(target.is_fullscreen());
                self.target = Some(target);
                self.apply_pending_window_icons();
            }
            Err(error) => {
                logger::error!("{error}");
                self.request_exit(error.exit_reason(), event_loop);
            }
        }
    }

    fn handle_redraw_requested(&mut self, event_loop: &dyn ActiveEventLoop) {
        profiling::scope!("app::redraw_requested");
        if self.exit_is_requested() {
            self.poll_graceful_shutdown(event_loop);
            return;
        }
        if let Some(target) = self.target.as_ref() {
            self.input.set_fullscreen(target.is_fullscreen());
            self.input
                .sync_window_resolution_logical(target.window().as_ref());
        }
        self.tick_frame(event_loop);
    }
}

impl ApplicationHandler for AppDriver {
    fn can_create_surfaces(&mut self, event_loop: &dyn ActiveEventLoop) {
        self.end_about_to_wait_span();
        self.ensure_render_target(event_loop);
    }

    fn resumed(&mut self, event_loop: &dyn ActiveEventLoop) {
        self.end_about_to_wait_span();
        profiling::scope!("app::resumed");
        if self.exit_is_requested() {
            return;
        }
        event_loop.listen_device_events(DeviceEvents::Always);
        self.ensure_render_target(event_loop);
    }

    fn device_event(
        &mut self,
        _event_loop: &dyn ActiveEventLoop,
        _device_id: Option<winit::event::DeviceId>,
        event: DeviceEvent,
    ) {
        self.end_about_to_wait_span();
        profiling::scope!("app::device_event");
        if self.exit_is_requested() {
            return;
        }
        apply_device_event(&mut self.input, &event);
    }

    fn window_event(
        &mut self,
        event_loop: &dyn ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let redraw_requested = {
            self.end_about_to_wait_span();
            profiling::scope!("app::window_event");
            let Some(target) = self.target.as_ref() else {
                return;
            };
            if target.window().id() != window_id {
                return;
            }
            if matches!(event, WindowEvent::RedrawRequested) {
                true
            } else {
                let window = std::sync::Arc::clone(target.window());

                self.handle_window_event(event_loop, window, event);
                false
            }
        };

        if redraw_requested {
            self.handle_redraw_requested(event_loop);
            self.flush_logs_if_due();
        }
    }

    fn about_to_wait(&mut self, event_loop: &dyn ActiveEventLoop) {
        self.begin_about_to_wait_span();
        profiling::scope!("app::about_to_wait_body");
        let window_has_keyboard_focus = self
            .target
            .as_ref()
            .map_or(self.input.window_focused, |target| {
                target.window().has_focus()
            });
        if self.input.window_focused != window_has_keyboard_focus {
            self.input.set_window_focused(window_has_keyboard_focus);
        }
        crate::profiling::plot_window_focused(self.input.window_focused);
        if self.exit_is_requested() {
            self.poll_graceful_shutdown(event_loop);
            self.flush_logs_if_due();
            return;
        }
        if self.check_external_shutdown(event_loop) {
            self.flush_logs_if_due();
            return;
        }

        let wants_more_idle_asset_work = self
            .runtime
            .run_asset_integration_while_waiting_for_submit(std::time::Instant::now());

        let vsync = self
            .runtime
            .settings()
            .read()
            .map(|settings| settings.rendering.vsync)
            .unwrap_or(VsyncMode::Off);
        let frame_pacing_caps = self.runtime.desktop_frame_pacing_caps();
        let plan = plan_redraw(RedrawInputs {
            has_window: self.target.is_some(),
            exit_requested: self.exit_is_requested(),
            hmd_compositor_paced: self.hmd_compositor_paced_last_frame,
            vsync,
            window_has_keyboard_focus: self.input.window_focused,
            foreground_fps_cap: frame_pacing_caps.foreground_fps_cap,
            background_fps_cap: frame_pacing_caps.background_fps_cap,
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
                if wants_more_idle_asset_work {
                    event_loop.set_control_flow(ControlFlow::Poll);
                    self.flush_logs_if_due();
                    return;
                }
                if let Some(target) = self.target.as_ref() {
                    target.window().request_redraw();
                }
            }
            RedrawDecision::Idle => {}
        }

        if !self.exit_is_requested() {
            event_loop.set_control_flow(ControlFlow::Poll);
        }
        self.flush_logs_if_due();
    }
}

impl AppDriver {
    fn handle_window_event(
        &mut self,
        event_loop: &dyn ActiveEventLoop,
        window: std::sync::Arc<dyn winit::window::Window>,
        event: WindowEvent,
    ) {
        if imgui_visibility_shortcut(&event) {
            self.runtime.toggle_imgui_visibility();
            window.request_redraw();
            self.flush_logs_if_due();
            return;
        }

        apply_window_event(&mut self.input, window.as_ref(), &event);

        if fullscreen_toggle_shortcut(&event, self.input.keyboard_modifiers())
            && let Some(target) = self.target.as_ref()
        {
            let fullscreen = target.toggle_borderless_fullscreen();
            self.input.set_fullscreen(fullscreen);
            logger::info!(
                "Window fullscreen {}",
                if fullscreen { "enabled" } else { "disabled" }
            );
            window.request_redraw();
        }

        match event {
            WindowEvent::CloseRequested => {
                logger::info!("Window close requested");
                self.request_exit(ExitReason::WindowClosed, event_loop);
            }
            WindowEvent::SurfaceResized(size) => {
                profiling::scope!("app::window_event_resize");
                if !self.exit_is_requested()
                    && let Some(target) = self.target.as_mut()
                {
                    target.reconfigure_physical_size(size.width, size.height);
                }
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                profiling::scope!("app::window_event_scale_factor");
                if !self.exit_is_requested()
                    && let Some(target) = self.target.as_mut()
                {
                    target.reconfigure_for_window();
                }
            }
            _ => {}
        }

        self.flush_logs_if_due();
    }
}
