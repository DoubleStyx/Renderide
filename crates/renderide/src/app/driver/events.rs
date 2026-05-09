//! Winit event handling for the app driver.

use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, DeviceEvents};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::WindowId;

use crate::frontend::input::{apply_device_event, apply_window_event};

use super::super::exit::ExitReason;
use super::super::frame_clock::{RedrawDecision, RedrawInputs, plan_redraw};
use super::{AppDriver, RenderTarget};

impl AppDriver {
    fn ensure_render_target(&mut self, event_loop: &dyn ActiveEventLoop) {
        if self.target.is_some() {
            return;
        }
        profiling::scope!("startup::ensure_render_target");
        match RenderTarget::create(event_loop, &mut self.runtime, self.startup_gpu) {
            Ok(target) => {
                self.input
                    .sync_window_resolution_logical(target.window().as_ref());
                self.input.set_fullscreen(target.is_fullscreen());
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
    fn can_create_surfaces(&mut self, event_loop: &dyn ActiveEventLoop) {
        self.ensure_render_target(event_loop);
    }

    fn resumed(&mut self, event_loop: &dyn ActiveEventLoop) {
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
        let Some(target) = self.target.as_ref() else {
            return;
        };
        if target.window().id() != window_id {
            return;
        }
        let window = std::sync::Arc::clone(target.window());

        profiling::scope!("app::window_event");
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
            WindowEvent::RedrawRequested => {
                profiling::scope!("app::redraw_requested");
                if self.exit_is_requested() {
                    self.poll_graceful_shutdown(event_loop);
                    self.flush_logs_if_due();
                    return;
                }
                if let Some(target) = self.target.as_ref() {
                    self.input.set_fullscreen(target.is_fullscreen());
                    self.input
                        .sync_window_resolution_logical(target.window().as_ref());
                }
                self.tick_frame(event_loop);
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

    fn about_to_wait(&mut self, event_loop: &dyn ActiveEventLoop) {
        profiling::scope!("app::about_to_wait");
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
            exit_requested: self.exit_is_requested(),
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

        if !self.exit_is_requested() {
            event_loop.set_control_flow(ControlFlow::Poll);
        }
        self.flush_logs_if_due();
    }
}

fn fullscreen_toggle_shortcut(event: &WindowEvent, modifiers: ModifiersState) -> bool {
    let WindowEvent::KeyboardInput {
        event,
        is_synthetic,
        ..
    } = event
    else {
        return false;
    };
    fullscreen_toggle_shortcut_from_parts(
        event.physical_key,
        event.state,
        event.repeat,
        *is_synthetic,
        modifiers,
    )
}

fn fullscreen_toggle_shortcut_from_parts(
    physical_key: PhysicalKey,
    state: ElementState,
    repeat: bool,
    is_synthetic: bool,
    modifiers: ModifiersState,
) -> bool {
    !is_synthetic
        && !repeat
        && state == ElementState::Pressed
        && modifiers.alt_key()
        && matches!(
            physical_key,
            PhysicalKey::Code(KeyCode::Enter | KeyCode::NumpadEnter)
        )
}

fn imgui_visibility_shortcut(event: &WindowEvent) -> bool {
    let WindowEvent::KeyboardInput {
        event,
        is_synthetic,
        ..
    } = event
    else {
        return false;
    };
    imgui_visibility_shortcut_from_parts(
        event.physical_key,
        event.state,
        event.repeat,
        *is_synthetic,
    )
}

fn imgui_visibility_shortcut_from_parts(
    physical_key: PhysicalKey,
    state: ElementState,
    repeat: bool,
    is_synthetic: bool,
) -> bool {
    !is_synthetic
        && !repeat
        && state == ElementState::Pressed
        && physical_key == PhysicalKey::Code(KeyCode::F7)
}

#[cfg(test)]
mod tests {
    use winit::event::ElementState;
    use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};

    use super::{fullscreen_toggle_shortcut_from_parts, imgui_visibility_shortcut_from_parts};

    #[test]
    fn fullscreen_toggle_shortcut_accepts_alt_enter_and_alt_numpad_enter() {
        assert!(fullscreen_toggle_shortcut_from_parts(
            PhysicalKey::Code(KeyCode::Enter),
            ElementState::Pressed,
            false,
            false,
            ModifiersState::ALT,
        ));
        assert!(fullscreen_toggle_shortcut_from_parts(
            PhysicalKey::Code(KeyCode::NumpadEnter),
            ElementState::Pressed,
            false,
            false,
            ModifiersState::ALT,
        ));
    }

    #[test]
    fn fullscreen_toggle_shortcut_rejects_missing_alt_repeat_release_synthetic_and_other_keys() {
        assert!(!fullscreen_toggle_shortcut_from_parts(
            PhysicalKey::Code(KeyCode::Enter),
            ElementState::Pressed,
            false,
            false,
            ModifiersState::empty(),
        ));
        assert!(!fullscreen_toggle_shortcut_from_parts(
            PhysicalKey::Code(KeyCode::Enter),
            ElementState::Pressed,
            true,
            false,
            ModifiersState::ALT,
        ));
        assert!(!fullscreen_toggle_shortcut_from_parts(
            PhysicalKey::Code(KeyCode::Enter),
            ElementState::Released,
            false,
            false,
            ModifiersState::ALT,
        ));
        assert!(!fullscreen_toggle_shortcut_from_parts(
            PhysicalKey::Code(KeyCode::Enter),
            ElementState::Pressed,
            false,
            true,
            ModifiersState::ALT,
        ));
        assert!(!fullscreen_toggle_shortcut_from_parts(
            PhysicalKey::Code(KeyCode::KeyA),
            ElementState::Pressed,
            false,
            false,
            ModifiersState::ALT,
        ));
    }

    #[test]
    fn imgui_visibility_shortcut_accepts_f7_press() {
        assert!(imgui_visibility_shortcut_from_parts(
            PhysicalKey::Code(KeyCode::F7),
            ElementState::Pressed,
            false,
            false,
        ));
    }

    #[test]
    fn imgui_visibility_shortcut_rejects_repeat_release_synthetic_and_other_keys() {
        assert!(!imgui_visibility_shortcut_from_parts(
            PhysicalKey::Code(KeyCode::F7),
            ElementState::Pressed,
            true,
            false,
        ));
        assert!(!imgui_visibility_shortcut_from_parts(
            PhysicalKey::Code(KeyCode::F7),
            ElementState::Released,
            false,
            false,
        ));
        assert!(!imgui_visibility_shortcut_from_parts(
            PhysicalKey::Code(KeyCode::F7),
            ElementState::Pressed,
            false,
            true,
        ));
        assert!(!imgui_visibility_shortcut_from_parts(
            PhysicalKey::Code(KeyCode::F6),
            ElementState::Pressed,
            false,
            false,
        ));
    }
}
