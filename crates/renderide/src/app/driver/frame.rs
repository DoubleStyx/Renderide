//! Per-redraw frame phase orchestration for the app driver.

use std::sync::Arc;
use std::time::Instant;

use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

use crate::frontend::input::{
    apply_output_state_to_window, apply_per_frame_cursor_lock_when_locked,
};
use crate::xr::OpenxrFrameTick;

use super::super::exit::ExitReason;
use super::AppDriver;

/// Prefix for per-phase trace lines in the app frame tick.
const TICK_TRACE_PREFIX: &str = "renderide::tick";

/// Emits a trace line naming the current frame phase.
pub(super) fn tick_phase_trace(phase: &'static str) {
    logger::trace!("{} phase={phase}", TICK_TRACE_PREFIX);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FrameTickOutcome {
    Presented,
    ExitRequested,
    MissingTarget,
}

/// Render path used for the current frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum FrameRenderMode {
    /// HMD multiview path had a projection layer.
    HmdMultiview,
    /// VR frame without an HMD projection layer; render secondary cameras only.
    VrSecondaryOnly,
    /// Ordinary desktop world render.
    Desktop,
}

/// Result of rendering this tick's planned views.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct RenderViewsOutcome {
    /// Selected render path.
    pub(super) mode: FrameRenderMode,
    /// Whether an OpenXR projection layer was submitted.
    pub(super) hmd_projection_ended: bool,
}

impl AppDriver {
    /// One winit redraw tick.
    pub(super) fn tick_frame(&mut self, event_loop: &ActiveEventLoop) {
        profiling::scope!("tick::frame");
        let frame_start = Instant::now();
        if let Some(heartbeat) = self.main_heartbeat.as_ref() {
            heartbeat.pet();
        }
        let _outcome = self.drive_frame_phases(event_loop, frame_start);
        self.finish_frame_tick();
    }

    fn drive_frame_phases(
        &mut self,
        event_loop: &ActiveEventLoop,
        frame_start: Instant,
    ) -> FrameTickOutcome {
        self.frame_tick_prologue(frame_start);
        self.poll_ipc_and_window();
        if self.check_external_shutdown(event_loop) {
            return FrameTickOutcome::ExitRequested;
        }
        self.runtime.update_decoupling_activation(Instant::now());
        {
            profiling::scope!("tick::asset_integration");
            self.runtime.run_asset_integration();
        };
        if let Some(target) = self.target.as_mut() {
            self.runtime.maintain_nonblocking_gpu_jobs(target.gpu_mut());
        }

        let xr_pause = self
            .main_heartbeat
            .as_ref()
            .map(|heartbeat| heartbeat.pause());
        let xr_tick = self.xr_begin_tick();
        drop(xr_pause);

        self.lock_step_exchange();
        if self.handle_frame_exit_requests(event_loop) {
            return FrameTickOutcome::ExitRequested;
        }

        let Some(window) = self
            .target
            .as_ref()
            .map(|target| Arc::clone(target.window()))
        else {
            return FrameTickOutcome::MissingTarget;
        };
        let Some(render_outcome) = self.render_views(&window, xr_tick.as_ref()) else {
            return FrameTickOutcome::MissingTarget;
        };
        self.present_and_diagnostics(xr_tick, render_outcome.hmd_projection_ended);
        FrameTickOutcome::Presented
    }

    fn finish_frame_tick(&mut self) {
        self.frame_tick_epilogue();
        crate::profiling::emit_frame_mark();
    }

    fn frame_tick_prologue(&mut self, frame_start: Instant) {
        profiling::scope!("tick::prologue");
        tick_phase_trace("frame_tick_prologue");
        let sample = self.frame_clock.begin_frame(frame_start);
        if let Some(idle_ms) = sample.event_loop_idle_ms {
            crate::profiling::plot_event_loop_idle_ms(idle_ms);
        }
        self.runtime
            .set_debug_hud_wall_frame_time_ms(sample.wall_frame_time_ms);
        self.sync_log_level_from_settings();
        self.runtime.tick_frame_wall_clock_begin(frame_start);
        if let Some(target) = self.target.as_mut() {
            let gpu = target.gpu_mut();
            gpu.begin_frame_timing(frame_start);
            if let Ok(settings) = self.runtime.settings().read() {
                gpu.set_present_mode(settings.rendering.vsync);
                gpu.set_max_frame_latency(settings.rendering.resolved_max_frame_latency());
            }
        }
    }

    fn poll_ipc_and_window(&mut self) {
        profiling::scope!("tick::poll_ipc_and_window");
        tick_phase_trace("poll_ipc_and_window");
        self.runtime.poll_ipc();

        if let (Some(target), Some(output_state)) = (
            self.target.as_ref(),
            self.runtime.take_pending_output_state(),
        ) && let Err(error) = apply_output_state_to_window(
            target.window().as_ref(),
            &output_state,
            &mut self.cursor_output_tracking,
        ) {
            logger::debug!("apply_output_state_to_window: {error:?}");
        }

        if let Some(target) = self.target.as_ref()
            && self.runtime.host_cursor_lock_requested()
        {
            let lock_pos = self
                .runtime
                .last_output_state()
                .and_then(|state| state.lock_cursor_position);
            if let Err(error) = apply_per_frame_cursor_lock_when_locked(
                target.window().as_ref(),
                &mut self.input,
                lock_pos,
            ) {
                logger::trace!("apply_per_frame_cursor_lock_when_locked: {error:?}");
            }
        }
    }

    fn lock_step_exchange(&mut self) {
        profiling::scope!("tick::lock_step_exchange");
        tick_phase_trace("lock_step_exchange");
        if self.runtime.should_send_begin_frame() {
            let lock = self.runtime.host_cursor_lock_requested();
            let mut inputs = self.input.take_input_state(lock);
            crate::diagnostics::sanitize_input_state_for_imgui_host(
                &mut inputs,
                self.runtime.debug_hud_last_want_capture_mouse(),
                self.runtime.debug_hud_last_want_capture_keyboard(),
            );
            let output_device = self
                .target
                .as_ref()
                .map_or(crate::shared::HeadOutputDevice::Screen, |target| {
                    target.output_device()
                });
            if let Some(vr) = self.xr_input_cache.build_vr_input(output_device) {
                inputs.vr = Some(vr);
            }
            self.runtime.pre_frame(inputs);
        } else {
            profiling::scope!("lock_step::skipped");
        }
    }

    fn handle_frame_exit_requests(&mut self, event_loop: &ActiveEventLoop) -> bool {
        if let Some(target) = self.target.as_ref()
            && let Some(session) = target.xr_session()
            && session.handles.xr_session.exit_requested()
        {
            logger::info!("OpenXR requested exit");
            self.request_exit(ExitReason::OpenxrExit, event_loop);
            return true;
        }

        if self.runtime.shutdown_requested() {
            logger::info!("Renderer shutdown requested by host");
            self.request_exit(ExitReason::HostShutdown, event_loop);
            return true;
        }

        if self.runtime.fatal_error() {
            logger::error!("Renderer fatal IPC error");
            self.request_exit(ExitReason::FatalIpc, event_loop);
            return true;
        }

        false
    }

    fn render_views(
        &mut self,
        window: &Arc<Window>,
        xr_tick: Option<&OpenxrFrameTick>,
    ) -> Option<RenderViewsOutcome> {
        profiling::scope!("tick::render_views");
        tick_phase_trace("render_views");
        if let Some(target) = self.target.as_mut() {
            self.runtime.drain_hi_z_readback(target.gpu().device());
        }

        let hmd_projection_ended = self.try_hmd_multiview_submit(xr_tick);
        let mode = if hmd_projection_ended {
            FrameRenderMode::HmdMultiview
        } else if self.runtime.vr_active() {
            FrameRenderMode::VrSecondaryOnly
        } else {
            FrameRenderMode::Desktop
        };
        logger::trace!(
            "frame render mode: {:?} hmd_projection_ended={} vr_active={}",
            mode,
            hmd_projection_ended,
            self.runtime.vr_active(),
        );

        if !hmd_projection_ended {
            self.render_non_hmd_views(mode)?;
        }

        let hud_in =
            crate::diagnostics::DebugHudInput::from_winit(window.as_ref(), &mut self.input);
        self.runtime.set_debug_hud_input(hud_in);

        Some(RenderViewsOutcome {
            mode,
            hmd_projection_ended,
        })
    }

    fn try_hmd_multiview_submit(&mut self, xr_tick: Option<&OpenxrFrameTick>) -> bool {
        let Some(tick) = xr_tick else {
            return false;
        };
        let Some(target) = self.target.as_mut() else {
            return false;
        };
        let Some((gpu, session)) = target.openxr_parts_mut() else {
            return false;
        };
        profiling::scope!("xr::hmd_multiview_submit");
        crate::xr::try_openxr_hmd_multiview_submit(gpu, session, &mut self.runtime, tick)
    }

    fn render_non_hmd_views(&mut self, mode: FrameRenderMode) -> Option<()> {
        let target = self.target.as_mut()?;
        use crate::xr::XrFrameRenderer;
        let result = match mode {
            FrameRenderMode::HmdMultiview => Ok(()),
            FrameRenderMode::VrSecondaryOnly => {
                self.runtime.submit_secondary_only(target.gpu_mut())
            }
            FrameRenderMode::Desktop => self.runtime.render_desktop_frame(target.gpu_mut()),
        };
        if let Err(error) = result {
            self.handle_frame_graph_error(error);
        }
        Some(())
    }

    fn frame_tick_epilogue(&mut self) {
        profiling::scope!("tick::epilogue");
        tick_phase_trace("frame_tick_epilogue");
        self.drain_driver_thread_error();
        self.end_frame_timing_and_hud_capture();
        let gpu_render_time_seconds = self
            .target
            .as_ref()
            .and_then(|target| target.gpu().last_completed_gpu_render_time_seconds());
        self.runtime
            .tick_frame_render_time_end(gpu_render_time_seconds);
        self.runtime.note_render_tick_complete();
        self.frame_clock.end_tick(Instant::now());
    }

    fn drain_driver_thread_error(&self) {
        let Some(target) = self.target.as_ref() else {
            return;
        };
        let gpu = target.gpu();
        if let Some(err) = gpu.take_driver_error() {
            logger::error!("{err}");
        }
        // Cheap (two atomic loads); plotted alongside `event_loop_idle_ms` so a regression
        // in driver-thread pipelining is visible in the same Tracy trace as a regression in
        // frame timing.
        crate::profiling::plot_driver_submit_backlog(gpu.driver_submit_backlog());
    }

    fn end_frame_timing_and_hud_capture(&mut self) {
        let Some(target) = self.target.as_mut() else {
            return;
        };
        let gpu = target.gpu_mut();
        // Capture the main-thread CPU duration just before finalizing the frame's timing --
        // every per-frame submit has been dispatched by now, but the event loop has not yet
        // yielded, so this represents the time the main thread spent on this tick.
        gpu.record_main_thread_cpu_end(Instant::now());
        gpu.end_frame_timing();
        gpu.end_gpu_profiler_frame();
        self.runtime.capture_debug_hud_after_frame_end(gpu);
    }
}

#[cfg(test)]
mod tests {
    use super::{FrameRenderMode, RenderViewsOutcome};

    #[test]
    fn render_views_outcome_records_hmd_projection() {
        let outcome = RenderViewsOutcome {
            mode: FrameRenderMode::HmdMultiview,
            hmd_projection_ended: true,
        };
        assert!(outcome.hmd_projection_ended);
        assert_eq!(outcome.mode, FrameRenderMode::HmdMultiview);
    }
}
