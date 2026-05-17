//! Per-redraw frame phase orchestration for the app driver.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Instant;

use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

use crate::diagnostics::crash_context::{self, RenderMode};
use crate::frontend::input::{
    apply_output_state_to_window, apply_per_frame_cursor_lock_when_locked,
};
use crate::present::present_clear_frame;
use crate::render_graph::GraphExecuteError;
use crate::xr::{HmdSubmitOutcome, OpenxrFrameTick};

use super::super::exit::ExitReason;
use super::AppDriver;

/// Sentinel used before the first frame render mode has been observed.
const UNSEEN_FRAME_RENDER_MODE: u8 = u8::MAX;

/// Last render mode logged by the app driver.
static LAST_FRAME_RENDER_MODE: AtomicU8 = AtomicU8::new(UNSEEN_FRAME_RENDER_MODE);

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
    /// HMD graph work was queued, but no projection layer was queued.
    VrRenderedWithoutProjection,
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

fn runtime_exit_reason(shutdown_requested: bool, fatal_error: bool) -> Option<ExitReason> {
    if shutdown_requested {
        Some(ExitReason::HostShutdown)
    } else if fatal_error {
        Some(ExitReason::FatalIpc)
    } else {
        None
    }
}

impl AppDriver {
    /// One winit redraw tick.
    pub(super) fn tick_frame(&mut self, event_loop: &dyn ActiveEventLoop) {
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
        event_loop: &dyn ActiveEventLoop,
        frame_start: Instant,
    ) -> FrameTickOutcome {
        self.frame_tick_prologue(frame_start);
        self.poll_ipc_and_window();
        if self.check_external_shutdown(event_loop) {
            return FrameTickOutcome::ExitRequested;
        }
        if self.handle_runtime_exit_requests(event_loop) {
            return FrameTickOutcome::ExitRequested;
        }
        if self.handle_gpu_device_loss_request(event_loop) {
            return FrameTickOutcome::ExitRequested;
        }
        self.runtime.update_decoupling_activation(Instant::now());
        {
            profiling::scope!("tick::asset_integration");
            self.runtime.run_asset_integration();
        };
        if let Some(target) = self.target.as_mut() {
            let gpu = target.gpu_mut();
            self.runtime.maintain_nonblocking_gpu_jobs(gpu);
            self.runtime.drain_reflection_probe_render_tasks(gpu);
            self.runtime.drain_camera_render_tasks(gpu);
        }

        let xr_pause = self
            .main_heartbeat
            .as_ref()
            .map(|heartbeat| heartbeat.pause());
        let xr_tick = self.xr_begin_tick();
        drop(xr_pause);

        self.lock_step_exchange();
        if self.handle_openxr_exit_request(event_loop) {
            self.queue_empty_openxr_frame_if_needed(xr_tick);
            self.poll_graceful_shutdown(event_loop);
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
        let hmd_projection_ended = render_outcome.hmd_projection_ended;
        if self.handle_gpu_device_loss_request(event_loop) {
            if !hmd_projection_ended {
                self.queue_empty_openxr_frame_if_needed(xr_tick);
            }
            self.poll_graceful_shutdown(event_loop);
            return FrameTickOutcome::ExitRequested;
        }
        self.present_and_diagnostics(xr_tick, hmd_projection_ended);
        FrameTickOutcome::Presented
    }

    fn finish_frame_tick(&mut self) {
        self.frame_tick_epilogue();
        crate::profiling::flush_resource_churn_plots();
        crate::profiling::emit_frame_mark();
    }

    fn frame_tick_prologue(&mut self, frame_start: Instant) {
        profiling::scope!("tick::prologue");
        super::tick_phase_trace("frame_tick_prologue");
        let sample = self.frame_clock.begin_frame(frame_start);
        if let Some(idle_ms) = sample.event_loop_idle_ms {
            crate::profiling::plot_event_loop_idle_ms(idle_ms);
        }
        self.xr_haptics
            .set_frame_delta_seconds((sample.wall_frame_time_ms * 0.001) as f32);
        self.runtime
            .set_debug_hud_wall_frame_time_ms(sample.wall_frame_time_ms);
        self.sync_log_level_from_settings();
        self.runtime.tick_frame_wall_clock_begin(frame_start);
        if let Some(target) = self.target.as_mut() {
            let gpu = target.gpu_mut();
            gpu.begin_frame_timing(frame_start);
            if let Ok(settings) = self.runtime.settings().read() {
                gpu.set_present_mode(settings.rendering.vsync);
            }
        }
    }

    fn poll_ipc_and_window(&mut self) {
        profiling::scope!("tick::poll_ipc_and_window");
        super::tick_phase_trace("poll_ipc_and_window");
        self.runtime.poll_ipc();

        if let Some(output_state) = self.runtime.take_pending_output_state() {
            self.apply_host_vr_haptics(&output_state);
            if let Some(target) = self.target.as_ref()
                && let Err(error) = apply_output_state_to_window(
                    target.window().as_ref(),
                    &mut self.input,
                    &output_state,
                    &mut self.cursor_output_tracking,
                )
            {
                logger::debug!("apply_output_state_to_window: {error:?}");
            }
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
                &self.cursor_output_tracking,
            ) {
                logger::trace!("apply_per_frame_cursor_lock_when_locked: {error:?}");
            }
        }
    }

    fn lock_step_exchange(&mut self) {
        profiling::scope!("tick::lock_step_exchange");
        super::tick_phase_trace("lock_step_exchange");
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

    fn handle_runtime_exit_requests(&mut self, event_loop: &dyn ActiveEventLoop) -> bool {
        let Some(reason) = runtime_exit_reason(
            self.runtime.shutdown_requested(),
            self.runtime.fatal_error(),
        ) else {
            return false;
        };

        match reason {
            ExitReason::HostShutdown => {
                logger::info!("Renderer shutdown requested by host");
                self.runtime
                    .log_compact_renderer_summary("host-shutdown-requested");
            }
            ExitReason::FatalIpc => {
                logger::error!("Renderer fatal IPC error");
                self.runtime.log_compact_renderer_summary("fatal-ipc");
            }
            _ => {}
        }

        self.request_exit(reason, event_loop);
        true
    }

    /// Requests renderer shutdown when wgpu reports that the active device is lost.
    fn handle_gpu_device_loss_request(&mut self, event_loop: &dyn ActiveEventLoop) -> bool {
        let Some(generation) = self
            .target
            .as_mut()
            .and_then(|target| target.gpu_mut().take_device_lost())
        else {
            return false;
        };

        logger::error!("GPU device lost; shutting down renderer: generation={generation}");
        if let Some(target) = self.target.as_ref() {
            target.gpu().record_device_loss_observed(generation);
            target
                .gpu()
                .dump_gpu_flight_recorder_once("gpu-device-lost");
        }
        self.runtime.log_compact_renderer_summary("gpu-device-lost");
        self.request_exit(ExitReason::GpuDeviceLost, event_loop);
        true
    }

    fn handle_openxr_exit_request(&mut self, event_loop: &dyn ActiveEventLoop) -> bool {
        if let Some(target) = self.target.as_ref()
            && let Some(session) = target.xr_session()
            && session.handles.xr_session.exit_requested()
        {
            logger::info!("OpenXR requested exit");
            self.request_exit(ExitReason::OpenxrExit, event_loop);
            return true;
        }

        false
    }

    fn render_views(
        &mut self,
        window: &Arc<dyn Window>,
        xr_tick: Option<&OpenxrFrameTick>,
    ) -> Option<RenderViewsOutcome> {
        profiling::scope!("tick::render_views");
        super::tick_phase_trace("render_views");
        if let Some(target) = self.target.as_mut() {
            self.runtime.drain_hi_z_readback(target.gpu_mut());
        }

        let hmd_submit_outcome = self.try_hmd_multiview_submit(xr_tick);
        let hmd_projection_ended = hmd_submit_outcome.projection_queued();
        let mode = if hmd_projection_ended {
            FrameRenderMode::HmdMultiview
        } else if matches!(
            hmd_submit_outcome,
            HmdSubmitOutcome::RenderedWithoutProjection
        ) {
            FrameRenderMode::VrRenderedWithoutProjection
        } else if self.runtime.vr_active() {
            FrameRenderMode::VrSecondaryOnly
        } else {
            FrameRenderMode::Desktop
        };
        crash_context::set_render_mode(match mode {
            FrameRenderMode::HmdMultiview => RenderMode::HmdMultiview,
            FrameRenderMode::VrRenderedWithoutProjection => RenderMode::HmdMultiview,
            FrameRenderMode::VrSecondaryOnly => RenderMode::VrSecondariesOnly,
            FrameRenderMode::Desktop => RenderMode::IpcDesktop,
        });
        log_frame_render_mode_transition(mode, hmd_submit_outcome, self.runtime.vr_active());
        logger::trace!(
            "frame render mode: {:?} hmd_submit_outcome={:?} vr_active={}",
            mode,
            hmd_submit_outcome,
            self.runtime.vr_active(),
        );

        if hmd_submit_outcome.should_render_non_hmd_views() {
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

    fn render_non_hmd_views(&mut self, mode: FrameRenderMode) -> Option<()> {
        let target = self.target.as_mut()?;
        use crate::xr::XrFrameRenderer;
        let desktop_owned_by_blit = matches!(mode, FrameRenderMode::Desktop)
            && self
                .runtime
                .scene()
                .desktop_blit_for_display(super::DESKTOP_DISPLAY_INDEX)
                .is_some();
        let result = match mode {
            FrameRenderMode::HmdMultiview => Ok(()),
            FrameRenderMode::VrRenderedWithoutProjection => Ok(()),
            FrameRenderMode::VrSecondaryOnly => {
                self.runtime.submit_secondary_only(target.gpu_mut())
            }
            FrameRenderMode::Desktop if desktop_owned_by_blit => {
                // The desktop window will be filled by the present-time `BlitToDisplay` pass;
                // skip the world-camera swapchain output and only run secondary RTs (which the
                // dash camera, mirror previews, and avatar previews still depend on).
                self.runtime.submit_secondary_only(target.gpu_mut())
            }
            FrameRenderMode::Desktop => self.runtime.render_desktop_frame(target.gpu_mut()),
        };
        if let Err(error) = result {
            let kind = crash_context::graph_error_kind(&error);
            crash_context::set_last_graph_error(kind);
            self.handle_frame_graph_error(error);
        }
        Some(())
    }

    /// Reacts to a per-frame [`GraphExecuteError`]: when the graph itself is missing, drive a
    /// clear-only present so the swapchain still progresses; otherwise log and reconfigure.
    fn handle_frame_graph_error(&mut self, error: GraphExecuteError) {
        let Some(target) = self.target.as_mut() else {
            return;
        };
        if matches!(error, GraphExecuteError::NoFrameGraph) {
            if let Err(present_error) = present_clear_frame(target.gpu_mut()) {
                logger::warn!("present fallback failed: {present_error:?}");
                target.reconfigure_for_window();
            }
        } else {
            logger::warn!("frame graph failed: {error:?}");
            target.reconfigure_for_window();
        }
    }

    fn frame_tick_epilogue(&mut self) {
        profiling::scope!("tick::epilogue");
        super::tick_phase_trace("frame_tick_epilogue");
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
            gpu.dump_gpu_flight_recorder_once("driver-thread-error");
        }
        // Cheap (two atomic loads); plotted alongside `event_loop_idle_ms` so a regression
        // in driver-thread pipelining is visible in the same Tracy trace as a regression in
        // frame timing.
        let backlog = gpu.driver_submit_backlog();
        crash_context::set_driver_backlog(backlog);
        crate::profiling::plot_driver_submit_backlog(backlog);
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

/// Logs first-observed and changed render modes without emitting per-frame debug noise.
fn log_frame_render_mode_transition(
    mode: FrameRenderMode,
    hmd_submit_outcome: HmdSubmitOutcome,
    vr_active: bool,
) {
    let code = frame_render_mode_code(mode);
    let previous = LAST_FRAME_RENDER_MODE.swap(code, Ordering::Relaxed);
    if previous == code {
        return;
    }
    logger::debug!(
        "frame render mode selected: {} previous={} hmd_submit_outcome={:?} vr_active={}",
        frame_render_mode_label(mode),
        frame_render_mode_code_label(previous),
        hmd_submit_outcome,
        vr_active,
    );
}

/// Stable compact code for one frame render mode.
fn frame_render_mode_code(mode: FrameRenderMode) -> u8 {
    match mode {
        FrameRenderMode::HmdMultiview => 0,
        FrameRenderMode::VrRenderedWithoutProjection => 1,
        FrameRenderMode::VrSecondaryOnly => 2,
        FrameRenderMode::Desktop => 3,
    }
}

/// Human-readable label for one frame render mode.
fn frame_render_mode_label(mode: FrameRenderMode) -> &'static str {
    match mode {
        FrameRenderMode::HmdMultiview => "hmd-multiview",
        FrameRenderMode::VrRenderedWithoutProjection => "vr-rendered-without-projection",
        FrameRenderMode::VrSecondaryOnly => "vr-secondaries-only",
        FrameRenderMode::Desktop => "desktop",
    }
}

/// Human-readable label for one compact frame render mode code.
fn frame_render_mode_code_label(code: u8) -> &'static str {
    match code {
        0 => "hmd-multiview",
        1 => "vr-rendered-without-projection",
        2 => "vr-secondaries-only",
        3 => "desktop",
        UNSEEN_FRAME_RENDER_MODE => "unseen",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::{
        FrameRenderMode, RenderViewsOutcome, frame_render_mode_code, frame_render_mode_code_label,
        frame_render_mode_label, runtime_exit_reason,
    };
    use crate::app::exit::ExitReason;
    use crate::xr::HmdSubmitOutcome;

    #[test]
    fn render_views_outcome_records_hmd_projection() {
        let outcome = RenderViewsOutcome {
            mode: FrameRenderMode::HmdMultiview,
            hmd_projection_ended: true,
        };
        assert!(outcome.hmd_projection_ended);
        assert_eq!(outcome.mode, FrameRenderMode::HmdMultiview);
    }

    #[test]
    fn runtime_exit_reason_prefers_host_shutdown_over_fatal_ipc() {
        assert_eq!(
            runtime_exit_reason(true, true),
            Some(ExitReason::HostShutdown)
        );
    }

    #[test]
    fn runtime_exit_reason_maps_fatal_ipc_when_shutdown_is_absent() {
        assert_eq!(runtime_exit_reason(false, true), Some(ExitReason::FatalIpc));
    }

    #[test]
    fn runtime_exit_reason_ignores_running_runtime() {
        assert_eq!(runtime_exit_reason(false, false), None);
    }

    #[test]
    fn frame_render_mode_codes_have_stable_labels() {
        assert_eq!(frame_render_mode_code(FrameRenderMode::HmdMultiview), 0);
        assert_eq!(
            frame_render_mode_code(FrameRenderMode::VrRenderedWithoutProjection),
            1
        );
        assert_eq!(
            frame_render_mode_label(FrameRenderMode::VrSecondaryOnly),
            "vr-secondaries-only"
        );
        assert_eq!(frame_render_mode_code_label(3), "desktop");
    }

    #[test]
    fn hmd_submit_outcome_controls_non_hmd_rerendering() {
        assert!(HmdSubmitOutcome::SkippedBeforeRender.should_render_non_hmd_views());
        assert!(!HmdSubmitOutcome::RenderedWithoutProjection.should_render_non_hmd_views());
        assert!(!HmdSubmitOutcome::ProjectionQueued.should_render_non_hmd_views());
    }

    #[test]
    fn hmd_submit_outcome_identifies_projection_finalize() {
        assert!(HmdSubmitOutcome::ProjectionQueued.projection_queued());
        assert!(!HmdSubmitOutcome::SkippedBeforeRender.projection_queued());
        assert!(!HmdSubmitOutcome::RenderedWithoutProjection.projection_queued());
    }
}
