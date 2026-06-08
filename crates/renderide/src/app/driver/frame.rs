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
use crate::frontend::{HostWaitReason, LockstepPipelineAction};
use crate::present::present_clear_frame;
use crate::render_graph::GraphExecuteError;
use crate::xr::{HmdSubmitOutcome, OpenxrFrameTick};

use super::super::exit::ExitReason;
use super::super::window_icon::apply_host_window_icon;
use super::AppDriver;

/// Sentinel used before the first frame render mode has been observed.
const UNSEEN_FRAME_RENDER_MODE: u8 = u8::MAX;

/// Last render mode logged by the app driver.
static LAST_FRAME_RENDER_MODE: AtomicU8 = AtomicU8::new(UNSEEN_FRAME_RENDER_MODE);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FrameTickOutcome {
    /// A world render was attempted and presentation/diagnostics ran.
    Presented,
    /// Rendering was intentionally skipped while coupled lockstep waits for the host.
    RenderSkipped,
    /// The app requested event-loop exit during this tick.
    ExitRequested,
    /// No render target was available for this tick.
    MissingTarget,
}

impl FrameTickOutcome {
    /// Whether this outcome should publish visible frame timing and HUD samples.
    const fn records_frame_timing(self) -> bool {
        !matches!(self, Self::RenderSkipped)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PreXrLockstepAction {
    /// Continue toward OpenXR frame begin.
    Continue,
    /// Wait for an already requested host submit before opening an OpenXR frame.
    WaitForSubmit,
    /// Send the next begin-frame, then wait before opening an OpenXR frame.
    SendBeginThenWait,
    /// No render or host request is currently possible; skip without opening an OpenXR frame.
    SkipUntilHostReady,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PreXrLockstepInput {
    vr_active: bool,
    awaiting_frame_submit: bool,
    should_render_frame: bool,
    should_send_begin_frame: bool,
}

fn pre_xr_lockstep_action(input: PreXrLockstepInput) -> PreXrLockstepAction {
    if input.should_render_frame {
        return PreXrLockstepAction::Continue;
    }
    if input.awaiting_frame_submit {
        return PreXrLockstepAction::WaitForSubmit;
    }
    if !input.vr_active {
        return PreXrLockstepAction::Continue;
    }
    if input.should_send_begin_frame {
        PreXrLockstepAction::SendBeginThenWait
    } else {
        PreXrLockstepAction::SkipUntilHostReady
    }
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

fn desktop_frame_owned_by_explicit_blit(
    mode: FrameRenderMode,
    explicit_desktop_blit_active: bool,
) -> bool {
    matches!(mode, FrameRenderMode::Desktop) && explicit_desktop_blit_active
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
        {
            profiling::scope!("tick::frame");
            let frame_start = Instant::now();
            if let Some(heartbeat) = self.main_heartbeat.as_ref() {
                heartbeat.pet();
            }
            let outcome = self.drive_frame_phases(event_loop, frame_start);
            self.finish_frame_tick(outcome);
        }
        crate::profiling::emit_frame_mark();
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
        let mut one_credit_begin_sent = self.drain_completion_and_try_desktop_one_credit();
        if self.runtime.should_send_begin_frame_before_wait_work() {
            self.lock_step_exchange();
        }
        self.run_asset_integration_phase();
        let mut vr_active = self.runtime.vr_active();
        let pre_xr_action = pre_xr_lockstep_action(PreXrLockstepInput {
            vr_active,
            awaiting_frame_submit: self.runtime.awaiting_frame_submit(),
            should_render_frame: self.runtime.should_render_frame(),
            should_send_begin_frame: self.runtime.should_send_begin_frame(),
        });
        match pre_xr_action {
            PreXrLockstepAction::Continue => {}
            PreXrLockstepAction::WaitForSubmit => {
                if let Some(outcome) = self.wait_for_host_submit_before_xr(event_loop) {
                    return outcome;
                }
            }
            PreXrLockstepAction::SendBeginThenWait => {
                self.lock_step_exchange();
                if let Some(outcome) = self.wait_for_host_submit_before_xr(event_loop) {
                    return outcome;
                }
            }
            PreXrLockstepAction::SkipUntilHostReady => return FrameTickOutcome::RenderSkipped,
        }
        vr_active = self.runtime.vr_active();

        let xr_pause = self
            .main_heartbeat
            .as_ref()
            .map(|heartbeat| heartbeat.pause());
        let xr_tick = self.xr_begin_tick();
        drop(xr_pause);

        if !vr_active {
            self.lock_step_exchange();
        }
        if self.handle_openxr_exit_request(event_loop) {
            self.queue_empty_openxr_frame_if_needed(xr_tick);
            self.poll_graceful_shutdown(event_loop);
            return FrameTickOutcome::ExitRequested;
        }
        if vr_active && xr_tick.is_some() && self.runtime.should_send_one_credit_begin_frame() {
            one_credit_begin_sent = self.one_credit_lock_step_exchange() || one_credit_begin_sent;
        }
        if !self.runtime.should_render_frame() {
            if !vr_active {
                self.runtime
                    .wait_for_coupled_submit_or_decoupling(HostWaitReason::DesktopAwaitingSubmit);
                if self.handle_runtime_exit_requests(event_loop) {
                    self.queue_empty_openxr_frame_if_needed(xr_tick);
                    return FrameTickOutcome::ExitRequested;
                }
            }
            if !self.runtime.should_render_frame() {
                self.consume_unrendered_one_credit_submit(one_credit_begin_sent);
                self.queue_empty_openxr_frame_if_needed(xr_tick);
                return FrameTickOutcome::RenderSkipped;
            }
        }

        let Some(window) = self
            .target
            .as_ref()
            .map(|target| Arc::clone(target.window()))
        else {
            self.consume_unrendered_one_credit_submit(one_credit_begin_sent);
            return FrameTickOutcome::MissingTarget;
        };
        let Some(render_outcome) = self.render_views(&window, xr_tick.as_ref()) else {
            self.consume_unrendered_one_credit_submit(one_credit_begin_sent);
            return FrameTickOutcome::MissingTarget;
        };
        self.runtime.note_frame_render_attempted();
        let hmd_projection_ended = render_outcome.hmd_projection_ended;
        if self.handle_gpu_device_loss_request(event_loop) {
            if !hmd_projection_ended {
                self.queue_empty_openxr_frame_if_needed(xr_tick);
            }
            self.poll_graceful_shutdown(event_loop);
            return FrameTickOutcome::ExitRequested;
        }
        self.present_and_diagnostics(xr_tick, hmd_projection_ended);
        self.drain_submit_completion_work();
        if vr_active || !one_credit_begin_sent {
            self.lock_step_exchange();
        }
        FrameTickOutcome::Presented
    }

    fn drain_completion_and_try_desktop_one_credit(&mut self) -> bool {
        self.runtime.update_decoupling_activation(Instant::now());
        self.drain_submit_completion_work();
        let action = self.runtime.record_lockstep_pipeline_decision();
        if !self.runtime.vr_active()
            && self.target.is_some()
            && action == LockstepPipelineAction::SendEarlyNextFrame
        {
            self.one_credit_lock_step_exchange()
        } else {
            false
        }
    }

    fn drain_submit_completion_work(&mut self) {
        if let Some(target) = self.target.as_mut() {
            let gpu = target.gpu_mut();
            self.runtime.maintain_nonblocking_gpu_jobs(gpu);
            self.runtime.drain_reflection_probe_render_tasks(gpu);
            self.runtime.drain_camera_render_tasks(gpu);
        }
    }

    fn run_asset_integration_phase(&mut self) {
        self.runtime.update_decoupling_activation(Instant::now());
        {
            profiling::scope!("tick::asset_integration");
            self.runtime.run_asset_integration();
        };
    }

    fn finish_frame_tick(&mut self, outcome: FrameTickOutcome) {
        self.frame_tick_epilogue(outcome);
        crate::profiling::flush_resource_churn_plots();
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
        self.apply_pending_window_icons();

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

    pub(super) fn apply_pending_window_icons(&mut self) {
        let Some(window) = self
            .target
            .as_ref()
            .map(|target| Arc::clone(target.window()))
        else {
            return;
        };
        let mut requests = self.runtime.take_pending_window_icon_requests();
        while let Some(request) = requests.pop_front() {
            let success = if request.is_overlay {
                logger::warn!(
                    "runtime: taskbar overlay window icons are unsupported request_id={}",
                    request.request_id
                );
                false
            } else {
                self.apply_main_window_icon_request(window.as_ref(), &request)
            };
            self.runtime
                .send_window_icon_result(request.request_id, success);
        }
    }

    fn apply_main_window_icon_request(
        &mut self,
        window: &dyn Window,
        request: &crate::shared::SetWindowIcon,
    ) -> bool {
        let Some(bgra) = self.runtime.load_window_icon_bgra(request) else {
            return false;
        };
        match apply_host_window_icon(window, request.size, &bgra) {
            Ok(()) => {
                logger::info!(
                    "runtime: applied host window icon request_id={} size={}x{}",
                    request.request_id,
                    request.size.x,
                    request.size.y
                );
                true
            }
            Err(error) => {
                logger::warn!(
                    "runtime: failed to apply host window icon request_id={}: {error}",
                    request.request_id
                );
                false
            }
        }
    }

    fn wait_for_host_submit_before_xr(
        &mut self,
        event_loop: &dyn ActiveEventLoop,
    ) -> Option<FrameTickOutcome> {
        self.runtime
            .wait_for_coupled_submit_or_decoupling(HostWaitReason::XrBeforeFrame);
        if self.handle_runtime_exit_requests(event_loop) {
            return Some(FrameTickOutcome::ExitRequested);
        }
        if self.runtime.should_render_frame() {
            None
        } else {
            Some(FrameTickOutcome::RenderSkipped)
        }
    }

    fn build_lock_step_inputs(&mut self) -> crate::shared::InputState {
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
        inputs
    }

    fn lock_step_exchange(&mut self) -> bool {
        profiling::scope!("tick::lock_step_exchange");
        super::tick_phase_trace("lock_step_exchange");
        self.runtime.record_lockstep_pipeline_decision();
        if self.runtime.should_send_begin_frame() {
            let inputs = self.build_lock_step_inputs();
            self.runtime.pre_frame(inputs)
        } else {
            profiling::scope!("lock_step::skipped");
            false
        }
    }

    fn one_credit_lock_step_exchange(&mut self) -> bool {
        profiling::scope!("tick::one_credit_lock_step_exchange");
        super::tick_phase_trace("one_credit_lock_step_exchange");
        if self.runtime.record_lockstep_pipeline_decision()
            == LockstepPipelineAction::SendEarlyNextFrame
        {
            let inputs = self.build_lock_step_inputs();
            self.runtime.pre_frame_one_credit(inputs)
        } else {
            profiling::scope!("lock_step::one_credit_skipped");
            false
        }
    }

    fn consume_unrendered_one_credit_submit(&mut self, one_credit_begin_sent: bool) {
        if one_credit_begin_sent {
            logger::warn!(
                "one-credit host frame request sent but render was skipped; marking applied submit consumed before the next IPC poll"
            );
            self.runtime.note_frame_render_attempted();
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
        if let Some(target) = self.target.as_mut() {
            target.gpu_mut().abandon_surface_after_device_loss();
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
        let desktop_owned_by_blit = desktop_frame_owned_by_explicit_blit(
            mode,
            self.runtime
                .scene()
                .active_blit_for_display(super::DESKTOP_DISPLAY_INDEX)
                .is_some(),
        );
        let result = match mode {
            FrameRenderMode::HmdMultiview => Ok(()),
            FrameRenderMode::VrRenderedWithoutProjection => Ok(()),
            FrameRenderMode::VrSecondaryOnly => {
                self.runtime.submit_vr_secondaries_only(target.gpu_mut())
            }
            FrameRenderMode::Desktop if desktop_owned_by_blit => {
                // An explicit display blit owns the desktop window this tick; skip the
                // world-camera swapchain output and only run secondary RTs.
                self.runtime
                    .render_desktop_secondaries_frame(target.gpu_mut())
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

    fn frame_tick_epilogue(&mut self, outcome: FrameTickOutcome) {
        profiling::scope!("tick::epilogue");
        super::tick_phase_trace("frame_tick_epilogue");
        self.drain_driver_thread_error();
        if outcome.records_frame_timing() {
            self.end_frame_timing_and_hud_capture();
        }
        let gpu_render_time_seconds = self
            .target
            .as_ref()
            .and_then(|target| target.gpu().last_completed_gpu_render_time_seconds());
        self.runtime
            .tick_frame_render_time_end(gpu_render_time_seconds);
        if outcome != FrameTickOutcome::RenderSkipped {
            self.runtime.note_render_tick_complete();
        }
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
        let excluded_wait = self.runtime.drain_frame_timing_excluded_wait();
        let Some(target) = self.target.as_mut() else {
            return;
        };
        let gpu = target.gpu_mut();
        // Capture the main-thread CPU duration just before finalizing the frame's timing --
        // every per-frame submit has been dispatched by now, but the event loop has not yet
        // yielded. Explicit display/compositor waits are subtracted so the HUD CPU value
        // represents active renderer work rather than frame pacing.
        gpu.record_frame_timing_excluded_wait(excluded_wait);
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
        FrameRenderMode, FrameTickOutcome, PreXrLockstepAction, PreXrLockstepInput,
        RenderViewsOutcome, desktop_frame_owned_by_explicit_blit, frame_render_mode_code,
        frame_render_mode_code_label, frame_render_mode_label, pre_xr_lockstep_action,
        runtime_exit_reason,
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
    fn desktop_explicit_blit_uses_secondaries_only_schedule() {
        assert!(desktop_frame_owned_by_explicit_blit(
            FrameRenderMode::Desktop,
            true
        ));
        assert!(!desktop_frame_owned_by_explicit_blit(
            FrameRenderMode::Desktop,
            false
        ));
        assert!(!desktop_frame_owned_by_explicit_blit(
            FrameRenderMode::VrSecondaryOnly,
            true
        ));
    }

    #[test]
    fn render_skipped_ticks_do_not_record_visible_frame_timing() {
        assert!(!FrameTickOutcome::RenderSkipped.records_frame_timing());
        assert!(FrameTickOutcome::Presented.records_frame_timing());
        assert!(FrameTickOutcome::ExitRequested.records_frame_timing());
    }

    #[test]
    fn vr_awaiting_submit_waits_before_openxr_begin() {
        assert_eq!(
            pre_xr_lockstep_action(PreXrLockstepInput {
                vr_active: true,
                awaiting_frame_submit: true,
                should_render_frame: false,
                should_send_begin_frame: false,
            }),
            PreXrLockstepAction::WaitForSubmit
        );
    }

    #[test]
    fn vr_idle_lockstep_sends_begin_before_openxr_begin() {
        assert_eq!(
            pre_xr_lockstep_action(PreXrLockstepInput {
                vr_active: true,
                awaiting_frame_submit: false,
                should_render_frame: false,
                should_send_begin_frame: true,
            }),
            PreXrLockstepAction::SendBeginThenWait
        );
    }

    #[test]
    fn vr_renderable_frame_reaches_openxr_before_next_begin() {
        assert_eq!(
            pre_xr_lockstep_action(PreXrLockstepInput {
                vr_active: true,
                awaiting_frame_submit: false,
                should_render_frame: true,
                should_send_begin_frame: true,
            }),
            PreXrLockstepAction::Continue
        );
    }

    #[test]
    fn vr_nonrenderable_without_begin_skips_before_openxr_begin() {
        assert_eq!(
            pre_xr_lockstep_action(PreXrLockstepInput {
                vr_active: true,
                awaiting_frame_submit: false,
                should_render_frame: false,
                should_send_begin_frame: false,
            }),
            PreXrLockstepAction::SkipUntilHostReady
        );
    }

    #[test]
    fn desktop_idle_lockstep_reaches_non_vr_begin_path() {
        assert_eq!(
            pre_xr_lockstep_action(PreXrLockstepInput {
                vr_active: false,
                awaiting_frame_submit: false,
                should_render_frame: false,
                should_send_begin_frame: true,
            }),
            PreXrLockstepAction::Continue
        );
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
