//! Typed frame phase helpers for [`super::AppDriver::drive_frame_phases`].

use std::sync::Arc;
use std::time::Instant;

use winit::event_loop::ActiveEventLoop;

use crate::app::driver::AppDriver;
use crate::frontend::HostWaitReason;

use super::{
    FrameStartupState, FrameTickOutcome, PreXrLockstepInput, XrFrameBeginState,
    pre_xr_lockstep_action,
};

impl AppDriver {
    pub(super) fn run_frame_startup_phase(
        &mut self,
        event_loop: &dyn ActiveEventLoop,
        frame_start: Instant,
    ) -> Result<FrameStartupState, FrameTickOutcome> {
        self.hmd_compositor_paced_last_frame = false;
        self.frame_tick_prologue(frame_start);
        self.poll_ipc_and_window();
        if self.check_external_shutdown(event_loop)
            || self.handle_runtime_exit_requests(event_loop)
            || self.handle_gpu_device_loss_request(event_loop)
        {
            return Err(FrameTickOutcome::ExitRequested);
        }
        let one_credit_begin_sent = self.drain_completion_and_try_desktop_one_credit();
        if self.runtime.should_send_begin_frame_before_wait_work() {
            self.lock_step_exchange();
        }
        self.run_asset_integration_phase();
        Ok(FrameStartupState {
            one_credit_begin_sent,
            vr_active: self.runtime.vr_active(),
        })
    }

    pub(super) fn run_pre_xr_lockstep_phase(
        &mut self,
        event_loop: &dyn ActiveEventLoop,
        vr_active: bool,
    ) -> Result<(), FrameTickOutcome> {
        let pre_xr_action = pre_xr_lockstep_action(PreXrLockstepInput {
            vr_active,
            awaiting_frame_submit: self.runtime.awaiting_frame_submit(),
            should_render_frame: self.runtime.should_render_frame(),
            should_send_begin_frame: self.runtime.should_send_begin_frame(),
        });
        match pre_xr_action {
            super::PreXrLockstepAction::Continue => Ok(()),
            super::PreXrLockstepAction::WaitForSubmit => self
                .wait_for_host_submit_before_xr(event_loop)
                .map_or(Ok(()), Err),
            super::PreXrLockstepAction::SendBeginThenWait => {
                self.lock_step_exchange();
                self.wait_for_host_submit_before_xr(event_loop)
                    .map_or(Ok(()), Err)
            }
            super::PreXrLockstepAction::SkipUntilHostReady => Err(FrameTickOutcome::RenderSkipped),
        }
    }

    pub(super) fn begin_xr_frame_phase(
        &mut self,
        event_loop: &dyn ActiveEventLoop,
        startup: FrameStartupState,
    ) -> Result<XrFrameBeginState, FrameTickOutcome> {
        let xr_pause = self
            .main_heartbeat
            .as_ref()
            .map(|heartbeat| heartbeat.pause());
        let xr_tick = self.xr_begin_tick();
        drop(xr_pause);

        if !startup.vr_active {
            self.lock_step_exchange();
        }
        if self.handle_openxr_exit_request(event_loop) {
            self.queue_empty_openxr_frame_if_needed(xr_tick);
            self.poll_graceful_shutdown(event_loop);
            return Err(FrameTickOutcome::ExitRequested);
        }
        let one_credit_begin_sent = if startup.vr_active
            && xr_tick.is_some()
            && self.runtime.should_send_one_credit_begin_frame()
        {
            self.one_credit_lock_step_exchange() || startup.one_credit_begin_sent
        } else {
            startup.one_credit_begin_sent
        };
        Ok(XrFrameBeginState {
            one_credit_begin_sent,
            vr_active: startup.vr_active,
            xr_tick,
        })
    }

    pub(super) fn ensure_frame_renderable_after_xr(
        &mut self,
        event_loop: &dyn ActiveEventLoop,
        state: XrFrameBeginState,
    ) -> Result<XrFrameBeginState, FrameTickOutcome> {
        if self.runtime.should_render_frame() {
            return Ok(state);
        }
        if !state.vr_active {
            self.runtime
                .wait_for_coupled_submit_or_decoupling(HostWaitReason::DesktopAwaitingSubmit);
            if self.handle_runtime_exit_requests(event_loop) {
                self.queue_empty_openxr_frame_if_needed(state.xr_tick);
                return Err(FrameTickOutcome::ExitRequested);
            }
        }
        if self.runtime.should_render_frame() {
            Ok(state)
        } else {
            self.consume_unrendered_one_credit_submit(state.one_credit_begin_sent);
            self.queue_empty_openxr_frame_if_needed(state.xr_tick);
            Err(FrameTickOutcome::RenderSkipped)
        }
    }

    pub(super) fn render_and_present_frame(
        &mut self,
        event_loop: &dyn ActiveEventLoop,
        state: XrFrameBeginState,
    ) -> FrameTickOutcome {
        let Some(window) = self
            .target
            .as_ref()
            .map(|target| Arc::clone(target.window()))
        else {
            self.consume_unrendered_one_credit_submit(state.one_credit_begin_sent);
            return FrameTickOutcome::MissingTarget;
        };
        let Some(render_outcome) = self.render_views(&window, state.xr_tick.as_ref()) else {
            self.consume_unrendered_one_credit_submit(state.one_credit_begin_sent);
            return FrameTickOutcome::MissingTarget;
        };
        self.runtime.note_frame_render_attempted();
        let hmd_projection_ended = render_outcome.hmd_projection_ended;
        self.hmd_compositor_paced_last_frame = hmd_projection_ended;
        if self.handle_gpu_device_loss_request(event_loop) {
            if !hmd_projection_ended {
                self.queue_empty_openxr_frame_if_needed(state.xr_tick);
            }
            self.poll_graceful_shutdown(event_loop);
            return FrameTickOutcome::ExitRequested;
        }
        self.present_and_diagnostics(state.xr_tick, hmd_projection_ended);
        self.drain_submit_completion_work();
        if state.vr_active || !state.one_credit_begin_sent {
            self.lock_step_exchange();
        }
        FrameTickOutcome::Presented
    }
}
