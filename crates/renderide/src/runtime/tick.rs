//! Per-tick lifecycle on [`super::RendererRuntime`].
//!
//! Owns the prologue, the lock-step / output forwards the app driver invokes inside one
//! redraw iteration, and the two top-level tick entry points that compose [`Self::poll_ipc`],
//! [`Self::maintain_nonblocking_gpu_jobs`], [`Self::drain_reflection_probe_render_tasks`],
//! [`Self::drain_camera_render_tasks`], [`Self::pre_frame`], [`Self::pre_frame_one_credit`],
//! [`Self::run_asset_integration`], and [`Self::render_desktop_frame`] in their fixed order.

use std::time::{Duration, Instant};

use crate::diagnostics::crash_context::{self, TickPhase};
use crate::frontend::{
    HostWaitReason, LockstepPipelineAction, LockstepPipelineInput, OneCreditBlockReason,
    decide_lockstep_pipeline, one_credit_block_reason,
};
use crate::gpu::GpuContext;
use crate::shared::{InputState, OutputState};

use super::{RendererRuntime, TickOutcome};

/// Longest single semaphore wait while host lock-step is coupled.
const MAX_COUPLED_LOCKSTEP_WAIT_SLICE: Duration = Duration::from_secs(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BeginFrameBeforeWaitWorkInput {
    awaiting_frame_submit: bool,
    should_render_frame: bool,
    should_send_begin_frame: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RegularBeginFrameInput {
    frontend_allows_begin_frame: bool,
    submit_completion_work_drained: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct OneCreditBeginFrameInput {
    awaiting_frame_submit: bool,
    pending_frame_submit_render: bool,
    should_send_begin_frame: bool,
    submit_completion_work_drained: bool,
}

fn should_send_begin_frame_before_wait_work(input: BeginFrameBeforeWaitWorkInput) -> bool {
    input.should_send_begin_frame && !input.awaiting_frame_submit && !input.should_render_frame
}

fn should_send_regular_begin_frame(input: RegularBeginFrameInput) -> bool {
    input.frontend_allows_begin_frame && input.submit_completion_work_drained
}

fn should_send_one_credit_begin_frame(input: OneCreditBeginFrameInput) -> bool {
    input.should_send_begin_frame
        && !input.awaiting_frame_submit
        && input.pending_frame_submit_render
        && input.submit_completion_work_drained
}

impl RendererRuntime {
    /// Whether the next tick should build [`InputState`] and call [`Self::pre_frame`].
    pub fn should_send_begin_frame(&self) -> bool {
        should_send_regular_begin_frame(RegularBeginFrameInput {
            frontend_allows_begin_frame: self.frontend.should_send_begin_frame(),
            submit_completion_work_drained: self.submit_completion_work_drained(),
        })
    }

    /// Whether the current tick may render world state under host lockstep and decoupling rules.
    pub fn should_render_frame(&self) -> bool {
        self.frontend.should_render_frame()
    }

    /// Whether a `FrameStartData` has been sent and the matching host submit is still outstanding.
    pub fn awaiting_frame_submit(&self) -> bool {
        self.frontend.awaiting_frame_submit()
    }

    /// Whether submit-attached host-critical completion work is drained before host frame finalization.
    pub(crate) fn submit_completion_work_drained(&self) -> bool {
        self.tick_state.pending_camera_render_tasks.is_empty()
            && self
                .tick_state
                .pending_reflection_probe_render_tasks
                .is_empty()
    }

    /// Computes the current one-credit lock-step pipeline decision.
    pub(crate) fn lockstep_pipeline_decision(
        &self,
    ) -> (LockstepPipelineAction, OneCreditBlockReason) {
        let input = LockstepPipelineInput {
            begin_frame_base_allowed: self.frontend.begin_frame_base_allowed(),
            regular_begin_frame_allowed: self.should_send_begin_frame(),
            awaiting_frame_submit: self.awaiting_frame_submit(),
            pending_frame_submit_render: self.frontend.pending_frame_submit_render(),
            should_render_frame: self.should_render_frame(),
            submit_completion_work_drained: self.submit_completion_work_drained(),
        };
        (
            decide_lockstep_pipeline(input),
            one_credit_block_reason(input),
        )
    }

    /// Records and returns the current lock-step pipeline decision.
    pub(crate) fn record_lockstep_pipeline_decision(&mut self) -> LockstepPipelineAction {
        let (action, block) = self.lockstep_pipeline_decision();
        self.tick_state
            .record_lockstep_pipeline_decision(action, block);
        action
    }

    /// Records why a host-submit wait fallback is active.
    pub(crate) fn record_lockstep_wait_reason(&mut self, reason: HostWaitReason) {
        self.tick_state.record_lockstep_wait_reason(reason);
    }

    /// Whether the next host frame may be requested before rendering the current submit.
    pub(crate) fn should_send_one_credit_begin_frame(&self) -> bool {
        if self.shutdown_requested() || self.fatal_error() {
            return false;
        }
        let submit_completion_work_drained = self.submit_completion_work_drained();
        should_send_one_credit_begin_frame(OneCreditBeginFrameInput {
            awaiting_frame_submit: self.awaiting_frame_submit(),
            pending_frame_submit_render: self.frontend.pending_frame_submit_render(),
            should_send_begin_frame: self
                .frontend
                .should_send_one_credit_begin_frame(submit_completion_work_drained),
            submit_completion_work_drained,
        })
    }

    /// Whether idle lock-step should send `FrameStartData` before renderer wait-work this tick.
    pub(crate) fn should_send_begin_frame_before_wait_work(&self) -> bool {
        should_send_begin_frame_before_wait_work(BeginFrameBeforeWaitWorkInput {
            awaiting_frame_submit: self.awaiting_frame_submit(),
            should_render_frame: self.should_render_frame(),
            should_send_begin_frame: self.should_send_begin_frame(),
        })
    }

    /// Marks any processed host frame submit as having had a renderer-side draw attempt.
    pub fn note_frame_render_attempted(&mut self) {
        self.frontend.note_frame_render_attempted();
    }

    /// Records wall-clock spacing for host FPS metrics. Call at the very start of each winit tick,
    /// before [`Self::poll_ipc`], OpenXR, and [`Self::pre_frame`].
    pub fn tick_frame_wall_clock_begin(&mut self, now: Instant) {
        profiling::scope!("tick::frame_wall_clock_begin");
        crash_context::record_tick_start();
        self.tick_state.reset_for_tick();
        self.tick_state.note_frame_wall_clock_begin(now);
        self.frontend.reset_ipc_outbound_drop_tick_flags();
        self.backend.reset_light_prep_for_tick();
        self.frontend.on_tick_frame_wall_clock(now);
        let (primary, background) = self.frontend.ipc_consecutive_outbound_drop_streaks();
        crash_context::set_ipc_drop_streaks(primary, background);
    }

    /// Adds main-thread pacing time observed outside [`GpuContext`] to the current frame timing.
    pub(crate) fn note_frame_timing_excluded_wait(&mut self, wait: Duration) {
        self.tick_state.note_frame_timing_excluded_wait(wait);
    }

    /// Drains pacing time accumulated outside [`GpuContext`] for HUD CPU-frame accounting.
    pub(crate) fn drain_frame_timing_excluded_wait(&mut self) -> Duration {
        self.tick_state.drain_frame_timing_excluded_wait()
    }

    /// Per-tick decoupling activation check. Call **after** [`Self::poll_ipc`] so a
    /// `FrameSubmitData` already drained this tick clears the awaiting flag and prevents a
    /// stale-wait spurious activation. Also call it before [`Self::run_asset_integration`] so the
    /// decoupled-mode asset budget reflects the latest state. Do not call after
    /// [`Self::pre_frame`] when the goal is checking an existing wait: a fresh BeginFrame send
    /// would zero the elapsed wait.
    pub fn update_decoupling_activation(&mut self, now: Instant) {
        self.frontend.update_decoupling_activation(now);
    }

    /// Most recent `FrameStartData` send to matching `FrameSubmitData` receive duration.
    pub fn last_frame_begin_to_submit(&self) -> Option<Duration> {
        self.frontend.last_frame_begin_to_submit()
    }

    /// Waits for the coupled host frame submit or for renderer decoupling to activate.
    ///
    /// Host-compatible lockstep waits inside the same tick after sending `FrameStartData` when a
    /// mode can safely wait before rendering. Desktop uses this after begin-frame sends; VR uses it
    /// before `xrBeginFrame` so host waits cannot produce empty OpenXR frames. Active asset
    /// integration remains counted as CPU work; only the semaphore wait itself is excluded from HUD
    /// CPU-frame timing.
    pub fn wait_for_coupled_submit_or_decoupling(&mut self, wait_reason: HostWaitReason) {
        profiling::scope!("tick::coupled_lockstep_wait");
        self.record_lockstep_wait_reason(wait_reason);
        let mut excluded_wait = Duration::ZERO;
        loop {
            let now = Instant::now();
            self.update_decoupling_activation(now);
            if self.should_render_frame()
                || !self.awaiting_frame_submit()
                || self.shutdown_requested()
                || self.fatal_error()
            {
                break;
            }

            let has_more_asset_work = {
                profiling::scope!("tick::lockstep_wait::asset_work");
                self.run_asset_integration_while_waiting_for_submit(now)
            };
            if has_more_asset_work {
                continue;
            }

            let now = Instant::now();
            self.update_decoupling_activation(now);
            if self.should_render_frame()
                || !self.awaiting_frame_submit()
                || self.shutdown_requested()
                || self.fatal_error()
            {
                break;
            }

            let Some(timeout) = self
                .frontend
                .decoupling_activation_wait_timeout(now, MAX_COUPLED_LOCKSTEP_WAIT_SLICE)
            else {
                break;
            };
            if timeout.is_zero() {
                {
                    profiling::scope!("tick::lockstep_wait::ipc_command_processing");
                    self.poll_ipc();
                }
                {
                    profiling::scope!("tick::lockstep_wait::asset_work_after_ipc");
                    self.run_asset_integration_after_wait_poll();
                }
                continue;
            }
            excluded_wait = excluded_wait.saturating_add({
                profiling::scope!("tick::lockstep_wait::host_wait");
                self.poll_ipc_after_primary_wait(timeout)
            });
            {
                profiling::scope!("tick::lockstep_wait::asset_work_after_ipc");
                self.run_asset_integration_after_wait_poll();
            }
        }
        self.note_frame_timing_excluded_wait(excluded_wait);
        self.record_lockstep_pipeline_decision();
    }

    /// Increments the renderer-tick counter feeding
    /// [`crate::shared::PerformanceState::rendered_frames_since_last`]. Call once per completed
    /// tick from the app driver's redraw epilogue.
    pub fn note_render_tick_complete(&mut self) {
        self.frontend.note_render_tick_complete();
    }

    /// Forwards the most recently completed whole-frame GPU interval to the frontend so the next
    /// [`crate::shared::PerformanceState::render_time`] reports Unity-compatible GPU render time.
    /// Pass [`None`] when no full frame GPU sample has completed yet; the frontend maps that to
    /// the host-visible `-1.0` sentinel.
    ///
    /// Call once before every return from the app driver's redraw tick.
    pub fn tick_frame_render_time_end(&mut self, gpu_render_time_seconds: Option<f32>) {
        self.frontend
            .set_perf_last_render_time_seconds(gpu_render_time_seconds);
    }

    /// Host [`OutputState::lock_cursor`] bit merged into packed mouse state.
    pub fn host_cursor_lock_requested(&self) -> bool {
        self.frontend.host_cursor_lock_requested()
    }

    /// If connected and init is complete, sends [`FrameStartData`](crate::shared::FrameStartData) when we are ready for the next host frame.
    ///
    /// Drains latest video clock-error samples produced by the asset integrator into the
    /// frontend so the next outgoing [`FrameStartData`](crate::shared::FrameStartData::video_clock_errors)
    /// carries them. The drain runs unconditionally because the frontend itself decides whether
    /// the begin-frame send fires; if the send is skipped, samples remain bounded to the latest
    /// value per video asset until the next allowed send. Returns whether the primary command was
    /// actually enqueued.
    pub fn pre_frame(&mut self, inputs: InputState) -> bool {
        profiling::scope!("tick::pre_frame");
        if !self.should_send_begin_frame() {
            return false;
        }
        let video_clock_errors = self.backend.take_pending_video_clock_errors();
        self.frontend.enqueue_video_clock_errors(video_clock_errors);
        self.frontend.pre_frame(inputs)
    }

    /// Sends a one-credit [`FrameStartData`](crate::shared::FrameStartData) before rendering.
    pub(crate) fn pre_frame_one_credit(&mut self, inputs: InputState) -> bool {
        profiling::scope!("tick::pre_frame_one_credit");
        if !self.should_send_one_credit_begin_frame() {
            return false;
        }
        let video_clock_errors = self.backend.take_pending_video_clock_errors();
        self.frontend.enqueue_video_clock_errors(video_clock_errors);
        self.frontend
            .pre_frame_one_credit(inputs, self.submit_completion_work_drained())
    }

    /// Drains pending host window policy after [`Self::poll_ipc`].
    pub fn take_pending_output_state(&mut self) -> Option<OutputState> {
        self.frontend.take_pending_output_state()
    }

    /// Last [`OutputState`] from the host (for per-frame cursor lock / warp).
    pub fn last_output_state(&self) -> Option<&OutputState> {
        self.frontend.last_output_state()
    }

    /// Runs the canonical per-frame phase order shared between the winit-driven
    /// app-driver redraw tick (non-VR) and the headless interval driver.
    ///
    /// Phases: drain IPC, drain completed GPU/offscreen results, emit idle lock-step
    /// `FrameStartData` via [`Self::pre_frame`] before wait-work when allowed, dispatch asset
    /// integration, emit any still-allowed frame-start, and call [`Self::render_frame`] with the
    /// main camera included. Mode-specific epilogue (HUD overlay encode + present in winit, PNG
    /// readback in headless) happens on the caller side after this returns.
    pub fn tick_one_frame(&mut self, gpu: &mut GpuContext, inputs: InputState) -> TickOutcome {
        profiling::scope!("tick::one_frame");
        crash_context::set_tick_phase(TickPhase::IpcPoll);
        self.poll_ipc();
        if self.shutdown_requested() {
            return TickOutcome {
                shutdown_requested: true,
                ..Default::default()
            };
        }
        if self.fatal_error() {
            return TickOutcome {
                fatal_error: true,
                ..Default::default()
            };
        }
        self.update_decoupling_activation(Instant::now());
        self.maintain_nonblocking_gpu_jobs(gpu);
        self.drain_reflection_probe_render_tasks(gpu);
        self.drain_camera_render_tasks(gpu);
        crash_context::set_tick_phase(TickPhase::Lockstep);
        if self.record_lockstep_pipeline_decision() == LockstepPipelineAction::SendEarlyNextFrame {
            self.pre_frame_one_credit(inputs.clone());
        }
        crash_context::set_tick_phase(TickPhase::Lockstep);
        if self.should_send_begin_frame_before_wait_work() {
            self.pre_frame(inputs.clone());
        }
        self.update_decoupling_activation(Instant::now());
        crash_context::set_tick_phase(TickPhase::AssetIntegration);
        self.run_asset_integration();
        crash_context::set_tick_phase(TickPhase::Lockstep);
        self.record_lockstep_pipeline_decision();
        if self.should_send_begin_frame() {
            self.pre_frame(inputs);
        }
        if !self.should_render_frame() {
            return TickOutcome {
                render_skipped: true,
                ..Default::default()
            };
        }
        crash_context::set_tick_phase(TickPhase::RenderViews);
        let graph_error = self.render_desktop_frame(gpu).err();
        self.note_frame_render_attempted();
        TickOutcome {
            graph_error,
            ..Default::default()
        }
    }

    /// Same as [`Self::tick_one_frame`] but skips the render call.
    ///
    /// Used by lockstep-only drivers that need IPC, asset integration, and GPU maintenance
    /// without a main-view render in this tick.
    pub fn tick_one_frame_lockstep_only(
        &mut self,
        gpu: Option<&mut GpuContext>,
        inputs: InputState,
    ) -> TickOutcome {
        profiling::scope!("tick::one_frame_lockstep_only");
        crash_context::set_tick_phase(TickPhase::IpcPoll);
        self.poll_ipc();
        if self.shutdown_requested() {
            return TickOutcome {
                shutdown_requested: true,
                ..Default::default()
            };
        }
        if self.fatal_error() {
            return TickOutcome {
                fatal_error: true,
                ..Default::default()
            };
        }
        self.update_decoupling_activation(Instant::now());
        if let Some(gpu) = gpu {
            self.maintain_nonblocking_gpu_jobs(gpu);
            self.drain_reflection_probe_render_tasks(gpu);
            self.drain_camera_render_tasks(gpu);
        }
        crash_context::set_tick_phase(TickPhase::Lockstep);
        if self.should_send_begin_frame_before_wait_work() {
            self.pre_frame(inputs.clone());
        }
        self.update_decoupling_activation(Instant::now());
        crash_context::set_tick_phase(TickPhase::AssetIntegration);
        self.run_asset_integration();
        crash_context::set_tick_phase(TickPhase::Lockstep);
        self.record_lockstep_pipeline_decision();
        if self.should_send_begin_frame() {
            self.pre_frame(inputs);
        }
        TickOutcome::default()
    }
}

#[cfg(test)]
mod tests {
    use super::{BeginFrameBeforeWaitWorkInput, should_send_begin_frame_before_wait_work};
    use super::{OneCreditBeginFrameInput, should_send_one_credit_begin_frame};
    use super::{RegularBeginFrameInput, should_send_regular_begin_frame};

    fn input() -> BeginFrameBeforeWaitWorkInput {
        BeginFrameBeforeWaitWorkInput {
            awaiting_frame_submit: false,
            should_render_frame: false,
            should_send_begin_frame: true,
        }
    }

    #[test]
    fn idle_lockstep_sends_begin_before_wait_work() {
        assert!(should_send_begin_frame_before_wait_work(input()));
    }

    #[test]
    fn awaiting_submit_does_not_send_another_begin_before_wait_work() {
        assert!(!should_send_begin_frame_before_wait_work(
            BeginFrameBeforeWaitWorkInput {
                awaiting_frame_submit: true,
                ..input()
            }
        ));
    }

    #[test]
    fn renderable_frames_defer_begin_to_the_existing_render_path() {
        assert!(!should_send_begin_frame_before_wait_work(
            BeginFrameBeforeWaitWorkInput {
                should_render_frame: true,
                ..input()
            }
        ));
    }

    fn one_credit_input() -> OneCreditBeginFrameInput {
        OneCreditBeginFrameInput {
            awaiting_frame_submit: false,
            pending_frame_submit_render: true,
            should_send_begin_frame: true,
            submit_completion_work_drained: true,
        }
    }

    #[test]
    fn one_credit_sends_for_renderable_processed_submit() {
        assert!(should_send_one_credit_begin_frame(one_credit_input()));
    }

    #[test]
    fn one_credit_does_not_duplicate_in_flight_begin() {
        assert!(!should_send_one_credit_begin_frame(
            OneCreditBeginFrameInput {
                awaiting_frame_submit: true,
                ..one_credit_input()
            }
        ));
    }

    #[test]
    fn one_credit_waits_for_submit_completion_work() {
        assert!(!should_send_one_credit_begin_frame(
            OneCreditBeginFrameInput {
                submit_completion_work_drained: false,
                ..one_credit_input()
            }
        ));
    }

    #[test]
    fn one_credit_requires_pending_renderable_submit() {
        assert!(!should_send_one_credit_begin_frame(
            OneCreditBeginFrameInput {
                pending_frame_submit_render: false,
                ..one_credit_input()
            }
        ));
    }

    fn regular_begin_input() -> RegularBeginFrameInput {
        RegularBeginFrameInput {
            frontend_allows_begin_frame: true,
            submit_completion_work_drained: true,
        }
    }

    #[test]
    fn regular_begin_sends_when_frontend_and_submit_completion_allow() {
        assert!(should_send_regular_begin_frame(regular_begin_input()));
    }

    #[test]
    fn regular_begin_waits_for_submit_completion_work() {
        assert!(!should_send_regular_begin_frame(RegularBeginFrameInput {
            submit_completion_work_drained: false,
            ..regular_begin_input()
        }));
    }

    #[test]
    fn regular_begin_respects_frontend_gate() {
        assert!(!should_send_regular_begin_frame(RegularBeginFrameInput {
            frontend_allows_begin_frame: false,
            ..regular_begin_input()
        }));
    }
}
