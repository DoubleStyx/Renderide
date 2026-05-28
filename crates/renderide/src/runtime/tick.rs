//! Per-tick lifecycle on [`super::RendererRuntime`].
//!
//! Owns the prologue, the lock-step / output forwards the app driver invokes inside one
//! redraw iteration, and the two top-level tick entry points that compose [`Self::poll_ipc`],
//! [`Self::run_asset_integration`], [`Self::maintain_nonblocking_gpu_jobs`],
//! [`Self::drain_reflection_probe_render_tasks`], [`Self::drain_camera_render_tasks`],
//! [`Self::pre_frame`], and [`Self::render_desktop_frame`] in their fixed order.

use std::time::{Duration, Instant};

use crate::diagnostics::crash_context::{self, TickPhase};
use crate::gpu::GpuContext;
use crate::shared::{InputState, OutputState};

use super::{RendererRuntime, TickOutcome};

/// Longest single semaphore wait while desktop lock-step is coupled.
const MAX_DESKTOP_LOCKSTEP_WAIT_SLICE: Duration = Duration::from_secs(1);

impl RendererRuntime {
    /// Whether the next tick should build [`InputState`] and call [`Self::pre_frame`].
    pub fn should_send_begin_frame(&self) -> bool {
        self.frontend.should_send_begin_frame()
    }

    /// Whether the current tick may render world state under host lockstep and decoupling rules.
    pub fn should_render_frame(&self) -> bool {
        self.frontend.should_render_frame()
    }

    /// Whether a `FrameStartData` has been sent and the matching host submit is still outstanding.
    pub fn awaiting_frame_submit(&self) -> bool {
        self.frontend.awaiting_frame_submit()
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

    /// Per-tick decoupling activation check. Call **after** [`Self::poll_ipc`] (so a
    /// `FrameSubmitData` already drained this tick clears the awaiting flag and prevents a
    /// stale-wait spurious activation) and **before** [`Self::run_asset_integration`] (so the
    /// decoupled-mode asset budget reflects the latest state). Do not call after
    /// [`Self::pre_frame`]: a fresh BeginFrame send would zero the elapsed wait and the check
    /// would never fire.
    pub fn update_decoupling_activation(&mut self, now: Instant) {
        self.frontend.update_decoupling_activation(now);
    }

    /// Most recent `FrameStartData` send to matching `FrameSubmitData` receive duration.
    pub fn last_frame_begin_to_submit(&self) -> Option<Duration> {
        self.frontend.last_frame_begin_to_submit()
    }

    /// Waits for the coupled desktop frame submit or for renderer decoupling to activate.
    ///
    /// Host-compatible lockstep waits inside the same tick after sending `FrameStartData`; doing
    /// that here prevents desktop from alternating one redraw that only requests a host frame with
    /// the next redraw that renders it. Active asset integration remains counted as CPU work; only
    /// the semaphore wait itself is excluded from HUD CPU-frame timing.
    pub fn wait_for_desktop_coupled_submit_or_decoupling(&mut self) {
        profiling::scope!("tick::desktop_lockstep_wait");
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

            if self.run_asset_integration_while_waiting_for_submit(now) {
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
                .decoupling_activation_wait_timeout(now, MAX_DESKTOP_LOCKSTEP_WAIT_SLICE)
            else {
                break;
            };
            if timeout.is_zero() {
                self.poll_ipc();
                self.run_asset_integration_after_wait_poll();
                continue;
            }
            excluded_wait = excluded_wait.saturating_add(self.poll_ipc_after_primary_wait(timeout));
            self.run_asset_integration_after_wait_poll();
        }
        self.note_frame_timing_excluded_wait(excluded_wait);
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
    /// value per video asset until the next allowed send.
    pub fn pre_frame(&mut self, inputs: InputState) {
        profiling::scope!("tick::pre_frame");
        let video_clock_errors = self.backend.take_pending_video_clock_errors();
        self.frontend.enqueue_video_clock_errors(video_clock_errors);
        self.frontend.pre_frame(inputs);
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
    /// Phases: drain IPC, dispatch asset integration, drain reflection-probe and camera readbacks,
    /// emit lock-step `FrameStartData` via [`Self::pre_frame`] (when allowed), and call
    /// [`Self::render_frame`] with the main camera included. Mode-specific epilogue (HUD overlay
    /// encode + present in winit, PNG readback in headless) happens on the caller side after this
    /// returns.
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
        crash_context::set_tick_phase(TickPhase::AssetIntegration);
        self.run_asset_integration();
        self.maintain_nonblocking_gpu_jobs(gpu);
        self.drain_reflection_probe_render_tasks(gpu);
        self.drain_camera_render_tasks(gpu);
        crash_context::set_tick_phase(TickPhase::Lockstep);
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
    /// Used by the desktop VR path which runs its own HMD multiview submit + secondary cameras
    /// to render textures + mirror blit instead of [`Self::render_frame`]. Phase order stays
    /// in this method so VR cannot drift from desktop / headless lock-step semantics.
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
        crash_context::set_tick_phase(TickPhase::AssetIntegration);
        self.run_asset_integration();
        if let Some(gpu) = gpu {
            self.maintain_nonblocking_gpu_jobs(gpu);
            self.drain_reflection_probe_render_tasks(gpu);
            self.drain_camera_render_tasks(gpu);
        }
        crash_context::set_tick_phase(TickPhase::Lockstep);
        if self.should_send_begin_frame() {
            self.pre_frame(inputs);
        }
        TickOutcome::default()
    }
}
