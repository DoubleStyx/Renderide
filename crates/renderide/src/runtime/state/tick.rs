//! Runtime-owned per-tick scratch and phase gates.

use std::time::{Duration, Instant};

use crate::frontend::{
    HostWaitReason, LockstepPipelineAction, LockstepPipelineHudLabels, OneCreditBlockReason,
};
use crate::scene::{ReflectionProbeOnChangesRenderRequest, RenderSpaceId};
use crate::shared::{CameraRenderTask, ReflectionProbeRenderResult, ReflectionProbeRenderTask};

use crate::runtime::offscreen_tasks::reflection_probe::{
    ActiveOnChangesReflectionProbeCapture, ActiveRealtimeReflectionProbeCapture,
};

/// Reflection-probe bake task plus the render space that carried it.
#[derive(Clone, Debug)]
pub(in crate::runtime) struct QueuedReflectionProbeRenderTask {
    /// Host render space containing the reflection probe.
    pub(in crate::runtime) render_space_id: RenderSpaceId,
    /// Host bake task payload.
    pub(in crate::runtime) task: ReflectionProbeRenderTask,
}

/// Per-tick gates and reusable view-planning scratch.
pub(in crate::runtime) struct RuntimeTickState {
    /// Wall-clock anchor for Unity-style shader time inputs.
    started_at: Instant,
    /// Elapsed renderer runtime in seconds captured at the start of the current tick.
    frame_time_seconds: f32,
    /// Set when asset integration completed for the current winit tick.
    did_integrate_this_tick: bool,
    /// Main-thread compositor pacing waits observed outside [`crate::gpu::GpuContext`] this tick.
    frame_timing_excluded_wait: Duration,
    /// Last host/renderer lock-step pipeline action selected this tick.
    lockstep_pipeline_action: LockstepPipelineAction,
    /// Last reason an early one-credit begin-frame was blocked this tick.
    lockstep_one_credit_block: OneCreditBlockReason,
    /// Last reason the runtime waited for a host submit this tick.
    lockstep_wait_reason: HostWaitReason,
    /// Reusable per-frame scratch for secondary render-texture view collection.
    pub(in crate::runtime) secondary_view_tasks_scratch: Vec<(RenderSpaceId, f32, usize)>,
    /// Host camera readback tasks waiting for a GPU context before the next begin-frame send.
    pub(in crate::runtime) pending_camera_render_tasks: Vec<CameraRenderTask>,
    /// Host reflection-probe bake tasks waiting for a GPU context before the next begin-frame send.
    pub(in crate::runtime) pending_reflection_probe_render_tasks:
        Vec<QueuedReflectionProbeRenderTask>,
    /// Reflection-probe bake results waiting for the background IPC queue to accept them.
    pub(in crate::runtime) pending_reflection_probe_render_results:
        Vec<ReflectionProbeRenderResult>,
    /// OnChanges reflection-probe capture requests waiting for GPU processing.
    pub(in crate::runtime) pending_onchanges_reflection_probe_requests:
        Vec<ReflectionProbeOnChangesRenderRequest>,
    /// OnChanges reflection-probe captures that may span multiple ticks.
    pub(in crate::runtime) active_onchanges_reflection_probe_captures:
        Vec<ActiveOnChangesReflectionProbeCapture>,
    /// Next renderer-side OnChanges cubemap capture generation.
    pub(in crate::runtime) next_onchanges_reflection_probe_generation: u64,
    /// Realtime reflection-probe captures that may span multiple ticks.
    pub(in crate::runtime) active_realtime_reflection_probe_captures:
        Vec<ActiveRealtimeReflectionProbeCapture>,
    /// Next renderer-side realtime cubemap capture generation.
    pub(in crate::runtime) next_realtime_reflection_probe_generation: u64,
}

impl RuntimeTickState {
    /// Creates empty tick state.
    pub(in crate::runtime) fn new() -> Self {
        let started_at = Instant::now();
        Self {
            started_at,
            frame_time_seconds: 0.0,
            did_integrate_this_tick: false,
            frame_timing_excluded_wait: Duration::ZERO,
            lockstep_pipeline_action: LockstepPipelineAction::None,
            lockstep_one_credit_block: OneCreditBlockReason::None,
            lockstep_wait_reason: HostWaitReason::None,
            secondary_view_tasks_scratch: Vec::new(),
            pending_camera_render_tasks: Vec::new(),
            pending_reflection_probe_render_tasks: Vec::new(),
            pending_reflection_probe_render_results: Vec::new(),
            pending_onchanges_reflection_probe_requests: Vec::new(),
            active_onchanges_reflection_probe_captures: Vec::new(),
            next_onchanges_reflection_probe_generation: 1,
            active_realtime_reflection_probe_captures: Vec::new(),
            next_realtime_reflection_probe_generation: 1,
        }
    }

    /// Clears once-per-tick gates at the start of a new winit tick.
    pub(in crate::runtime) fn reset_for_tick(&mut self) {
        self.did_integrate_this_tick = false;
        self.frame_timing_excluded_wait = Duration::ZERO;
        self.lockstep_pipeline_action = LockstepPipelineAction::None;
        self.lockstep_one_credit_block = OneCreditBlockReason::None;
        self.lockstep_wait_reason = HostWaitReason::None;
    }

    /// Captures the frame-start wall clock for material shader time inputs.
    pub(in crate::runtime) fn note_frame_wall_clock_begin(&mut self, now: Instant) {
        self.frame_time_seconds = now.saturating_duration_since(self.started_at).as_secs_f32();
    }

    /// Elapsed renderer runtime in seconds captured at the start of the current tick.
    pub(in crate::runtime) fn frame_time_seconds(&self) -> f32 {
        self.frame_time_seconds
    }

    /// Whether asset integration already ran this tick.
    pub(in crate::runtime) fn did_integrate_assets_this_tick(&self) -> bool {
        self.did_integrate_this_tick
    }

    /// Marks asset integration as completed for this tick.
    pub(in crate::runtime) fn mark_integrated_assets_this_tick(&mut self) {
        self.did_integrate_this_tick = true;
    }

    /// Adds compositor or display pacing time that should not count as active CPU frame work.
    pub(in crate::runtime) fn note_frame_timing_excluded_wait(&mut self, wait: Duration) {
        self.frame_timing_excluded_wait = self.frame_timing_excluded_wait.saturating_add(wait);
    }

    /// Drains accumulated non-GPU-context pacing time for publication into frame timing.
    pub(in crate::runtime) fn drain_frame_timing_excluded_wait(&mut self) -> Duration {
        let wait = self.frame_timing_excluded_wait;
        self.frame_timing_excluded_wait = Duration::ZERO;
        wait
    }

    /// Records the latest lock-step pipeline decision for profiling and HUD diagnostics.
    pub(in crate::runtime) fn record_lockstep_pipeline_decision(
        &mut self,
        action: LockstepPipelineAction,
        one_credit_block: OneCreditBlockReason,
    ) {
        self.lockstep_pipeline_action = action;
        self.lockstep_one_credit_block = one_credit_block;
        self.plot_lockstep_pipeline();
    }

    /// Records why the runtime entered a host-submit wait fallback.
    pub(in crate::runtime) fn record_lockstep_wait_reason(&mut self, reason: HostWaitReason) {
        self.lockstep_pipeline_action = LockstepPipelineAction::WaitForSubmit;
        self.lockstep_wait_reason = reason;
        self.plot_lockstep_pipeline();
    }

    /// Labels for the frame-timing HUD.
    pub(in crate::runtime) fn lockstep_pipeline_hud_labels(&self) -> LockstepPipelineHudLabels {
        LockstepPipelineHudLabels {
            action: self.lockstep_pipeline_action.label(),
            one_credit_block: self.lockstep_one_credit_block.label(),
            wait_reason: self.lockstep_wait_reason.label(),
        }
    }

    fn plot_lockstep_pipeline(&self) {
        crate::profiling::plot_lockstep_pipeline(
            &crate::profiling::LockstepPipelineProfileSample {
                action: self.lockstep_pipeline_action.plot_code(),
                one_credit_block: self.lockstep_one_credit_block.plot_code(),
                wait_reason: self.lockstep_wait_reason.plot_code(),
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::RuntimeTickState;

    #[test]
    fn asset_integration_gate_resets_per_tick() {
        let mut state = RuntimeTickState::new();

        assert!(!state.did_integrate_assets_this_tick());
        state.mark_integrated_assets_this_tick();
        assert!(state.did_integrate_assets_this_tick());
        state.reset_for_tick();
        assert!(!state.did_integrate_assets_this_tick());
    }

    #[test]
    fn frame_time_is_captured_from_runtime_start() {
        let mut state = RuntimeTickState::new();
        let later = state.started_at + std::time::Duration::from_millis(250);

        state.note_frame_wall_clock_begin(later);

        assert!((state.frame_time_seconds() - 0.25).abs() < 0.001);
    }

    #[test]
    fn frame_timing_excluded_wait_accumulates_and_drains() {
        let mut state = RuntimeTickState::new();
        state.note_frame_timing_excluded_wait(std::time::Duration::from_millis(2));
        state.note_frame_timing_excluded_wait(std::time::Duration::from_millis(3));

        assert_eq!(
            state.drain_frame_timing_excluded_wait(),
            std::time::Duration::from_millis(5)
        );
        assert_eq!(
            state.drain_frame_timing_excluded_wait(),
            std::time::Duration::ZERO
        );
    }
}
