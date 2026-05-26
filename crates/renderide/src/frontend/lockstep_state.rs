//! Lock-step frame cadence state and pending begin-frame payload queues.

use crate::shared::{
    FrameStartData, InputState, PerformanceState, ReflectionProbeChangeRenderResult,
    VideoTextureClockErrorState,
};

use super::begin_frame::{
    BeginFrameBuildInput, BeginFrameCommit, BeginFrameDecision, BeginFrameGateInput,
    build_frame_start, decide_begin_frame,
};

/// Runtime context supplied to the lockstep begin-frame gate.
pub(crate) struct LockstepBeginFrameContext {
    /// Whether the init handshake has finalized.
    pub(crate) init_finalized: bool,
    /// Whether a fatal frontend error suppresses lockstep.
    pub(crate) fatal_error: bool,
    /// Whether IPC is connected.
    pub(crate) ipc_connected: bool,
    /// Whether stale-scene rendering and early next-frame requests are currently allowed.
    pub(crate) renderer_decoupled: bool,
}

/// Lock-step state that determines when the renderer may ask the host for another frame.
pub(crate) struct LockstepState {
    /// Whether a host frame submit was applied since the last begin-frame send.
    last_frame_data_processed: bool,
    /// Whether the processed host frame has not yet had a renderer-side draw attempt.
    pending_frame_submit_render: bool,
    /// Whether `RendererEngineReady` has enabled strict host lockstep gating.
    host_lockstep_activated: bool,
    /// Current host frame index echoed in outgoing frame-start data.
    last_frame_index: i32,
    /// Whether the first bootstrap frame-start has already been sent.
    sent_bootstrap_frame_start: bool,
    /// Reflection probes completed since the previous successful frame-start send.
    pending_rendered_reflection_probes: Vec<ReflectionProbeChangeRenderResult>,
    /// Latest video texture clock errors since the previous successful frame-start send.
    pending_video_clock_errors: Vec<VideoTextureClockErrorState>,
}

impl LockstepState {
    /// Builds lock-step state for either standalone or host-connected mode.
    pub(crate) fn new(standalone: bool) -> Self {
        Self {
            last_frame_data_processed: standalone,
            pending_frame_submit_render: false,
            host_lockstep_activated: false,
            last_frame_index: -1,
            sent_bootstrap_frame_start: false,
            pending_rendered_reflection_probes: Vec::new(),
            pending_video_clock_errors: Vec::new(),
        }
    }

    /// Current host frame index echoed in outgoing frame-start data.
    pub(crate) fn last_frame_index(&self) -> i32 {
        self.last_frame_index
    }

    /// Whether a host frame submit was applied since the last begin-frame send.
    #[cfg(test)]
    pub(crate) fn last_frame_data_processed(&self) -> bool {
        self.last_frame_data_processed
    }

    /// Whether a frame submit is currently awaited.
    pub(crate) fn awaiting_submit(&self) -> bool {
        !self.last_frame_data_processed
    }

    /// Whether a processed host frame still needs a renderer-side draw.
    pub(crate) fn pending_frame_submit_render(&self) -> bool {
        self.pending_frame_submit_render
    }

    /// Whether the host has enabled strict lockstep gating with `RendererEngineReady`.
    pub(crate) fn host_lockstep_activated(&self) -> bool {
        self.host_lockstep_activated
    }

    /// Marks that init data arrived and the initial begin-frame may be sent after finalization.
    pub(crate) fn mark_init_received(&mut self) {
        self.last_frame_data_processed = true;
    }

    /// Marks that the host has finished startup and regular lockstep gating should apply.
    pub(crate) fn activate_host_lockstep(&mut self) {
        self.host_lockstep_activated = true;
    }

    /// Updates lock-step state after applying a frame submit.
    pub(crate) fn note_frame_submit_processed(&mut self, frame_index: i32) {
        self.last_frame_index = frame_index;
        self.last_frame_data_processed = true;
        self.pending_frame_submit_render = true;
    }

    /// Marks the processed host frame as having had a renderer-side draw attempt.
    pub(crate) fn note_frame_render_attempted(&mut self) {
        self.pending_frame_submit_render = false;
    }

    /// Appends reflection-probe render completions for the next outgoing frame-start.
    pub(crate) fn enqueue_rendered_reflection_probes(
        &mut self,
        probes: impl IntoIterator<Item = ReflectionProbeChangeRenderResult>,
    ) {
        self.pending_rendered_reflection_probes.extend(probes);
    }

    /// Records latest video texture clock-error samples for the next outgoing frame-start.
    pub(crate) fn enqueue_video_clock_errors(
        &mut self,
        errors: impl IntoIterator<Item = VideoTextureClockErrorState>,
    ) {
        for state in errors {
            upsert_video_clock_error(&mut self.pending_video_clock_errors, state);
        }
    }

    /// Computes whether a begin-frame send is allowed this tick.
    pub(crate) fn begin_frame_decision(
        &self,
        context: LockstepBeginFrameContext,
    ) -> BeginFrameDecision {
        decide_begin_frame(BeginFrameGateInput {
            init_finalized: context.init_finalized,
            fatal_error: context.fatal_error,
            ipc_connected: context.ipc_connected,
            last_frame_data_processed: self.last_frame_data_processed,
            pending_frame_submit_render: self.pending_frame_submit_render,
            renderer_decoupled: context.renderer_decoupled,
            last_frame_index: self.last_frame_index,
            sent_bootstrap_frame_start: self.sent_bootstrap_frame_start,
        })
    }

    /// Builds outgoing frame-start data plus the commit that should be applied after a successful send.
    pub(crate) fn build_frame_start(
        &self,
        inputs: InputState,
        performance: Option<PerformanceState>,
    ) -> (FrameStartData, BeginFrameCommit) {
        build_frame_start(BeginFrameBuildInput {
            last_frame_index: self.last_frame_index,
            sent_bootstrap_frame_start: self.sent_bootstrap_frame_start,
            performance,
            inputs,
            rendered_reflection_probes: self.pending_rendered_reflection_probes.clone(),
            video_clock_errors: self.pending_video_clock_errors.clone(),
        })
    }

    /// Applies the lock-step commit after a frame-start send has succeeded.
    pub(crate) fn commit_begin_frame_sent(&mut self, commit: BeginFrameCommit) {
        self.pending_rendered_reflection_probes.clear();
        self.pending_video_clock_errors.clear();
        self.last_frame_data_processed = false;
        if commit.mark_bootstrap_sent {
            self.sent_bootstrap_frame_start = true;
        }
    }
}

fn upsert_video_clock_error(
    pending: &mut Vec<VideoTextureClockErrorState>,
    state: VideoTextureClockErrorState,
) {
    if let Some(existing) = pending
        .iter_mut()
        .find(|existing| existing.asset_id == state.asset_id)
    {
        *existing = state;
    } else {
        pending.push(state);
    }
}

#[cfg(test)]
mod tests {
    use super::LockstepState;
    use crate::shared::memory_packer::MemoryPacker;
    use crate::shared::polymorphic_memory_packable_entity::PolymorphicEncode;
    use crate::shared::{InputState, RendererCommand, VideoTextureClockErrorState};

    const IPC_SEND_BUFFER_CAP: usize = 65_536;

    #[test]
    fn enqueue_video_clock_errors_keeps_latest_sample_per_asset() {
        let mut state = LockstepState::new(false);

        state.enqueue_video_clock_errors([
            VideoTextureClockErrorState {
                asset_id: 4,
                current_clock_error: 0.25,
            },
            VideoTextureClockErrorState {
                asset_id: 9,
                current_clock_error: -0.5,
            },
        ]);
        state.enqueue_video_clock_errors([VideoTextureClockErrorState {
            asset_id: 4,
            current_clock_error: 0.75,
        }]);

        let (frame_start, _) = state.build_frame_start(InputState::default(), None);

        assert_eq!(frame_start.video_clock_errors.len(), 2);
        assert_eq!(frame_start.video_clock_errors[0].asset_id, 4);
        assert_eq!(frame_start.video_clock_errors[0].current_clock_error, 0.75);
        assert_eq!(frame_start.video_clock_errors[1].asset_id, 9);
        assert_eq!(frame_start.video_clock_errors[1].current_clock_error, -0.5);
    }

    #[test]
    fn repeated_video_clock_error_retries_stay_within_ipc_send_buffer() {
        let mut state = LockstepState::new(false);
        for retry in 0..10_000 {
            state.enqueue_video_clock_errors([VideoTextureClockErrorState {
                asset_id: 4,
                current_clock_error: retry as f32,
            }]);
        }

        let (frame_start, _) = state.build_frame_start(InputState::default(), None);
        assert_eq!(frame_start.video_clock_errors.len(), 1);
        assert_eq!(frame_start.video_clock_errors[0].asset_id, 4);
        assert_eq!(
            frame_start.video_clock_errors[0].current_clock_error,
            9_999.0
        );

        let mut command = RendererCommand::FrameStartData(frame_start);
        let mut buffer = vec![0u8; IPC_SEND_BUFFER_CAP];
        let mut packer = MemoryPacker::new(&mut buffer);
        command.encode(&mut packer);

        assert!(
            !packer.had_overflow(),
            "repeated retries for one video asset must not overflow the IPC send buffer"
        );
    }
}
