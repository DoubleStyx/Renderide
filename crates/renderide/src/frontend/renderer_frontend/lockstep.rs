//! Frame-cadence methods on [`RendererFrontend`]: begin-frame gating, the
//! `pre_frame` lock-step send, and bookkeeping for received host submits.

use std::time::Instant;

use crate::shared::{
    InputState, ReflectionProbeChangeRenderResult, RendererCommand, VideoTextureClockErrorState,
};

use super::super::decoupling::logging::log_submit_decision;
use super::super::lockstep_state::LockstepBeginFrameContext;
use super::super::render_cadence::{
    RenderCadenceDecision, RenderCadenceInput, decide_render_cadence,
};
use super::RendererFrontend;

impl RendererFrontend {
    /// Lock-step: last host frame index echoed in outgoing [`crate::shared::FrameStartData`].
    pub fn last_frame_index(&self) -> i32 {
        self.lockstep.last_frame_index()
    }

    /// Whether the last [`crate::shared::FrameSubmitData`] was applied and another begin-frame may follow.
    #[cfg(test)]
    pub fn last_frame_data_processed(&self) -> bool {
        self.lockstep.last_frame_data_processed()
    }

    /// Whether a [`crate::shared::FrameStartData`] should be sent this tick.
    pub fn should_send_begin_frame(&self) -> bool {
        self.lockstep
            .begin_frame_decision(LockstepBeginFrameContext {
                init_finalized: self.session.init_state().is_finalized(),
                fatal_error: self.session.fatal_error(),
                ipc_connected: self.transport.is_ipc_connected(),
                renderer_decoupled: self.is_renderer_decoupled(),
            })
            .is_allowed()
    }

    /// Whether the renderer is waiting for the host's next [`crate::shared::FrameSubmitData`].
    pub fn awaiting_frame_submit(&self) -> bool {
        self.lockstep.awaiting_submit()
    }

    /// Whether the host has enabled regular lockstep through `RendererEngineReady`.
    #[cfg(test)]
    pub fn host_lockstep_activated(&self) -> bool {
        self.lockstep.host_lockstep_activated()
    }

    /// Whether a processed host submit still needs a renderer-side draw attempt.
    #[cfg(test)]
    pub fn pending_frame_submit_render(&self) -> bool {
        self.lockstep.pending_frame_submit_render()
    }

    /// Pure render-cadence decision for the current frontend state.
    pub(crate) fn render_cadence_decision(&self) -> RenderCadenceDecision {
        decide_render_cadence(RenderCadenceInput {
            standalone: self.transport.is_standalone(),
            host_lockstep_activated: self.lockstep.host_lockstep_activated(),
            renderer_decoupled: self.is_renderer_decoupled(),
            awaiting_frame_submit: self.lockstep.awaiting_submit(),
            pending_frame_submit_render: self.lockstep.pending_frame_submit_render(),
        })
    }

    /// Whether the current tick may render world state.
    pub fn should_render_frame(&self) -> bool {
        self.render_cadence_decision().should_render()
    }

    /// Marks any pending processed host submit as rendered by this process.
    pub fn note_frame_render_attempted(&mut self) {
        self.lockstep.note_frame_render_attempted();
    }

    /// Appends reflection-probe render completion rows for the next outgoing frame-start.
    pub fn enqueue_rendered_reflection_probes(
        &mut self,
        probes: impl IntoIterator<Item = ReflectionProbeChangeRenderResult>,
    ) {
        self.lockstep.enqueue_rendered_reflection_probes(probes);
    }

    /// Records latest video texture clock-error samples for the next outgoing frame-start.
    pub fn enqueue_video_clock_errors(
        &mut self,
        errors: impl IntoIterator<Item = VideoTextureClockErrorState>,
    ) {
        self.lockstep.enqueue_video_clock_errors(errors);
    }

    /// Lock-step begin-frame: sends frame-start data with `inputs` when allowed.
    pub fn pre_frame(&mut self, inputs: InputState) {
        profiling::scope!("frontend::pre_frame_send");
        if !self.should_send_begin_frame() {
            return;
        }

        let performance = self.performance.step_for_frame_start();
        let (frame_start, commit) = self.lockstep.build_frame_start(inputs, performance);
        if let Some(ipc) = self.transport.ipc_mut()
            && !ipc.send_primary(RendererCommand::FrameStartData(frame_start))
        {
            logger::warn!(
                "IPC primary queue full: FrameStartData not sent; will retry on the next tick"
            );
            return;
        }
        self.lockstep.commit_begin_frame_sent(commit);
        self.decoupling.record_frame_start_sent(Instant::now());
    }

    /// Updates lock-step state after the host submits a frame.
    pub fn note_frame_submit_processed(&mut self, frame_index: i32) {
        self.lockstep.note_frame_submit_processed(frame_index);
        let decision = self.decoupling.record_frame_submit_received(Instant::now());
        log_submit_decision(decision, &self.decoupling, frame_index);
    }
}
