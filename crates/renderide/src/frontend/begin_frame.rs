//! Pure transitions for outgoing [`crate::shared::FrameStartData`] lock-step sends.
//!
//! [`RendererFrontend::should_send_begin_frame`](crate::frontend::RendererFrontend::should_send_begin_frame)
//! delegates here so the lock-step rules are unit-testable without constructing a full frontend.

use crate::shared::{
    FrameStartData, InputState, PerformanceState, ReflectionProbeChangeRenderResult,
    VideoTextureClockErrorState,
};

/// Inputs to the pure begin-frame gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BeginFrameGateInput {
    /// Whether the init handshake has finalized.
    pub(crate) init_finalized: bool,
    /// Whether a fatal frontend error suppresses lock-step.
    pub(crate) fatal_error: bool,
    /// Whether IPC is connected.
    pub(crate) ipc_connected: bool,
    /// Whether the previous host frame submit was processed.
    pub(crate) last_frame_data_processed: bool,
    /// Whether that processed host frame has not yet been rendered.
    pub(crate) pending_frame_submit_render: bool,
    /// Whether the renderer is currently allowed to render stale scene state.
    pub(crate) renderer_decoupled: bool,
    /// Host frame index echoed to the host.
    pub(crate) last_frame_index: i32,
    /// Whether the initial bootstrap frame-start was already sent.
    pub(crate) sent_bootstrap_frame_start: bool,
}

/// Pure decision returned by the begin-frame gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BeginFrameDecision {
    /// Whether the renderer may send a frame-start this tick.
    pub(crate) allowed: bool,
    /// Whether the allowed send is the first bootstrap send.
    pub(crate) bootstrap: bool,
}

impl BeginFrameDecision {
    /// Returns `true` when the renderer may send a frame-start.
    pub(crate) fn is_allowed(self) -> bool {
        self.allowed
    }
}

/// Inputs to build an outgoing frame-start payload.
pub(crate) struct BeginFrameBuildInput {
    /// Host frame index echoed to the host.
    pub(crate) last_frame_index: i32,
    /// Whether the initial bootstrap frame-start was already sent.
    pub(crate) sent_bootstrap_frame_start: bool,
    /// Optional performance payload for the host.
    pub(crate) performance: Option<PerformanceState>,
    /// Input snapshot captured for this frame-start.
    pub(crate) inputs: InputState,
    /// Reflection probes completed since the previous successful frame-start send.
    pub(crate) rendered_reflection_probes: Vec<ReflectionProbeChangeRenderResult>,
    /// Latest video clock-error samples captured since the previous successful frame-start send.
    pub(crate) video_clock_errors: Vec<VideoTextureClockErrorState>,
}

/// State mutation to apply after the payload is successfully enqueued to the primary IPC queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BeginFrameCommit {
    /// Whether this send should mark the bootstrap frame-start as sent.
    pub(crate) mark_bootstrap_sent: bool,
}

/// Computes whether a frame-start send is allowed and why.
pub(crate) fn decide_begin_frame(input: BeginFrameGateInput) -> BeginFrameDecision {
    if !input.init_finalized || input.fatal_error || !input.ipc_connected {
        return BeginFrameDecision {
            allowed: false,
            bootstrap: false,
        };
    }
    let bootstrap = input.last_frame_index < 0 && !input.sent_bootstrap_frame_start;
    let processed_frame_allows_send = input.last_frame_data_processed
        && (input.renderer_decoupled || !input.pending_frame_submit_render);
    BeginFrameDecision {
        allowed: processed_frame_allows_send || bootstrap,
        bootstrap,
    }
}

/// Builds the outgoing frame-start payload and the commit needed after a successful send.
pub(crate) fn build_frame_start(input: BeginFrameBuildInput) -> (FrameStartData, BeginFrameCommit) {
    let bootstrap = input.last_frame_index < 0 && !input.sent_bootstrap_frame_start;
    (
        FrameStartData {
            last_frame_index: input.last_frame_index,
            performance: input.performance,
            inputs: Some(input.inputs),
            rendered_reflection_probes: input.rendered_reflection_probes,
            video_clock_errors: input.video_clock_errors,
        },
        BeginFrameCommit {
            mark_bootstrap_sent: bootstrap,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::{BeginFrameBuildInput, BeginFrameGateInput, build_frame_start, decide_begin_frame};
    use crate::shared::InputState;

    fn finalized_processed_gate() -> BeginFrameGateInput {
        BeginFrameGateInput {
            init_finalized: true,
            fatal_error: false,
            ipc_connected: true,
            last_frame_data_processed: true,
            pending_frame_submit_render: false,
            renderer_decoupled: false,
            last_frame_index: 0,
            sent_bootstrap_frame_start: true,
        }
    }

    #[test]
    fn not_finalized_blocks() {
        assert!(
            !decide_begin_frame(BeginFrameGateInput {
                init_finalized: false,
                ..finalized_processed_gate()
            })
            .is_allowed()
        );
    }

    #[test]
    fn fatal_blocks() {
        assert!(
            !decide_begin_frame(BeginFrameGateInput {
                fatal_error: true,
                ..finalized_processed_gate()
            })
            .is_allowed()
        );
    }

    #[test]
    fn no_ipc_blocks() {
        assert!(
            !decide_begin_frame(BeginFrameGateInput {
                ipc_connected: false,
                ..finalized_processed_gate()
            })
            .is_allowed()
        );
    }

    #[test]
    fn finalized_ipc_processed_allows() {
        assert!(
            decide_begin_frame(BeginFrameGateInput {
                last_frame_index: 5,
                ..finalized_processed_gate()
            })
            .is_allowed()
        );
    }

    #[test]
    fn processed_unrendered_frame_blocks_while_coupled() {
        assert!(
            !decide_begin_frame(BeginFrameGateInput {
                pending_frame_submit_render: true,
                renderer_decoupled: false,
                ..finalized_processed_gate()
            })
            .is_allowed()
        );
    }

    #[test]
    fn processed_unrendered_frame_allows_while_decoupled() {
        assert!(
            decide_begin_frame(BeginFrameGateInput {
                pending_frame_submit_render: true,
                renderer_decoupled: true,
                ..finalized_processed_gate()
            })
            .is_allowed()
        );
    }

    #[test]
    fn bootstrap_before_first_submit_allows_without_processed_flag() {
        let decision = decide_begin_frame(BeginFrameGateInput {
            last_frame_data_processed: false,
            last_frame_index: -1,
            sent_bootstrap_frame_start: false,
            ..finalized_processed_gate()
        });
        assert!(decision.allowed);
        assert!(decision.bootstrap);
    }

    #[test]
    fn after_bootstrap_without_new_submit_blocks() {
        assert!(
            !decide_begin_frame(BeginFrameGateInput {
                last_frame_data_processed: false,
                last_frame_index: -1,
                sent_bootstrap_frame_start: true,
                ..finalized_processed_gate()
            })
            .is_allowed()
        );
    }

    #[test]
    fn positive_frame_index_without_processed_blocks_unless_bootstrap() {
        assert!(
            !decide_begin_frame(BeginFrameGateInput {
                last_frame_data_processed: false,
                last_frame_index: 3,
                ..finalized_processed_gate()
            })
            .is_allowed()
        );
    }

    #[test]
    fn build_frame_start_carries_payloads_and_marks_bootstrap() {
        let (frame_start, commit) = build_frame_start(BeginFrameBuildInput {
            last_frame_index: -1,
            sent_bootstrap_frame_start: false,
            performance: None,
            inputs: InputState::default(),
            rendered_reflection_probes: Vec::new(),
            video_clock_errors: Vec::new(),
        });
        assert_eq!(frame_start.last_frame_index, -1);
        assert!(frame_start.inputs.is_some());
        assert!(commit.mark_bootstrap_sent);
    }
}
