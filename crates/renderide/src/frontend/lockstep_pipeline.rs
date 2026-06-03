//! Pure host/renderer lock-step pipeline decisions.
//!
//! The renderer keeps one outstanding [`crate::shared::FrameStartData`] credit. When a submitted
//! host frame is already available, the preferred path spends that credit before rendering the
//! current submit so host simulation overlaps renderer work.

/// High-level pipeline action selected for the current frame phase.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum LockstepPipelineAction {
    /// No lock-step action has been recorded for this tick yet.
    #[default]
    None,
    /// Send the next host frame request before rendering the current submitted frame.
    SendEarlyNextFrame,
    /// Render the currently submitted frame without an early next-frame request.
    RenderCurrentSubmit,
    /// Send a host frame request after render, or from an idle lock-step phase.
    SendPostRender,
    /// Wait for an already-requested host frame submit.
    WaitForSubmit,
    /// No render or host request is currently possible.
    SkipIdle,
}

impl LockstepPipelineAction {
    /// Stable numeric code for Tracy plots.
    pub(crate) const fn plot_code(self) -> f64 {
        match self {
            Self::None => 0.0,
            Self::SendEarlyNextFrame => 1.0,
            Self::RenderCurrentSubmit => 2.0,
            Self::SendPostRender => 3.0,
            Self::WaitForSubmit => 4.0,
            Self::SkipIdle => 5.0,
        }
    }

    /// Short label for HUD rows and trace logs.
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::SendEarlyNextFrame => "early begin",
            Self::RenderCurrentSubmit => "render submit",
            Self::SendPostRender => "post begin",
            Self::WaitForSubmit => "wait submit",
            Self::SkipIdle => "idle",
        }
    }
}

/// Reason an early one-credit begin-frame could not be sent.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum OneCreditBlockReason {
    /// Early begin-frame was legal or has not been evaluated.
    #[default]
    None,
    /// Init/fatal/IPC state prevents any begin-frame send.
    BeginFrameBlocked,
    /// A host frame request is already outstanding.
    AwaitingSubmit,
    /// No processed host submit is waiting for a renderer-side draw attempt.
    NoPendingSubmitRender,
    /// Host-finalization-critical GPU completion work must drain first.
    SubmitCompletionWorkPending,
}

impl OneCreditBlockReason {
    /// Stable numeric code for Tracy plots.
    pub(crate) const fn plot_code(self) -> f64 {
        match self {
            Self::None => 0.0,
            Self::BeginFrameBlocked => 1.0,
            Self::AwaitingSubmit => 2.0,
            Self::NoPendingSubmitRender => 3.0,
            Self::SubmitCompletionWorkPending => 4.0,
        }
    }

    /// Short label for HUD rows and trace logs.
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::BeginFrameBlocked => "begin blocked",
            Self::AwaitingSubmit => "awaiting submit",
            Self::NoPendingSubmitRender => "no submitted frame",
            Self::SubmitCompletionWorkPending => "completion work",
        }
    }
}

/// Reason the runtime entered a host-submit wait fallback.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum HostWaitReason {
    /// No wait fallback is active.
    #[default]
    None,
    /// Desktop lock-step has no renderable submit yet.
    DesktopAwaitingSubmit,
    /// VR must not open an OpenXR frame until a renderable host submit or decoupling is available.
    XrBeforeFrame,
}

impl HostWaitReason {
    /// Stable numeric code for Tracy plots.
    pub(crate) const fn plot_code(self) -> f64 {
        match self {
            Self::None => 0.0,
            Self::DesktopAwaitingSubmit => 1.0,
            Self::XrBeforeFrame => 2.0,
        }
    }

    /// Short label for HUD rows and trace logs.
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::DesktopAwaitingSubmit => "desktop await",
            Self::XrBeforeFrame => "xr pre-frame",
        }
    }
}

/// Labels captured by the frame-timing HUD.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct LockstepPipelineHudLabels {
    /// Last high-level pipeline action.
    pub(crate) action: &'static str,
    /// Last early-send block reason.
    pub(crate) one_credit_block: &'static str,
    /// Last host-wait fallback reason.
    pub(crate) wait_reason: &'static str,
}

/// Inputs for the pure pipeline decision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct LockstepPipelineInput {
    /// Whether init, fatal, and IPC state allow any begin-frame send.
    pub(crate) begin_frame_base_allowed: bool,
    /// Whether a regular begin-frame send is legal in the current lock-step state.
    pub(crate) regular_begin_frame_allowed: bool,
    /// Whether a `FrameStartData` is already waiting for a matching submit.
    pub(crate) awaiting_frame_submit: bool,
    /// Whether a processed host submit still needs a renderer-side draw attempt.
    pub(crate) pending_frame_submit_render: bool,
    /// Whether the current tick may render world state.
    pub(crate) should_render_frame: bool,
    /// Whether host-finalization-critical completion work is drained.
    pub(crate) submit_completion_work_drained: bool,
}

/// Returns the selected pipeline action for the current frame phase.
pub(crate) fn decide_lockstep_pipeline(input: LockstepPipelineInput) -> LockstepPipelineAction {
    if one_credit_block_reason(input) == OneCreditBlockReason::None {
        return LockstepPipelineAction::SendEarlyNextFrame;
    }
    if input.should_render_frame {
        return LockstepPipelineAction::RenderCurrentSubmit;
    }
    if input.awaiting_frame_submit {
        return LockstepPipelineAction::WaitForSubmit;
    }
    if input.regular_begin_frame_allowed {
        return LockstepPipelineAction::SendPostRender;
    }
    LockstepPipelineAction::SkipIdle
}

/// Explains why a one-credit early begin-frame is blocked.
pub(crate) fn one_credit_block_reason(input: LockstepPipelineInput) -> OneCreditBlockReason {
    if !input.begin_frame_base_allowed {
        return OneCreditBlockReason::BeginFrameBlocked;
    }
    if input.awaiting_frame_submit {
        return OneCreditBlockReason::AwaitingSubmit;
    }
    if !input.pending_frame_submit_render {
        return OneCreditBlockReason::NoPendingSubmitRender;
    }
    if !input.submit_completion_work_drained {
        return OneCreditBlockReason::SubmitCompletionWorkPending;
    }
    OneCreditBlockReason::None
}

#[cfg(test)]
mod tests {
    use super::{
        HostWaitReason, LockstepPipelineAction, LockstepPipelineHudLabels, LockstepPipelineInput,
        OneCreditBlockReason, decide_lockstep_pipeline, one_credit_block_reason,
    };

    fn renderable_submit() -> LockstepPipelineInput {
        LockstepPipelineInput {
            begin_frame_base_allowed: true,
            regular_begin_frame_allowed: true,
            awaiting_frame_submit: false,
            pending_frame_submit_render: true,
            should_render_frame: true,
            submit_completion_work_drained: true,
        }
    }

    #[test]
    fn renderer_slower_than_host_sends_early_next_frame() {
        assert_eq!(
            decide_lockstep_pipeline(renderable_submit()),
            LockstepPipelineAction::SendEarlyNextFrame
        );
        assert_eq!(
            one_credit_block_reason(renderable_submit()),
            OneCreditBlockReason::None
        );
    }

    #[test]
    fn already_awaiting_submit_waits_when_nothing_renderable() {
        let input = LockstepPipelineInput {
            awaiting_frame_submit: true,
            pending_frame_submit_render: false,
            should_render_frame: false,
            ..renderable_submit()
        };
        assert_eq!(
            decide_lockstep_pipeline(input),
            LockstepPipelineAction::WaitForSubmit
        );
        assert_eq!(
            one_credit_block_reason(input),
            OneCreditBlockReason::AwaitingSubmit
        );
    }

    #[test]
    fn duplicate_in_flight_request_is_blocked_but_current_submit_can_render() {
        let input = LockstepPipelineInput {
            awaiting_frame_submit: true,
            ..renderable_submit()
        };
        assert_eq!(
            decide_lockstep_pipeline(input),
            LockstepPipelineAction::RenderCurrentSubmit
        );
        assert_eq!(
            one_credit_block_reason(input),
            OneCreditBlockReason::AwaitingSubmit
        );
    }

    #[test]
    fn completion_work_blocks_early_send_but_not_current_render() {
        let input = LockstepPipelineInput {
            submit_completion_work_drained: false,
            ..renderable_submit()
        };
        assert_eq!(
            decide_lockstep_pipeline(input),
            LockstepPipelineAction::RenderCurrentSubmit
        );
        assert_eq!(
            one_credit_block_reason(input),
            OneCreditBlockReason::SubmitCompletionWorkPending
        );
    }

    #[test]
    fn decoupled_stale_rendering_keeps_rendering_without_early_send() {
        let input = LockstepPipelineInput {
            awaiting_frame_submit: true,
            pending_frame_submit_render: false,
            should_render_frame: true,
            ..renderable_submit()
        };
        assert_eq!(
            decide_lockstep_pipeline(input),
            LockstepPipelineAction::RenderCurrentSubmit
        );
    }

    #[test]
    fn post_render_or_idle_begin_is_selected_when_no_submit_is_pending() {
        let input = LockstepPipelineInput {
            pending_frame_submit_render: false,
            should_render_frame: false,
            ..renderable_submit()
        };
        assert_eq!(
            decide_lockstep_pipeline(input),
            LockstepPipelineAction::SendPostRender
        );
        assert_eq!(
            one_credit_block_reason(input),
            OneCreditBlockReason::NoPendingSubmitRender
        );
    }

    #[test]
    fn blocked_idle_state_skips() {
        let input = LockstepPipelineInput {
            begin_frame_base_allowed: false,
            regular_begin_frame_allowed: false,
            awaiting_frame_submit: false,
            pending_frame_submit_render: false,
            should_render_frame: false,
            ..renderable_submit()
        };
        assert_eq!(
            decide_lockstep_pipeline(input),
            LockstepPipelineAction::SkipIdle
        );
    }

    #[test]
    fn labels_are_stable_for_hud_defaults() {
        let labels = LockstepPipelineHudLabels {
            action: LockstepPipelineAction::SendEarlyNextFrame.label(),
            one_credit_block: OneCreditBlockReason::None.label(),
            wait_reason: HostWaitReason::None.label(),
        };
        assert_eq!(labels.action, "early begin");
        assert_eq!(labels.one_credit_block, "none");
        assert_eq!(labels.wait_reason, "none");
    }
}
