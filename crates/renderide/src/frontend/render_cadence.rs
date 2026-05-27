//! Pure render-cadence decisions for host lockstep and renderer decoupling.

/// Inputs needed to decide whether the current tick may render world state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RenderCadenceInput {
    /// Whether the renderer was started without a host IPC connection.
    pub(crate) standalone: bool,
    /// Whether the host has sent `RendererEngineReady` and lockstep gating is active.
    pub(crate) host_lockstep_activated: bool,
    /// Whether Renderite-style decoupling currently permits stale-scene rendering.
    pub(crate) renderer_decoupled: bool,
    /// Whether a `FrameStartData` has been sent and the matching host submit is still outstanding.
    pub(crate) awaiting_frame_submit: bool,
    /// Whether a host submit was applied and has not yet been rendered by this process.
    pub(crate) pending_frame_submit_render: bool,
}

/// Render-cadence decision for one renderer tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RenderCadenceDecision {
    /// The tick may render world state.
    Render,
    /// The tick must not render because the host has not submitted the requested frame yet.
    SkipAwaitingSubmit,
    /// The tick must not render because lockstep is idle and no submitted frame is pending.
    SkipIdleLockstep,
}

impl RenderCadenceDecision {
    /// Whether this decision allows the tick to render world state.
    pub(crate) const fn should_render(self) -> bool {
        matches!(self, Self::Render)
    }
}

/// Decides whether the current tick may render world state.
pub(crate) fn decide_render_cadence(input: RenderCadenceInput) -> RenderCadenceDecision {
    if input.standalone
        || !input.host_lockstep_activated
        || input.renderer_decoupled
        || input.pending_frame_submit_render
    {
        return RenderCadenceDecision::Render;
    }

    if input.awaiting_frame_submit {
        RenderCadenceDecision::SkipAwaitingSubmit
    } else {
        RenderCadenceDecision::SkipIdleLockstep
    }
}

#[cfg(test)]
mod tests {
    use super::{RenderCadenceDecision, RenderCadenceInput, decide_render_cadence};

    fn active_lockstep() -> RenderCadenceInput {
        RenderCadenceInput {
            standalone: false,
            host_lockstep_activated: true,
            renderer_decoupled: false,
            awaiting_frame_submit: false,
            pending_frame_submit_render: false,
        }
    }

    #[test]
    fn active_lockstep_awaiting_submit_skips_until_decoupled() {
        assert_eq!(
            decide_render_cadence(RenderCadenceInput {
                awaiting_frame_submit: true,
                ..active_lockstep()
            }),
            RenderCadenceDecision::SkipAwaitingSubmit
        );
    }

    #[test]
    fn decoupled_lockstep_allows_stale_rendering() {
        assert_eq!(
            decide_render_cadence(RenderCadenceInput {
                renderer_decoupled: true,
                awaiting_frame_submit: true,
                ..active_lockstep()
            }),
            RenderCadenceDecision::Render
        );
    }

    #[test]
    fn pre_engine_ready_renders_like_renderite_unity() {
        assert_eq!(
            decide_render_cadence(RenderCadenceInput {
                host_lockstep_activated: false,
                awaiting_frame_submit: true,
                ..active_lockstep()
            }),
            RenderCadenceDecision::Render
        );
    }

    #[test]
    fn pending_submit_render_allows_one_coupled_render() {
        assert_eq!(
            decide_render_cadence(RenderCadenceInput {
                pending_frame_submit_render: true,
                ..active_lockstep()
            }),
            RenderCadenceDecision::Render
        );
    }

    #[test]
    fn active_lockstep_without_pending_work_skips_idle_rendering() {
        assert_eq!(
            decide_render_cadence(active_lockstep()),
            RenderCadenceDecision::SkipIdleLockstep
        );
    }

    #[test]
    fn standalone_renderer_always_renders() {
        assert_eq!(
            decide_render_cadence(RenderCadenceInput {
                standalone: true,
                ..active_lockstep()
            }),
            RenderCadenceDecision::Render
        );
    }
}
