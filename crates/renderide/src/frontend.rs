//! Host transport and session layer: IPC queues, shared memory, init handshake, lock-step gating.
//!
//! Layout: [`renderer_frontend`] ([`RendererFrontend`]) composes small transport, session,
//! lock-step, performance, output-policy, and decoupling components; [`dispatch`] owns IPC command
//! classification/routing; [`input`] adapts winit/XR snapshots into [`crate::shared::InputState`].
//!
//! [`RendererFrontend`] is the side-effect facade for queue and shared-memory access. Pure
//! decisions such as begin-frame gating, init routing, output policy, and decoupling transitions
//! live in their domain modules and are applied by the facade/runtime.

mod begin_frame;
mod decoupling;
pub(crate) mod dispatch;
mod frame_start_performance;
mod init_state;
mod lockstep_state;
pub(crate) mod output_device;
mod output_policy;
mod render_cadence;
mod renderer_frontend;
mod session;
mod transport;

/// Winit adapter and [`WindowInputAccumulator`](input::WindowInputAccumulator) for [`crate::shared::InputState`].
pub mod input;

pub(crate) use frame_start_performance::AssetIntegrationPerformanceSample;
pub use init_state::InitState;
pub use renderer_frontend::RendererFrontend;

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use crate::connection::ConnectionParams;
    use crate::shared::RenderDecouplingConfig;

    use super::{InitState, RendererFrontend};

    const fn cfg(interval: f32, decoupled_max: f32, recouple: i32) -> RenderDecouplingConfig {
        RenderDecouplingConfig {
            decouple_activate_interval: interval,
            decoupled_max_asset_processing_time: decoupled_max,
            recouple_frame_count: recouple,
        }
    }

    #[test]
    fn standalone_frontend_starts_finalized() {
        let frontend = RendererFrontend::new(None);
        assert_eq!(frontend.init_state(), InitState::Finalized);
        assert!(frontend.last_frame_data_processed());
        assert_eq!(frontend.last_frame_index(), -1);
        assert!(!frontend.shutdown_requested());
        assert!(!frontend.fatal_error());
        assert!(frontend.pending_init().is_none());
        assert!(!frontend.should_send_begin_frame());
    }

    #[test]
    fn standalone_frontend_mutators_update_exit_state() {
        let mut frontend = RendererFrontend::new(None);

        frontend.note_frame_submit_processed(7);
        assert_eq!(frontend.last_frame_index(), 7);
        assert!(frontend.last_frame_data_processed());

        frontend.set_fatal_error(true);
        assert!(frontend.fatal_error());
        frontend.set_fatal_error(false);
        assert!(!frontend.fatal_error());

        frontend.set_shutdown_requested(true);
        assert!(frontend.shutdown_requested());
    }

    #[test]
    fn frontend_decoupling_config_updates_thresholds() {
        let mut frontend = RendererFrontend::new(None);
        frontend.set_decoupling_config(cfg(1.0 / 15.0, 0.008, 60));

        let state = frontend.decoupling_state();
        assert!((state.activate_interval_seconds() - 1.0 / 15.0).abs() < 1e-6);
        assert!((state.decoupled_max_asset_processing_seconds() - 0.008).abs() < 1e-6);
        assert_eq!(state.recouple_frame_count(), 60);
        assert!(!frontend.is_decoupled());
    }

    #[test]
    fn frontend_decoupling_activation_needs_recorded_send() {
        let mut frontend = RendererFrontend::new(None);
        frontend.set_decoupling_config(cfg(0.0, 0.004, 5));

        frontend.update_decoupling_activation(Instant::now());
        assert!(!frontend.is_decoupled());

        frontend.note_frame_submit_processed(7);
        assert!(!frontend.is_decoupled());
    }

    #[test]
    fn renderer_engine_ready_enables_strict_lockstep_predicate() {
        let mut frontend = RendererFrontend::new(Some(ConnectionParams {
            queue_name: "frontend_decoupling_gate_test".into(),
            queue_capacity: crate::connection::DEFAULT_QUEUE_CAPACITY,
        }));

        assert!(!frontend.host_lockstep_activated());
        assert!(frontend.is_renderer_decoupled());

        frontend.on_renderer_engine_ready();

        assert!(frontend.host_lockstep_activated());
        assert!(!frontend.is_renderer_decoupled());
    }

    #[test]
    fn frame_submit_pending_render_clears_after_render_attempt() {
        let mut frontend = RendererFrontend::new(None);

        frontend.note_frame_submit_processed(7);
        assert!(frontend.pending_frame_submit_render());

        frontend.note_frame_render_attempted();
        assert!(!frontend.pending_frame_submit_render());
    }

    #[test]
    fn coupled_budget_returns_local_default() {
        let frontend = RendererFrontend::new(None);
        assert_eq!(
            frontend
                .decoupling_state()
                .effective_asset_integration_budget_ms(8),
            8
        );
        assert_eq!(
            frontend
                .decoupling_state()
                .effective_asset_integration_budget_ms(0),
            1
        );
    }
}
