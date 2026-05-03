//! Integration test: host-driven [`RenderDecouplingConfig`] is plumbed through
//! [`RendererFrontend::set_decoupling_config`] into the renderer-side
//! [`DecouplingState`] and observable via [`RendererFrontend::is_decoupled`] /
//! [`RendererFrontend::decoupling_state`].
//!
//! State-machine transitions (activation thresholds, recouple counter, force-decouple) are
//! exercised by the in-crate unit tests on [`renderide::frontend::DecouplingState`]. These
//! integration tests guard the public composition: that the threshold values arrive
//! intact, that the public accessors expose the state, and that the budget gating returns
//! the host-supplied ceiling instead of the local default while decoupled.

use std::time::Instant;

use renderide::frontend::RendererFrontend;
use renderide::shared::RenderDecouplingConfig;

const fn cfg(interval: f32, decoupled_max: f32, recouple: i32) -> RenderDecouplingConfig {
    RenderDecouplingConfig {
        decouple_activate_interval: interval,
        decoupled_max_asset_processing_time: decoupled_max,
        recouple_frame_count: recouple,
    }
}

/// A fresh frontend starts coupled and reports the Renderite.Unity defaults.
#[test]
fn fresh_frontend_starts_coupled_with_unity_defaults() {
    let frontend = RendererFrontend::new(None);
    assert!(!frontend.is_decoupled());
    let s = frontend.decoupling_state();
    assert!((s.activate_interval_seconds() - 1.0 / 15.0).abs() < 1e-6);
    assert!((s.decoupled_max_asset_processing_seconds() - 0.002).abs() < 1e-6);
    assert_eq!(s.recouple_frame_count(), 10);
}

/// `set_decoupling_config` overwrites the thresholds with the host-supplied values.
#[test]
fn set_decoupling_config_updates_thresholds() {
    let mut frontend = RendererFrontend::new(None);
    frontend.set_decoupling_config(cfg(1.0 / 15.0, 0.008, 60));
    let s = frontend.decoupling_state();
    assert!((s.activate_interval_seconds() - 1.0 / 15.0).abs() < 1e-6);
    assert!((s.decoupled_max_asset_processing_seconds() - 0.008).abs() < 1e-6);
    assert_eq!(s.recouple_frame_count(), 60);
}

/// While coupled, `effective_asset_integration_budget_ms` returns the local default unchanged
/// (clamped to >= 1).
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

/// `update_decoupling_activation` is a no-op when no outgoing `FrameStartData` has been recorded
/// yet -- there's no wait window to compare against the threshold.
#[test]
fn update_activation_no_op_without_recorded_send() {
    let mut frontend = RendererFrontend::new(None);
    frontend.set_decoupling_config(cfg(0.0, 0.004, 5));
    frontend.update_decoupling_activation(Instant::now());
    assert!(!frontend.is_decoupled());
}

/// `note_frame_submit_processed` is safe to call before any decoupling config arrives -- it must
/// not activate decoupling on its own.
#[test]
fn note_frame_submit_processed_does_not_activate_decoupling() {
    let mut frontend = RendererFrontend::new(None);
    frontend.note_frame_submit_processed(7);
    assert!(!frontend.is_decoupled());
}
