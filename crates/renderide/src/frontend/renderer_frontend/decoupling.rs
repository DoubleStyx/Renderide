//! Decoupling-state methods on [`RendererFrontend`]. Logging for the
//! `Activate`/`Recouple`/`AdvanceProgress`/`ResetProgress` decisions lives in
//! [`super::super::decoupling::logging`] so this file stays focused on
//! delegation.

use std::time::{Duration, Instant};

use crate::shared::RenderDecouplingConfig;

#[cfg(test)]
use super::super::decoupling::DecouplingState;
use super::super::decoupling::decisions::DecouplingActivationDecision;
use super::super::decoupling::logging::log_activation;
use super::RendererFrontend;

impl RendererFrontend {
    /// Read-only handle to the host-driven decoupling state.
    #[cfg(test)]
    pub fn decoupling_state(&self) -> &DecouplingState {
        &self.decoupling
    }

    /// Whether the activation state machine has promoted into decoupled mode.
    #[cfg(test)]
    pub fn is_decoupled(&self) -> bool {
        self.decoupling.is_active()
    }

    /// Renderite-style decoupling predicate used by render and asset cadence.
    pub fn is_renderer_decoupled(&self) -> bool {
        !self.lockstep.host_lockstep_activated() || self.decoupling.is_active()
    }

    /// Asset-integration budget for the current Renderite-style decoupling mode.
    pub fn effective_asset_integration_budget_ms(&self, coupled_default_ms: u32) -> u32 {
        self.decoupling
            .effective_asset_integration_budget_ms_for_mode(
                coupled_default_ms,
                self.is_renderer_decoupled(),
            )
    }

    /// Replaces renderer-side decoupling thresholds with the host's config.
    pub fn set_decoupling_config(&mut self, cfg: RenderDecouplingConfig) {
        self.decoupling.apply_config(&cfg);
    }

    /// Per-tick decoupling activation check.
    pub fn update_decoupling_activation(&mut self, now: Instant) {
        let decision = self
            .decoupling
            .update_activation_for_tick(now, self.lockstep.awaiting_submit());
        if decision == DecouplingActivationDecision::Activate {
            log_activation(&self.decoupling, self.lockstep.last_frame_index());
        }
    }

    /// Bounded wait slice before the next decoupling activation check while a host submit is due.
    pub fn decoupling_activation_wait_timeout(
        &self,
        now: Instant,
        max_slice: Duration,
    ) -> Option<Duration> {
        self.decoupling
            .activation_wait_timeout(now, self.lockstep.awaiting_submit(), max_slice)
    }
}
