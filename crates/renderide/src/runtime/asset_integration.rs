//! Cooperative asset-integration phase on [`super::RendererRuntime`].
//!
//! Wraps [`crate::backend::RenderBackend::drain_asset_tasks`] in the runtime's once-per-tick
//! gate and the budget computation that switches between the
//! [`crate::config::RenderingSettings::asset_integration_budget_ms`] coupled default and the
//! decoupled-mode ceiling supplied by the host.

use std::time::{Duration, Instant};

use super::RendererRuntime;

impl RendererRuntime {
    /// Bounded cooperative mesh/texture asset integration.
    ///
    /// Uses [`crate::config::RenderingSettings::asset_integration_budget_ms`] for the wall-clock
    /// slice while coupled to host lock-step. While decoupled, the host-supplied
    /// [`crate::frontend::DecouplingState::decoupled_max_asset_processing_seconds`] ceiling
    /// replaces the local default so the renderer stays responsive while the host catches up.
    ///
    /// At most once per winit tick: a second call in the same tick is a no-op
    /// ([`Self::did_integrate_assets_this_tick`]).
    pub fn run_asset_integration(&mut self) {
        profiling::scope!("tick::asset_integration_runtime");
        if self.did_integrate_this_tick {
            return;
        }
        let coupled_default_ms = self
            .settings
            .read()
            .map(|s| s.rendering.asset_integration_budget_ms)
            .unwrap_or(3);
        let budget_ms = self
            .frontend
            .decoupling_state()
            .effective_asset_integration_budget_ms(coupled_default_ms);
        let deadline = Instant::now() + Duration::from_millis(u64::from(budget_ms));
        let (shm, ipc) = self.frontend.transport_pair_mut();
        let Some(shm) = shm else {
            return;
        };
        let mut ipc_opt = ipc;
        let summary = self.backend.drain_asset_tasks(shm, &mut ipc_opt, deadline);
        trace_asset_integration_summary(budget_ms, summary);
        self.did_integrate_this_tick = true;
    }

    /// Whether [`Self::run_asset_integration`] already ran this tick.
    pub fn did_integrate_assets_this_tick(&self) -> bool {
        self.did_integrate_this_tick
    }
}

fn trace_asset_integration_summary(
    budget_ms: u32,
    summary: crate::assets::asset_transfer_queue::AssetIntegrationDrainSummary,
) {
    if summary.total_before() == 0
        && summary.total_after() == 0
        && !summary.budget_exhausted()
        && summary.gpu_ready
    {
        return;
    }
    logger::trace!(
        "asset integration: budget_ms={} gpu_ready={} elapsed_ms={:.3} high {}->{} normal {}->{} exhausted_high={} exhausted_normal={}",
        budget_ms,
        summary.gpu_ready,
        summary.elapsed.as_secs_f64() * 1000.0,
        summary.high_priority_before,
        summary.high_priority_after,
        summary.normal_priority_before,
        summary.normal_priority_after,
        summary.high_priority_budget_exhausted,
        summary.normal_priority_budget_exhausted,
    );
}
