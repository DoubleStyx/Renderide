//! Cooperative asset-integration phase on [`super::RendererRuntime`].
//!
//! Wraps [`crate::backend::RenderBackend::drain_asset_tasks`] in the runtime's redraw-tick
//! gate, the host-wait idle drain, and the budget computation that switches between the
//! [`crate::config::RenderingSettings::asset_integration_budget_ms`] coupled default and the
//! decoupled-mode ceiling supplied by the host.

use std::time::{Duration, Instant};

use super::{RendererRuntime, ipc::shader_material};
use crate::diagnostics::log_throttle::LogThrottle;
use crate::gpu::GpuQueueAccessMode;

static ASSET_INTEGRATION_NO_SHM_LOG: LogThrottle = LogThrottle::new();
static ASSET_INTEGRATION_BUDGET_LOG: LogThrottle = LogThrottle::new();
static ASSET_INTEGRATION_GPU_NOT_READY_LOG: LogThrottle = LogThrottle::new();

impl RendererRuntime {
    /// Bounded cooperative mesh/texture asset integration.
    ///
    /// Uses [`crate::config::RenderingSettings::asset_integration_budget_ms`] for the wall-clock
    /// slice while coupled to host lock-step. While decoupled, the host-supplied
    /// [`crate::frontend::DecouplingState::decoupled_max_asset_processing_seconds`] ceiling
    /// replaces the local default so the renderer stays responsive while the host catches up.
    ///
    /// At most once per redraw tick: a second redraw-phase call in the same tick is a no-op
    /// ([`Self::did_integrate_assets_this_tick`]). The app driver may still call
    /// [`Self::run_asset_integration_while_waiting_for_submit`] between redraws while a host
    /// frame submit is outstanding.
    pub fn run_asset_integration(&mut self) {
        profiling::scope!("tick::asset_integration_runtime");
        if self.tick_state.did_integrate_assets_this_tick() {
            return;
        }
        let Some(summary) = self.run_asset_integration_pass(GpuQueueAccessMode::Blocking) else {
            return;
        };
        trace_asset_integration_summary(self.asset_integration_budget_ms(), summary);
        self.record_asset_integration_summary(summary, 0);
        self.tick_state.mark_integrated_assets_this_tick();
    }

    /// Runs an extra asset-integration slice while the renderer is waiting for a host frame submit.
    ///
    /// Returns `true` when more asset work remains queued after the slice and another idle pass
    /// would be useful.
    pub fn run_asset_integration_while_waiting_for_submit(&mut self, now: Instant) -> bool {
        profiling::scope!("tick::asset_integration_host_wait");
        self.frontend.update_decoupling_activation(now);
        let awaiting_frame_submit = self.frontend.awaiting_frame_submit();
        if awaiting_frame_submit {
            self.drain_pending_shader_resolutions_for_asset_integration();
        }
        if !awaiting_frame_submit || !self.backend.has_pending_asset_work() {
            if awaiting_frame_submit {
                self.frontend.record_asset_integration_handle_wait();
            }
            return false;
        }
        if self.frontend.shared_memory().is_none() {
            self.frontend.record_asset_integration_handle_wait();
            if let Some(occurrence) = ASSET_INTEGRATION_NO_SHM_LOG.should_log(4, 128) {
                logger::warn!(
                    "asset integration skipped while waiting for host frame submit: shared memory unavailable occurrence={occurrence}"
                );
            }
            return false;
        }
        let Some(summary) = self.run_asset_integration_pass(GpuQueueAccessMode::NonBlocking) else {
            return false;
        };
        let budget_ms = self.asset_integration_budget_ms();
        trace_asset_integration_summary(budget_ms, summary);
        self.record_asset_integration_summary(summary, 0);
        let made_non_gpu_progress =
            summary.processed_main_tasks > 0 || summary.processed_particle_tasks > 0;
        self.backend.has_pending_asset_work() && (summary.gpu_ready || made_non_gpu_progress)
    }

    /// Drains completed asynchronous shader uploads before an asset-integration slice.
    pub(crate) fn drain_pending_shader_resolutions_for_asset_integration(&mut self) {
        profiling::scope!("tick::asset_integration_shader_resolution");
        shader_material::drain_pending_shader_resolutions(
            &mut self.ipc_state.pending_shader_resolutions,
            &mut self.backend,
            &mut self.frontend,
        );
    }

    /// Integrates assets decoded during a primary-queue wait before rendering the submitted frame.
    pub(crate) fn run_asset_integration_after_wait_poll(&mut self) {
        profiling::scope!("tick::asset_integration_after_wait_poll");
        self.drain_pending_shader_resolutions_for_asset_integration();
        if !self.backend.has_pending_asset_work() {
            return;
        }
        let Some(summary) = self.run_asset_integration_pass(GpuQueueAccessMode::NonBlocking) else {
            return;
        };
        trace_asset_integration_summary(self.asset_integration_budget_ms(), summary);
        self.record_asset_integration_summary(summary, 0);
    }

    fn asset_integration_budget_ms(&self) -> u32 {
        let coupled_default_ms = self
            .config
            .settings
            .read()
            .map(|s| s.rendering.asset_integration_budget_ms)
            .unwrap_or(2);
        self.frontend
            .effective_asset_integration_budget_ms(coupled_default_ms)
    }

    fn asset_particle_integration_budget_ms(&self) -> u32 {
        self.config
            .settings
            .read()
            .map(|s| s.rendering.asset_particle_integration_budget_ms.max(1))
            .unwrap_or(4)
    }

    fn run_asset_integration_pass(
        &mut self,
        queue_access_mode: GpuQueueAccessMode,
    ) -> Option<crate::backend::AssetIntegrationDrainSummary> {
        let budget_ms = self.asset_integration_budget_ms();
        let now = Instant::now();
        let deadline = now + Duration::from_millis(u64::from(budget_ms));
        let particle_deadline = deadline
            + Duration::from_millis(u64::from(self.asset_particle_integration_budget_ms()));
        let pending_asset_work = self.backend.has_pending_asset_work();
        let (shm, ipc) = self.frontend.transport_pair_mut();
        let Some(shm) = shm else {
            if pending_asset_work
                && let Some(occurrence) = ASSET_INTEGRATION_NO_SHM_LOG.should_log(4, 128)
            {
                logger::warn!(
                    "asset integration skipped: shared memory unavailable with pending asset/material work occurrence={occurrence}"
                );
            }
            return None;
        };
        let mut ipc_opt = ipc;
        let summary = self.backend.drain_asset_tasks(
            shm,
            &mut ipc_opt,
            deadline,
            particle_deadline,
            queue_access_mode,
        );
        Some(summary)
    }

    fn record_asset_integration_summary(
        &mut self,
        summary: crate::backend::AssetIntegrationDrainSummary,
        handle_waits: i32,
    ) {
        self.frontend.record_asset_integration_stats(
            crate::frontend::AssetIntegrationPerformanceSample {
                integration_elapsed: summary.elapsed,
                particle_elapsed: summary.particle_elapsed,
                processed_tasks: summary.processed_tasks,
                high_priority_tasks: summary.high_priority_after,
                normal_priority_tasks: summary.normal_priority_after,
                render_tasks: summary.render_after,
                particle_tasks: summary.particle_after,
                handle_waits,
            },
        );
    }

    /// Whether [`Self::run_asset_integration`] already ran this tick.
    #[cfg(test)]
    pub fn did_integrate_assets_this_tick(&self) -> bool {
        self.tick_state.did_integrate_assets_this_tick()
    }
}

fn trace_asset_integration_summary(
    budget_ms: u32,
    summary: crate::backend::AssetIntegrationDrainSummary,
) {
    if summary.total_before() == 0
        && summary.total_after() == 0
        && !summary.budget_exhausted()
        && summary.gpu_ready
    {
        return;
    }
    logger::trace!(
        "asset integration: budget_ms={} gpu_ready={} elapsed_ms={:.3} particle_elapsed_ms={:.3} main {}->{} high {}->{} render {}->{} normal {}->{} particle {}->{} processed={} exhausted_high={} exhausted_render={} exhausted_normal={} exhausted_particle={} peak_queued={}",
        budget_ms,
        summary.gpu_ready,
        summary.elapsed.as_secs_f64() * 1000.0,
        summary.particle_elapsed.as_secs_f64() * 1000.0,
        summary.main_before,
        summary.main_after,
        summary.high_priority_before,
        summary.high_priority_after,
        summary.render_before,
        summary.render_after,
        summary.normal_priority_before,
        summary.normal_priority_after,
        summary.particle_before,
        summary.particle_after,
        summary.processed_tasks,
        summary.high_priority_budget_exhausted,
        summary.render_budget_exhausted,
        summary.normal_priority_budget_exhausted,
        summary.particle_budget_exhausted,
        summary.peak_queued,
    );
    if !summary.gpu_ready
        && summary.total_after() > 0
        && let Some(occurrence) = ASSET_INTEGRATION_GPU_NOT_READY_LOG.should_log(4, 128)
    {
        logger::warn!(
            "asset integration could not drain GPU work: gpu_ready=false queued_after={} main={} high={} render={} normal={} particle={} occurrence={occurrence}",
            summary.total_after(),
            summary.main_after,
            summary.high_priority_after,
            summary.render_after,
            summary.normal_priority_after,
            summary.particle_after,
        );
    }
    if summary.budget_exhausted()
        && summary.total_after() > 0
        && let Some(occurrence) = ASSET_INTEGRATION_BUDGET_LOG.should_log(4, 128)
    {
        logger::debug!(
            "asset integration yielded with backlog: queued_before={} queued_after={} processed={} elapsed_ms={:.3} particle_elapsed_ms={:.3} exhausted_high={} exhausted_render={} exhausted_normal={} exhausted_particle={} occurrence={occurrence}",
            summary.total_before(),
            summary.total_after(),
            summary.processed_tasks,
            summary.elapsed.as_secs_f64() * 1000.0,
            summary.particle_elapsed.as_secs_f64() * 1000.0,
            summary.high_priority_budget_exhausted,
            summary.render_budget_exhausted,
            summary.normal_priority_budget_exhausted,
            summary.particle_budget_exhausted,
        );
    }
}
