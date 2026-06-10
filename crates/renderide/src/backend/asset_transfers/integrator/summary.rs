//! Drain-summary types reported by [`super::drain::drain_asset_tasks`].

use std::time::Duration;

use super::super::AssetTransferQueue;

/// Queue and budget state observed during one cooperative asset-integration drain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AssetIntegrationDrainSummary {
    /// Main-thread tasks queued before the drain.
    pub main_before: usize,
    /// High-priority tasks queued before the drain.
    pub high_priority_before: usize,
    /// Normal-priority tasks queued before the drain.
    pub normal_priority_before: usize,
    /// Render-lane tasks queued before the drain.
    pub render_before: usize,
    /// Particle/dynamic-buffer tasks queued before the drain.
    pub particle_before: usize,
    /// Main-thread tasks queued after the drain.
    pub main_after: usize,
    /// High-priority tasks queued after the drain.
    pub high_priority_after: usize,
    /// Normal-priority tasks queued after the drain.
    pub normal_priority_after: usize,
    /// Render-lane tasks queued after the drain.
    pub render_after: usize,
    /// Particle/dynamic-buffer tasks queued after the drain.
    pub particle_after: usize,
    /// Whether the drain had GPU handles needed to execute upload work.
    pub gpu_ready: bool,
    /// Number of queue steps processed during the drain.
    pub processed_tasks: u32,
    /// Number of main-lane queue steps processed during the drain.
    pub processed_main_tasks: u32,
    /// Number of high-priority queue steps processed during the drain.
    pub processed_high_priority_tasks: u32,
    /// Number of normal-priority queue steps processed during the drain.
    pub processed_normal_priority_tasks: u32,
    /// Number of render-lane queue steps processed during the drain.
    pub processed_render_tasks: u32,
    /// Number of particle-lane queue steps processed during the drain.
    pub processed_particle_tasks: u32,
    /// Whether high-priority work exceeded the emergency budget.
    pub high_priority_budget_exhausted: bool,
    /// Whether normal-priority work exceeded the frame budget.
    pub normal_priority_budget_exhausted: bool,
    /// Whether render-lane work exceeded the frame budget.
    pub render_budget_exhausted: bool,
    /// Whether particle-lane work exceeded its separate post-main budget.
    pub particle_budget_exhausted: bool,
    /// Whether at least one task made useful forward progress during the drain.
    pub made_progress: bool,
    /// Whether queued work remains but every runnable task is waiting on background/GPU state.
    pub blocked_on_background: bool,
    /// Wall-clock time spent in non-particle integration lanes.
    pub elapsed: Duration,
    /// Wall-clock time spent in the particle lane.
    pub particle_elapsed: Duration,
    /// Highest combined queued task count observed since startup.
    pub peak_queued: usize,
}

impl AssetIntegrationDrainSummary {
    /// Captures queue state before integration starts.
    pub(super) fn start(asset: &AssetTransferQueue) -> Self {
        Self {
            main_before: asset.integrator.main.len(),
            high_priority_before: asset.integrator.high_priority.len(),
            normal_priority_before: asset.integrator.normal_priority.len(),
            render_before: asset.integrator.render.len(),
            particle_before: asset.integrator.particle.len(),
            ..Self::default()
        }
    }

    /// Completes the summary from the queue state after integration ends.
    pub(super) fn finish(mut self, asset: &AssetTransferQueue, finish: DrainFinishState) -> Self {
        self.main_after = asset.integrator.main.len();
        self.high_priority_after = asset.integrator.high_priority.len();
        self.normal_priority_after = asset.integrator.normal_priority.len();
        self.render_after = asset.integrator.render.len();
        self.particle_after = asset.integrator.particle.len();
        self.gpu_ready = finish.gpu_ready;
        self.high_priority_budget_exhausted = finish.budgets.high_priority;
        self.normal_priority_budget_exhausted = finish.budgets.normal_priority;
        self.render_budget_exhausted = finish.budgets.render;
        self.particle_budget_exhausted = finish.budgets.particle;
        self.processed_tasks = finish.processed.total();
        self.processed_main_tasks = finish.processed.main;
        self.processed_high_priority_tasks = finish.processed.high_priority;
        self.processed_normal_priority_tasks = finish.processed.normal_priority;
        self.processed_render_tasks = finish.processed.render;
        self.processed_particle_tasks = finish.processed.particle;
        self.made_progress = finish.made_progress;
        self.blocked_on_background = finish.blocked_on_background;
        self.elapsed = finish.elapsed;
        self.particle_elapsed = finish.particle_elapsed;
        self.peak_queued = asset.integrator.peak_queued();
        self
    }

    /// Combined queued work before the drain.
    pub fn total_before(self) -> usize {
        self.main_before
            + self.high_priority_before
            + self.render_before
            + self.normal_priority_before
            + self.particle_before
    }

    /// Combined queued work after the drain.
    pub fn total_after(self) -> usize {
        self.main_after
            + self.high_priority_after
            + self.render_after
            + self.normal_priority_after
            + self.particle_after
    }

    /// Whether any budget ceiling was reached while work remained queued.
    pub fn budget_exhausted(self) -> bool {
        self.high_priority_budget_exhausted
            || self.normal_priority_budget_exhausted
            || self.render_budget_exhausted
            || self.particle_budget_exhausted
    }
}

/// Per-lane budget exhaustion flags collected during a drain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct BudgetExhaustion {
    pub(super) high_priority: bool,
    pub(super) normal_priority: bool,
    pub(super) render: bool,
    pub(super) particle: bool,
}

/// Combined drain end-state fed into [`AssetIntegrationDrainSummary::finish`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct DrainFinishState {
    pub(super) gpu_ready: bool,
    pub(super) budgets: BudgetExhaustion,
    pub(super) processed: ProcessedLaneCounts,
    pub(super) made_progress: bool,
    pub(super) blocked_on_background: bool,
    pub(super) particle_elapsed: Duration,
    pub(super) elapsed: Duration,
}

/// Per-lane processed-step counters for one full drain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct ProcessedLaneCounts {
    pub(super) main: u32,
    pub(super) high_priority: u32,
    pub(super) normal_priority: u32,
    pub(super) render: u32,
    pub(super) particle: u32,
}

impl ProcessedLaneCounts {
    pub(super) fn total(self) -> u32 {
        self.main
            .saturating_add(self.high_priority)
            .saturating_add(self.normal_priority)
            .saturating_add(self.render)
            .saturating_add(self.particle)
    }
}
