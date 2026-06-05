//! Priority-separated cooperative upload queues and delayed-removal accounting.

use std::collections::VecDeque;

use super::retired::RetiredAssetResource;
use super::step::AssetTask;
use crate::backend::asset_transfers::limits::MAX_ASSET_INTEGRATION_QUEUE_TASKS;

/// Combined queued integration task count that emits queue-pressure diagnostics.
pub const ASSET_INTEGRATION_QUEUE_WARN_THRESHOLD: usize = 2048;

/// Queue-pressure log stride after [`ASSET_INTEGRATION_QUEUE_WARN_THRESHOLD`] is exceeded.
const ASSET_INTEGRATION_QUEUE_WARN_STRIDE: usize = 1024;

/// Number of integration updates a removed GPU resource is retained before drop.
const DELAYED_REMOVAL_UPDATES: usize = 3;

/// Logical scheduler lane for an [`AssetTask`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssetTaskLane {
    /// Renderer-main-thread tasks drained before other lanes.
    Main,
    /// Urgent upload lane.
    HighPriority,
    /// Standard upload lane.
    NormalPriority,
    /// Wgpu-native render-thread-adjacent work.
    Render,
    /// Dynamic-buffer / particle lane with a separate post-main budget.
    Particle,
}

/// Snapshot of cooperative asset-integration queue depths.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct AssetIntegratorDiagnosticSnapshot {
    /// Renderer-main-thread tasks waiting to run.
    pub(crate) main_queued: usize,
    /// Urgent upload tasks waiting to run.
    pub(crate) high_priority_queued: usize,
    /// Wgpu render-lane tasks waiting to run.
    pub(crate) render_queued: usize,
    /// Standard-priority upload tasks waiting to run.
    pub(crate) normal_priority_queued: usize,
    /// Dynamic-buffer and particle tasks waiting to run.
    pub(crate) particle_queued: usize,
    /// Total queued work across every lane.
    pub(crate) total_queued: usize,
    /// Highest total queued depth observed since startup.
    pub(crate) peak_queued: usize,
}

/// Priority-separated cooperative upload queues.
#[derive(Debug, Default)]
pub struct AssetIntegrator {
    /// Renderer-main-thread tasks run before the priority lanes.
    pub main: VecDeque<AssetTask>,
    /// [`MeshUploadData::high_priority`] / texture data `high_priority` tasks.
    pub high_priority: VecDeque<AssetTask>,
    /// Standard-priority tasks.
    pub normal_priority: VecDeque<AssetTask>,
    /// Wgpu render-lane tasks.
    pub render: VecDeque<AssetTask>,
    /// Dynamic-buffer / particle tasks.
    pub particle: VecDeque<AssetTask>,
    /// Removed resources held alive for the delayed-removal window.
    delayed_removals: VecDeque<RetiredAssetResource>,
    /// Per-bucket delayed-removal counts.
    delayed_removal_counts: [usize; DELAYED_REMOVAL_UPDATES],
    /// Current delayed-removal bucket.
    delayed_removal_bucket_index: usize,
    /// Highest combined queue depth observed since startup.
    max_total_queued: usize,
}

impl AssetIntegrator {
    /// Total queued tasks.
    pub fn total_queued(&self) -> usize {
        self.main.len()
            + self.high_priority.len()
            + self.render.len()
            + self.normal_priority.len()
            + self.particle.len()
    }

    /// Highest combined queued task count observed since startup.
    pub fn peak_queued(&self) -> usize {
        self.max_total_queued
    }

    /// Returns a compact queue-depth snapshot for lifecycle diagnostics.
    pub(crate) fn diagnostic_snapshot(&self) -> AssetIntegratorDiagnosticSnapshot {
        AssetIntegratorDiagnosticSnapshot {
            main_queued: self.main.len(),
            high_priority_queued: self.high_priority.len(),
            render_queued: self.render.len(),
            normal_priority_queued: self.normal_priority.len(),
            particle_queued: self.particle.len(),
            total_queued: self.total_queued(),
            peak_queued: self.peak_queued(),
        }
    }

    /// Pops the next task, preferring the high-priority queue.
    #[cfg(test)]
    pub fn pop_next(&mut self) -> Option<AssetTask> {
        self.main
            .pop_front()
            .or_else(|| self.high_priority.pop_front())
            .or_else(|| self.render.pop_front())
            .or_else(|| self.normal_priority.pop_front())
            .or_else(|| self.particle.pop_front())
    }

    /// Pushes a task to the front of the requested lane.
    pub fn push_front_lane(&mut self, task: AssetTask, lane: AssetTaskLane) {
        self.lane_mut(lane).push_front(task);
    }

    /// Pushes a task to the back of the requested lane.
    pub fn push_back_lane(&mut self, task: AssetTask, lane: AssetTaskLane) {
        self.lane_mut(lane).push_back(task);
    }

    /// Pops a task from the requested lane.
    pub fn pop_front_lane(&mut self, lane: AssetTaskLane) -> Option<AssetTask> {
        self.lane_mut(lane).pop_front()
    }

    /// Returns the queued count for `lane`.
    pub fn lane_len(&self, lane: AssetTaskLane) -> usize {
        match lane {
            AssetTaskLane::Main => self.main.len(),
            AssetTaskLane::HighPriority => self.high_priority.len(),
            AssetTaskLane::NormalPriority => self.normal_priority.len(),
            AssetTaskLane::Render => self.render.len(),
            AssetTaskLane::Particle => self.particle.len(),
        }
    }

    /// Whether `lane` has no queued work.
    pub fn lane_is_empty(&self, lane: AssetTaskLane) -> bool {
        self.lane_len(lane) == 0
    }

    fn lane_mut(&mut self, lane: AssetTaskLane) -> &mut VecDeque<AssetTask> {
        match lane {
            AssetTaskLane::Main => &mut self.main,
            AssetTaskLane::HighPriority => &mut self.high_priority,
            AssetTaskLane::NormalPriority => &mut self.normal_priority,
            AssetTaskLane::Render => &mut self.render,
            AssetTaskLane::Particle => &mut self.particle,
        }
    }

    /// Pushes a task to the front of the appropriate queue (resume after a `StepResult::Continue`).
    #[cfg(test)]
    pub fn push_front(&mut self, task: AssetTask, high_priority: bool) {
        if high_priority {
            self.push_front_lane(task, AssetTaskLane::HighPriority);
        } else {
            self.push_front_lane(task, AssetTaskLane::NormalPriority);
        }
    }

    /// Enqueues an upload task at the back of its priority lane.
    pub fn enqueue(&mut self, task: AssetTask, high_priority: bool) -> bool {
        if !self.admit_task() {
            return false;
        }
        if high_priority {
            self.push_back_lane(task, AssetTaskLane::HighPriority);
        } else {
            self.push_back_lane(task, AssetTaskLane::NormalPriority);
        }
        self.record_queue_depth();
        true
    }

    /// Enqueues a task in a specific scheduler lane.
    pub fn enqueue_lane(&mut self, task: AssetTask, lane: AssetTaskLane) -> bool {
        if !self.admit_task() {
            return false;
        }
        self.push_back_lane(task, lane);
        self.record_queue_depth();
        true
    }

    /// Enqueues a removed GPU resource for delayed drop.
    pub fn enqueue_delayed_removal(&mut self, resource: RetiredAssetResource) {
        self.delayed_removals.push_back(resource);
        self.delayed_removal_counts[self.delayed_removal_bucket_index] += 1;
    }

    /// Drops the delayed-removal bucket that has aged through the configured update window.
    pub fn process_delayed_removals(&mut self) -> usize {
        let index = (self.delayed_removal_bucket_index + (DELAYED_REMOVAL_UPDATES - 1))
            % DELAYED_REMOVAL_UPDATES;
        let count = self.delayed_removal_counts[index];
        let mut released_bytes = 0;
        for _ in 0..count {
            if let Some(resource) = self.delayed_removals.pop_front() {
                released_bytes += resource.resident_bytes();
            }
        }
        if count > 0 {
            logger::trace!(
                "asset integrator delayed removals released: count={count} bytes={released_bytes}"
            );
        }
        self.delayed_removal_counts[index] = 0;
        self.delayed_removal_bucket_index =
            (self.delayed_removal_bucket_index + 1) % DELAYED_REMOVAL_UPDATES;
        count
    }

    fn record_queue_depth(&mut self) {
        let queued = self.total_queued();
        self.max_total_queued = self.max_total_queued.max(queued);
        if should_log_asset_integration_queue_pressure(queued) {
            logger::warn!(
                "asset integrator backlog high: queued={} main={} high_priority={} render={} normal_priority={} particle={} threshold={}",
                queued,
                self.main.len(),
                self.high_priority.len(),
                self.render.len(),
                self.normal_priority.len(),
                self.particle.len(),
                ASSET_INTEGRATION_QUEUE_WARN_THRESHOLD
            );
        }
    }

    fn admit_task(&self) -> bool {
        let queued = self.total_queued();
        if queued >= MAX_ASSET_INTEGRATION_QUEUE_TASKS {
            logger::warn!(
                "asset integrator backlog full: queued={} cap={}",
                queued,
                MAX_ASSET_INTEGRATION_QUEUE_TASKS
            );
            return false;
        }
        true
    }
}

fn should_log_asset_integration_queue_pressure(queued: usize) -> bool {
    queued == ASSET_INTEGRATION_QUEUE_WARN_THRESHOLD
        || (queued > ASSET_INTEGRATION_QUEUE_WARN_THRESHOLD
            && queued.is_multiple_of(ASSET_INTEGRATION_QUEUE_WARN_STRIDE))
}
