//! Priority queues and wall-clock-bounded draining for cooperative mesh/texture upload tasks.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::gpu::GpuLimits;
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::profiling::{AssetIntegrationProfileSample, plot_asset_integration};

use super::AssetTransferQueue;
use super::cubemap_task::CubemapUploadTask;
use super::mesh_task::MeshUploadTask;
use super::texture_task::TextureUploadTask;
use super::texture3d_task::Texture3dUploadTask;

/// Maximum combined queued integration tasks (high + normal). Beyond this, new tasks are dropped with a warning.
pub const MAX_ASSET_INTEGRATION_QUEUED: usize = 2048;

/// Minimum extra wall-clock slice granted to high-priority integration before yielding.
const MIN_HIGH_PRIORITY_EMERGENCY_BUDGET: Duration = Duration::from_millis(1);

/// Queue and budget state observed during one cooperative asset-integration drain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AssetIntegrationDrainSummary {
    /// High-priority tasks queued before the drain.
    pub high_priority_before: usize,
    /// Normal-priority tasks queued before the drain.
    pub normal_priority_before: usize,
    /// High-priority tasks queued after the drain.
    pub high_priority_after: usize,
    /// Normal-priority tasks queued after the drain.
    pub normal_priority_after: usize,
    /// Whether the drain had GPU handles needed to execute upload work.
    pub gpu_ready: bool,
    /// Whether high-priority work exceeded the emergency budget.
    pub high_priority_budget_exhausted: bool,
    /// Whether normal-priority work exceeded the frame budget.
    pub normal_priority_budget_exhausted: bool,
    /// Wall-clock time spent in the drain.
    pub elapsed: Duration,
}

impl AssetIntegrationDrainSummary {
    /// Captures queue state before integration starts.
    fn start(asset: &AssetTransferQueue) -> Self {
        Self {
            high_priority_before: asset.integrator.high_priority.len(),
            normal_priority_before: asset.integrator.normal_priority.len(),
            ..Self::default()
        }
    }

    /// Completes the summary from the queue state after integration ends.
    fn finish(
        mut self,
        asset: &AssetTransferQueue,
        gpu_ready: bool,
        high_priority_budget_exhausted: bool,
        normal_priority_budget_exhausted: bool,
        drain_start: Instant,
    ) -> Self {
        self.high_priority_after = asset.integrator.high_priority.len();
        self.normal_priority_after = asset.integrator.normal_priority.len();
        self.gpu_ready = gpu_ready;
        self.high_priority_budget_exhausted = high_priority_budget_exhausted;
        self.normal_priority_budget_exhausted = normal_priority_budget_exhausted;
        self.elapsed = drain_start.elapsed();
        self
    }

    /// Combined queued work before the drain.
    pub fn total_before(self) -> usize {
        self.high_priority_before + self.normal_priority_before
    }

    /// Combined queued work after the drain.
    pub fn total_after(self) -> usize {
        self.high_priority_after + self.normal_priority_after
    }

    /// Whether any budget ceiling was reached while work remained queued.
    pub fn budget_exhausted(self) -> bool {
        self.high_priority_budget_exhausted || self.normal_priority_budget_exhausted
    }
}

/// One cooperative upload (mesh or texture data).
#[derive(Debug)]
pub enum AssetTask {
    /// Host mesh payload integration.
    Mesh(MeshUploadTask),
    /// Host Texture2D mip integration.
    Texture(TextureUploadTask),
    /// Host Texture3D mip integration.
    Texture3d(Texture3dUploadTask),
    /// Host cubemap face/mip integration.
    Cubemap(CubemapUploadTask),
}

/// Whether a task needs another [`AssetTask::step`] call in a later drain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepResult {
    /// More work remains for this logical upload.
    Continue,
    /// Upload finished (success or logged failure; host callbacks sent when applicable).
    Done,
    /// Task is waiting for a background thread to finish; push to the back of the queue.
    YieldBackground,
}

/// Priority-separated cooperative upload queues ([`Renderite.Unity.AssetIntegrator`]-style).
#[derive(Debug, Default)]
pub struct AssetIntegrator {
    /// [`MeshUploadData::high_priority`] / texture data `high_priority` tasks.
    pub high_priority: VecDeque<AssetTask>,
    /// Standard-priority tasks.
    pub normal_priority: VecDeque<AssetTask>,
}

impl AssetIntegrator {
    /// Total queued tasks.
    pub fn total_queued(&self) -> usize {
        self.high_priority.len() + self.normal_priority.len()
    }

    /// Pops the next task, preferring the high-priority queue.
    pub fn pop_next(&mut self) -> Option<AssetTask> {
        self.high_priority
            .pop_front()
            .or_else(|| self.normal_priority.pop_front())
    }

    /// Pushes a task to the front of the appropriate queue (resume after a [`StepResult::Continue`]).
    pub fn push_front(&mut self, task: AssetTask, high_priority: bool) {
        if high_priority {
            self.high_priority.push_front(task);
        } else {
            self.normal_priority.push_front(task);
        }
    }

    /// Enqueues at the back, or returns `false` if the integrator is full.
    pub fn try_enqueue(&mut self, task: AssetTask, high_priority: bool) -> bool {
        if self.total_queued() >= MAX_ASSET_INTEGRATION_QUEUED {
            return false;
        }
        if high_priority {
            self.high_priority.push_back(task);
        } else {
            self.normal_priority.push_back(task);
        }
        true
    }
}

/// Returns a stable tag for [`AssetTask`] variants, used as Tracy zone data.
#[cfg_attr(
    not(feature = "tracy"),
    expect(dead_code, reason = "tag only consumed by Tracy zones")
)]
fn asset_task_kind_tag(task: &AssetTask) -> &'static str {
    match task {
        AssetTask::Mesh(_) => "Mesh",
        AssetTask::Texture(_) => "Texture",
        AssetTask::Texture3d(_) => "Texture3d",
        AssetTask::Cubemap(_) => "Cubemap",
    }
}

/// GPU handles shared across all [`step_asset_task`] invocations in one drain.
struct AssetUploadGpuContext<'a> {
    /// Device for resource creation and format capability queries.
    device: &'a Arc<wgpu::Device>,
    /// GPU adapter limits shared with mesh upload paths.
    gpu_limits: &'a Arc<GpuLimits>,
    /// Queue for [`wgpu::Queue::write_texture`] / [`wgpu::Queue::write_buffer`] uploads.
    queue: &'a Arc<wgpu::Queue>,
    /// Shared GPU queue access gate for [`wgpu::Queue::write_texture`]; see
    /// [`crate::gpu::GpuQueueAccessGate`].
    gpu_queue_access_gate: &'a crate::gpu::GpuQueueAccessGate,
}

fn step_asset_task(
    asset: &mut AssetTransferQueue,
    gpu: &AssetUploadGpuContext<'_>,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
    task: &mut AssetTask,
) -> StepResult {
    profiling::scope!("asset::upload", asset_task_kind_tag(task));
    let device = gpu.device;
    let q = gpu.queue.as_ref();
    let gate = gpu.gpu_queue_access_gate;
    match task {
        AssetTask::Mesh(m) => m.step(asset, device, gpu.gpu_limits, gpu.queue, shm, ipc),
        AssetTask::Texture(t) => t.step(asset, device, q, gate, shm, ipc),
        AssetTask::Texture3d(t) => t.step(asset, device, q, gate, shm, ipc),
        AssetTask::Cubemap(t) => t.step(asset, device, q, gate, shm, ipc),
    }
}

/// Returns the emergency ceiling for high-priority tasks in a bounded drain.
fn high_priority_emergency_deadline(start: Instant, normal_deadline: Instant) -> Instant {
    let normal_budget = match normal_deadline.checked_duration_since(start) {
        Some(duration) => duration,
        None => Duration::ZERO,
    };
    let emergency_budget = normal_budget.max(MIN_HIGH_PRIORITY_EMERGENCY_BUDGET);
    let base_deadline = normal_deadline.max(start);
    match base_deadline.checked_add(emergency_budget) {
        Some(deadline) => deadline,
        None => base_deadline,
    }
}

/// Emits current asset integration queue pressure to the profiler.
fn plot_asset_integrator_backlog(
    asset: &AssetTransferQueue,
    high_priority_budget_exhausted: bool,
    normal_priority_budget_exhausted: bool,
) {
    plot_asset_integration(AssetIntegrationProfileSample {
        high_priority_queued: asset.integrator.high_priority.len(),
        normal_priority_queued: asset.integrator.normal_priority.len(),
        high_priority_budget_exhausted,
        normal_priority_budget_exhausted,
    });
}

/// Drains urgent upload tasks until empty, background-yielded, or the emergency ceiling is hit.
fn drain_high_priority_asset_tasks(
    asset: &mut AssetTransferQueue,
    gpu: &AssetUploadGpuContext<'_>,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
    high_priority_deadline: Instant,
) -> bool {
    profiling::scope!("asset::high_priority_drain");
    let mut yielded = 0;
    loop {
        if Instant::now() >= high_priority_deadline {
            return !asset.integrator.high_priority.is_empty();
        }
        let Some(mut task) = asset.integrator.high_priority.pop_front() else {
            return false;
        };
        let step_result = step_asset_task(asset, gpu, shm, ipc, &mut task);
        match step_result {
            StepResult::Continue => {
                asset.integrator.push_front(task, true);
                yielded = 0;
            }
            StepResult::YieldBackground => {
                asset.integrator.high_priority.push_back(task);
                yielded += 1;
                if yielded >= asset.integrator.high_priority.len() {
                    return false;
                }
            }
            StepResult::Done => {
                yielded = 0;
            }
        }
    }
}

/// Drains normal upload tasks until empty, background-yielded, or the frame budget is hit.
fn drain_normal_priority_asset_tasks(
    asset: &mut AssetTransferQueue,
    gpu: &AssetUploadGpuContext<'_>,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
    normal_deadline: Instant,
) -> bool {
    profiling::scope!("asset::normal_priority_drain");
    let mut yielded = 0;
    loop {
        if Instant::now() >= normal_deadline {
            return !asset.integrator.normal_priority.is_empty();
        }
        let Some(mut task) = asset.integrator.normal_priority.pop_front() else {
            return false;
        };
        let step_result = step_asset_task(asset, gpu, shm, ipc, &mut task);
        match step_result {
            StepResult::Continue => {
                asset.integrator.push_front(task, false);
                yielded = 0;
            }
            StepResult::YieldBackground => {
                asset.integrator.normal_priority.push_back(task);
                yielded += 1;
                if yielded >= asset.integrator.normal_priority.len() {
                    return false;
                }
            }
            StepResult::Done => {
                yielded = 0;
            }
        }
    }
}

/// Polls video texture players after upload integration.
///
/// Also samples each player's clock error against the host's last-applied playback request and
/// appends the result to [`AssetTransferQueue::pending_video_clock_errors`] so the runtime can
/// flush it into the next [`crate::shared::FrameStartData`].
fn poll_video_texture_events(asset: &mut AssetTransferQueue, ipc: &mut Option<&mut DualQueueIpc>) {
    profiling::scope!("asset::video_texture_poll_events");
    let mut video_textures = std::mem::take(&mut asset.video.video_players);
    let mut clock_errors = std::mem::take(&mut asset.video.pending_video_clock_errors);
    {
        profiling::scope!("video::sample_clock_errors");
        for player in video_textures.values_mut() {
            player.process_events(asset, ipc);
            if let Some(state) = player.sample_clock_error() {
                clock_errors.push(state);
            }
        }
    }
    asset.video.video_players = video_textures;
    asset.video.pending_video_clock_errors = clock_errors;
}

/// Runs integration steps: high-priority tasks get an emergency ceiling, then normal-priority tasks
/// run until `normal_deadline`.
pub fn drain_asset_tasks(
    asset: &mut AssetTransferQueue,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
    normal_deadline: Instant,
) -> AssetIntegrationDrainSummary {
    profiling::scope!("asset::drain_tasks");
    let drain_start = Instant::now();
    let summary = AssetIntegrationDrainSummary::start(asset);
    let high_priority_deadline = high_priority_emergency_deadline(drain_start, normal_deadline);
    let Some(device) = asset.gpu.gpu_device.clone() else {
        plot_asset_integrator_backlog(asset, false, false);
        return summary.finish(asset, false, false, false, drain_start);
    };
    let Some(gpu_limits) = asset.gpu.gpu_limits.clone() else {
        plot_asset_integrator_backlog(asset, false, false);
        return summary.finish(asset, false, false, false, drain_start);
    };
    let Some(queue_arc) = asset.gpu.gpu_queue.clone() else {
        plot_asset_integrator_backlog(asset, false, false);
        return summary.finish(asset, false, false, false, drain_start);
    };
    let Some(gate) = asset.gpu.gpu_queue_access_gate.clone() else {
        plot_asset_integrator_backlog(asset, false, false);
        return summary.finish(asset, false, false, false, drain_start);
    };
    let gpu = AssetUploadGpuContext {
        device: &device,
        gpu_limits: &gpu_limits,
        queue: &queue_arc,
        gpu_queue_access_gate: &gate,
    };

    let high_priority_budget_exhausted =
        drain_high_priority_asset_tasks(asset, &gpu, shm, ipc, high_priority_deadline);
    if high_priority_budget_exhausted {
        logger::trace!(
            "asset integrator: high-priority emergency budget exhausted with {} task(s) pending",
            asset.integrator.high_priority.len()
        );
    }

    let normal_priority_budget_exhausted =
        drain_normal_priority_asset_tasks(asset, &gpu, shm, ipc, normal_deadline);
    if normal_priority_budget_exhausted {
        // Tasks pending after wall-clock deadline. Not necessarily a bug -- asset arrival can
        // outpace integration on busy frames -- but persistent backlog growth indicates the
        // budget is too tight or a task is stuck. Per-frame at trace level so it does not
        // spam the default-level log.
        logger::trace!(
            "asset integrator: normal-priority budget exhausted with {} task(s) pending",
            asset.integrator.normal_priority.len()
        );
    }

    plot_asset_integrator_backlog(
        asset,
        high_priority_budget_exhausted,
        normal_priority_budget_exhausted,
    );

    poll_video_texture_events(asset, ipc);
    summary.finish(
        asset,
        true,
        high_priority_budget_exhausted,
        normal_priority_budget_exhausted,
        drain_start,
    )
}

/// Drains all queued tasks without a time limit (used on GPU attach before first frame).
pub fn drain_asset_tasks_unbounded(
    asset: &mut AssetTransferQueue,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
) {
    let far_future = Instant::now() + Duration::from_secs(3600);
    let _ = drain_asset_tasks(asset, shm, ipc, far_future);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::{SetTexture2DData, SetTexture2DFormat};

    fn texture_task(high_priority: bool) -> AssetTask {
        AssetTask::Texture(TextureUploadTask::new(
            SetTexture2DData {
                high_priority,
                ..Default::default()
            },
            SetTexture2DFormat::default(),
            wgpu::TextureFormat::Rgba8Unorm,
        ))
    }

    fn task_is_high_priority(task: &AssetTask) -> bool {
        match task {
            AssetTask::Mesh(task) => task.high_priority(),
            AssetTask::Texture(task) => task.high_priority(),
            AssetTask::Texture3d(task) => task.high_priority(),
            AssetTask::Cubemap(task) => task.high_priority(),
        }
    }

    #[test]
    fn asset_integration_summary_totals_and_budget_flag() {
        let summary = AssetIntegrationDrainSummary {
            high_priority_before: 2,
            normal_priority_before: 3,
            high_priority_after: 1,
            normal_priority_after: 4,
            normal_priority_budget_exhausted: true,
            ..Default::default()
        };

        assert_eq!(summary.total_before(), 5);
        assert_eq!(summary.total_after(), 5);
        assert!(summary.budget_exhausted());
    }

    #[test]
    fn high_priority_emergency_deadline_extends_normal_budget() {
        let start = Instant::now();
        let normal_deadline = start + Duration::from_millis(3);

        let deadline = high_priority_emergency_deadline(start, normal_deadline);

        assert_eq!(
            deadline.checked_duration_since(normal_deadline),
            Some(Duration::from_millis(3))
        );
    }

    #[test]
    fn high_priority_emergency_deadline_has_minimum_budget_when_normal_deadline_elapsed() {
        let start = Instant::now();
        let normal_deadline = match start.checked_sub(Duration::from_millis(5)) {
            Some(deadline) => deadline,
            None => start,
        };

        let deadline = high_priority_emergency_deadline(start, normal_deadline);

        assert_eq!(
            deadline.checked_duration_since(start),
            Some(MIN_HIGH_PRIORITY_EMERGENCY_BUDGET)
        );
    }

    #[test]
    fn pop_next_prefers_high_priority_queue() {
        let mut integrator = AssetIntegrator::default();
        assert!(integrator.try_enqueue(texture_task(false), false));
        assert!(integrator.try_enqueue(texture_task(true), true));

        let first = integrator.pop_next().unwrap();
        let second = integrator.pop_next().unwrap();

        assert!(task_is_high_priority(&first));
        assert!(!task_is_high_priority(&second));
        assert_eq!(integrator.total_queued(), 0);
    }

    #[test]
    fn push_front_preserves_priority_lane() {
        let mut integrator = AssetIntegrator::default();
        integrator.push_front(texture_task(false), false);
        integrator.push_front(texture_task(true), true);

        assert_eq!(integrator.high_priority.len(), 1);
        assert_eq!(integrator.normal_priority.len(), 1);
        assert!(task_is_high_priority(
            integrator.pop_next().as_ref().unwrap()
        ));
    }

    #[test]
    fn try_enqueue_rejects_when_combined_capacity_is_full() {
        let mut integrator = AssetIntegrator::default();
        for _ in 0..MAX_ASSET_INTEGRATION_QUEUED {
            assert!(integrator.try_enqueue(texture_task(false), false));
        }

        assert!(!integrator.try_enqueue(texture_task(true), true));
        assert_eq!(integrator.total_queued(), MAX_ASSET_INTEGRATION_QUEUED);
    }
}
