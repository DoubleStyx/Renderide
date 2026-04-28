//! Priority queues and wall-clock–bounded draining for cooperative mesh/texture upload tasks.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use crate::gpu::GpuLimits;
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};

use super::cubemap_task::CubemapUploadTask;
use super::mesh_task::MeshUploadTask;
use super::texture3d_task::Texture3dUploadTask;
use super::texture_task::TextureUploadTask;
use super::AssetTransferQueue;

/// Maximum combined queued integration tasks (high + normal). Beyond this, new tasks are dropped with a warning.
pub const MAX_ASSET_INTEGRATION_QUEUED: usize = 2048;

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

/// Priority-separated cooperative upload queues ([`Renderite.Unity.AssetIntegrator`]–style).
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

/// Runs integration steps: **all** high-priority tasks complete with **no** wall-clock limit, then
/// normal-priority tasks until `normal_deadline` (Renderite-style: urgent work is not time-sliced).
pub fn drain_asset_tasks(
    asset: &mut AssetTransferQueue,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
    normal_deadline: Instant,
) {
    profiling::scope!("asset::drain_tasks");
    let Some(device) = asset.gpu_device.clone() else {
        return;
    };
    let Some(gpu_limits) = asset.gpu_limits.clone() else {
        return;
    };
    let Some(queue_arc) = asset.gpu_queue.clone() else {
        return;
    };
    let Some(gate) = asset.gpu_queue_access_gate.clone() else {
        return;
    };
    let gpu = AssetUploadGpuContext {
        device: &device,
        gpu_limits: &gpu_limits,
        queue: &queue_arc,
        gpu_queue_access_gate: &gate,
    };

    {
        profiling::scope!("asset::high_priority_drain");
        let mut yielded = 0;
        while let Some(mut task) = asset.integrator.high_priority.pop_front() {
            let step_result = step_asset_task(asset, &gpu, shm, ipc, &mut task);
            match step_result {
                StepResult::Continue => {
                    asset.integrator.push_front(task, true);
                    yielded = 0;
                }
                StepResult::YieldBackground => {
                    asset.integrator.high_priority.push_back(task);
                    yielded += 1;
                    if yielded >= asset.integrator.high_priority.len() {
                        break;
                    }
                }
                StepResult::Done => {
                    yielded = 0;
                }
            }
        }
    }

    {
        profiling::scope!("asset::normal_priority_drain");
        let mut yielded = 0;
        let mut budget_exhausted = false;
        loop {
            if Instant::now() >= normal_deadline {
                budget_exhausted = !asset.integrator.normal_priority.is_empty();
                break;
            }
            let Some(mut task) = asset.integrator.normal_priority.pop_front() else {
                break;
            };
            let step_result = step_asset_task(asset, &gpu, shm, ipc, &mut task);
            match step_result {
                StepResult::Continue => {
                    asset.integrator.push_front(task, false);
                    yielded = 0;
                }
                StepResult::YieldBackground => {
                    asset.integrator.normal_priority.push_back(task);
                    yielded += 1;
                    if yielded >= asset.integrator.normal_priority.len() {
                        break;
                    }
                }
                StepResult::Done => {
                    yielded = 0;
                }
            }
        }
        if budget_exhausted {
            // Tasks pending after wall-clock deadline. Not necessarily a bug — asset arrival can
            // outpace integration on busy frames — but persistent backlog growth indicates the
            // budget is too tight or a task is stuck. Per-frame at trace level so it does not
            // spam the default-level log.
            logger::trace!(
                "asset integrator: normal-priority budget exhausted with {} task(s) pending",
                asset.integrator.normal_priority.len()
            );
        }
    }

    {
        profiling::scope!("asset::video_texture_poll_events");
        let mut video_textures = std::mem::take(&mut asset.video_players);
        for player in video_textures.values_mut() {
            player.process_events(asset, ipc);
        }
        asset.video_players = video_textures;
    }
}

/// Drains all queued tasks without a time limit (used on GPU attach before first frame).
pub fn drain_asset_tasks_unbounded(
    asset: &mut AssetTransferQueue,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
) {
    let far_future = Instant::now() + std::time::Duration::from_secs(3600);
    drain_asset_tasks(asset, shm, ipc, far_future);
}
