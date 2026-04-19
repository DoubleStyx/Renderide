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

fn step_asset_task(
    asset: &mut AssetTransferQueue,
    device: &Arc<wgpu::Device>,
    gpu_limits: &Arc<GpuLimits>,
    queue: &Arc<wgpu::Queue>,
    shm: &mut SharedMemoryAccessor,
    ipc: &mut Option<&mut DualQueueIpc>,
    task: &mut AssetTask,
) -> StepResult {
    let q = queue.as_ref();
    match task {
        AssetTask::Mesh(m) => m.step(asset, device, gpu_limits, q, shm, ipc),
        AssetTask::Texture(t) => t.step(asset, device, q, shm, ipc),
        AssetTask::Texture3d(t) => t.step(asset, device, q, shm, ipc),
        AssetTask::Cubemap(t) => t.step(asset, device, q, shm, ipc),
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
    let Some(device) = asset.gpu_device.clone() else {
        return;
    };
    let Some(gpu_limits) = asset.gpu_limits.clone() else {
        return;
    };
    let Some(queue_arc) = asset.gpu_queue.clone() else {
        return;
    };

    while let Some(mut task) = asset.integrator.high_priority.pop_front() {
        let step_result =
            step_asset_task(asset, &device, &gpu_limits, &queue_arc, shm, ipc, &mut task);
        if step_result == StepResult::Continue {
            asset.integrator.push_front(task, true);
        }
    }

    while Instant::now() < normal_deadline {
        let Some(mut task) = asset.integrator.normal_priority.pop_front() else {
            break;
        };
        let step_result =
            step_asset_task(asset, &device, &gpu_limits, &queue_arc, shm, ipc, &mut task);
        if step_result == StepResult::Continue {
            asset.integrator.push_front(task, false);
        }
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
