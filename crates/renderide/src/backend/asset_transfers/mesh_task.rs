//! Cooperative [`MeshUploadData`] integration: layout validation then GPU upload from shared memory.

use std::sync::Arc;

use crate::assets::mesh::{
    GpuMesh, MeshBufferLayout, MeshGpuUploadContext, compute_and_validate_mesh_layout,
    mesh_upload_input_fingerprint, try_upload_mesh_from_raw,
};
use crate::gpu::{GpuLimits, GpuMappedBufferHealth};
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::shared::{MeshUploadData, MeshUploadResult, RendererCommand};

use super::AssetTransferQueue;
use super::integrator::StepResult;

/// GPU handles needed by a mesh upload task.
pub(super) struct MeshTaskGpu<'a> {
    /// Logical device for mesh resource creation.
    pub(super) device: &'a Arc<wgpu::Device>,
    /// Effective GPU limits used by mesh upload validation.
    pub(super) gpu_limits: &'a Arc<GpuLimits>,
    /// Queue used for buffer writes.
    pub(super) queue: &'a Arc<wgpu::Queue>,
    /// Shared mapped-buffer invalidation generation from the active GPU context.
    pub(super) mapped_buffer_health: &'a Arc<GpuMappedBufferHealth>,
}

/// Completes a host mesh upload that carries no geometry payload.
pub(super) fn complete_empty_mesh_upload(
    queue: &mut AssetTransferQueue,
    data: &MeshUploadData,
    device: Option<&wgpu::Device>,
    ipc: &mut Option<&mut DualQueueIpc>,
) -> MeshUploadResult {
    profiling::scope!("asset::mesh_empty_upload_finalize");
    let asset_id = data.asset_id;
    let resident_replaced = if let Some(device) = device {
        let mesh = GpuMesh::empty(device, data);
        queue.pools.mesh_pool.insert(mesh)
    } else {
        queue.pools.mesh_pool.remove(asset_id)
    };
    let result = MeshUploadResult {
        asset_id,
        instance_changed: !resident_replaced,
    };
    send_mesh_upload_result(ipc, result.clone());
    result
}

/// Completes a host mesh upload with a negative result.
pub(super) fn complete_failed_mesh_upload(
    asset_id: i32,
    reason: &'static str,
    ipc: &mut Option<&mut DualQueueIpc>,
) -> MeshUploadResult {
    logger::warn!("mesh {asset_id}: upload failed: {reason}");
    let result = MeshUploadResult {
        asset_id,
        instance_changed: false,
    };
    send_mesh_upload_result(ipc, result.clone());
    result
}

/// Sends a mesh upload completion result to the host when IPC is available.
pub(super) fn send_mesh_upload_result(
    ipc: &mut Option<&mut DualQueueIpc>,
    result: MeshUploadResult,
) -> bool {
    if let Some(ipc) = ipc.as_mut() {
        let asset_id = result.asset_id;
        let instance_changed = result.instance_changed;
        let ack_queued = ipc.send_background_reliable(RendererCommand::MeshUploadResult(result));
        if !ack_queued {
            logger::warn!(
                "mesh {asset_id}: failed to enqueue reliable MeshUploadResult ack (instance_changed={instance_changed})"
            );
        }
        return ack_queued;
    }
    false
}

/// Stage for a single mesh upload task.
#[derive(Debug)]
enum MeshStage {
    /// Compute and cache [`MeshBufferLayout`] (CPU only).
    PendingLayout,
    /// Host bytes are captured and ready for renderer-thread GPU upload.
    PendingGpuUpload {
        raw: Arc<[u8]>,
        layout: MeshBufferLayout,
        existing: Option<Box<GpuMesh>>,
        mapped_buffer_generation: u64,
    },
}

/// One in-flight mesh upload driven by [`super::integrator::drain_asset_tasks`].
#[derive(Debug)]
pub struct MeshUploadTask {
    data: MeshUploadData,
    stage: MeshStage,
}

impl MeshUploadTask {
    /// Builds a task starting at layout validation.
    pub fn new(data: MeshUploadData) -> Self {
        Self {
            data,
            stage: MeshStage::PendingLayout,
        }
    }

    /// Returns whether this upload came from a high-priority host command.
    #[cfg(test)]
    pub fn high_priority(&self) -> bool {
        self.data.high_priority
    }

    /// Runs at most one stage (layout, then GPU upload).
    pub(super) fn step(
        &mut self,
        queue: &mut AssetTransferQueue,
        gpu: MeshTaskGpu<'_>,
        shm: &mut SharedMemoryAccessor,
        ipc: &mut Option<&mut DualQueueIpc>,
    ) -> StepResult {
        if matches!(self.stage, MeshStage::PendingLayout) {
            return self.start_pending_layout(queue, gpu, shm, ipc);
        }
        if matches!(self.stage, MeshStage::PendingGpuUpload { .. }) {
            return self.run_pending_gpu_upload(queue, gpu, ipc);
        }
        StepResult::Done
    }

    /// Starts layout resolution and shared-memory capture before the GPU upload step.
    fn start_pending_layout(
        &mut self,
        queue: &mut AssetTransferQueue,
        gpu: MeshTaskGpu<'_>,
        shm: &mut SharedMemoryAccessor,
        ipc: &mut Option<&mut DualQueueIpc>,
    ) -> StepResult {
        profiling::scope!("asset::mesh_pending_layout");
        let asset_id = self.data.asset_id;
        if self.data.buffer.length <= 0 {
            complete_empty_mesh_upload(queue, &self.data, Some(gpu.device.as_ref()), ipc);
            return StepResult::Done;
        }
        let Some(layout) = self.resolve_layout(queue) else {
            complete_failed_mesh_upload(asset_id, "invalid layout or buffer descriptor", ipc);
            return StepResult::Done;
        };

        let data = self.data.clone();
        let existing = queue.pools.mesh_pool.get(asset_id).cloned().map(Box::new);
        let raw_len = data.buffer.length.max(0) as usize;
        let raw_arc = Self::copy_mesh_payload(shm, &data, raw_len);
        let Some(raw) = raw_arc else {
            complete_failed_mesh_upload(asset_id, "shared memory payload unavailable", ipc);
            return StepResult::Done;
        };

        self.stage = MeshStage::PendingGpuUpload {
            raw,
            layout,
            existing,
            mapped_buffer_generation: gpu.mapped_buffer_health.generation(),
        };
        StepResult::Continue
    }

    /// Resolves and caches the mesh buffer layout for the upload.
    fn resolve_layout(&self, queue: &mut AssetTransferQueue) -> Option<MeshBufferLayout> {
        profiling::scope!("asset::mesh_layout");
        let asset_id = self.data.asset_id;
        let input_fp = mesh_upload_input_fingerprint(&self.data);
        if let Some(l) = queue
            .pools
            .mesh_pool
            .get_cached_mesh_layout(asset_id, input_fp)
        {
            return Some(l);
        }
        let Some(l) = compute_and_validate_mesh_layout(&self.data) else {
            logger::error!("mesh {asset_id}: invalid mesh layout or buffer descriptor");
            return None;
        };
        queue
            .pools
            .mesh_pool
            .set_cached_mesh_layout(asset_id, input_fp, l);
        Some(l)
    }

    /// Copies the shared-memory mesh payload into an owned slice for background upload.
    fn copy_mesh_payload(
        shm: &mut SharedMemoryAccessor,
        data: &MeshUploadData,
        raw_len: usize,
    ) -> Option<Arc<[u8]>> {
        profiling::scope!("asset::mesh_shared_memory_read");
        let asset_id = data.asset_id;
        shm.with_read_bytes(&data.buffer, |raw| {
            if raw.len() < raw_len {
                logger::error!(
                    "mesh {asset_id}: raw too short (need {}, got {})",
                    raw_len,
                    raw.len()
                );
                return None;
            }
            Some(Arc::from(&raw[..raw_len]))
        })
    }

    /// Uploads captured mesh data on the renderer's cooperative GPU integration path.
    fn run_pending_gpu_upload(
        &mut self,
        queue: &mut AssetTransferQueue,
        gpu: MeshTaskGpu<'_>,
        ipc: &mut Option<&mut DualQueueIpc>,
    ) -> StepResult {
        profiling::scope!("asset::mesh_gpu_upload_step");
        let stage = std::mem::replace(&mut self.stage, MeshStage::PendingLayout);
        let MeshStage::PendingGpuUpload {
            raw,
            layout,
            existing,
            mapped_buffer_generation,
        } = stage
        else {
            return StepResult::Done;
        };
        let asset_id = self.data.asset_id;
        let upload_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let ctx = MeshGpuUploadContext {
                device: gpu.device.as_ref(),
                queue: gpu.queue.as_ref(),
                gpu_limits: gpu.gpu_limits.as_ref(),
                mapped_buffer_health: gpu.mapped_buffer_health.as_ref(),
                mapped_buffer_generation,
            };
            let validation_scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let upload_result = try_upload_mesh_from_raw(
                ctx,
                &raw,
                &self.data,
                existing.map(|mesh| *mesh),
                &layout,
            );
            if let Some(err) = pollster::block_on(validation_scope.pop()) {
                logger::error!("mesh {asset_id}: GPU upload validation failed: {err}");
                return None;
            }
            upload_result
        }));
        let Ok(upload_result) = upload_result else {
            complete_failed_mesh_upload(asset_id, "GPU upload panicked", ipc);
            return StepResult::Done;
        };
        Self::finalize_gpu_upload(asset_id, upload_result, queue, ipc)
    }

    /// Stores a completed GPU upload and sends the host result.
    fn finalize_gpu_upload(
        asset_id: i32,
        upload_result: Option<GpuMesh>,
        queue: &mut AssetTransferQueue,
        ipc: &mut Option<&mut DualQueueIpc>,
    ) -> StepResult {
        let Some(mesh) = upload_result else {
            complete_failed_mesh_upload(asset_id, "GPU upload rejected", ipc);
            return StepResult::Done;
        };
        profiling::scope!("asset::mesh_upload_finalize");
        let existed_before = queue.pools.mesh_pool.insert(mesh);
        send_mesh_upload_result(
            ipc,
            MeshUploadResult {
                asset_id,
                instance_changed: !existed_before,
            },
        );
        StepResult::Done
    }
}

#[cfg(test)]
mod tests {
    use crate::shared::buffer::SharedMemoryBufferDescriptor;

    use super::*;

    fn empty_upload(asset_id: i32) -> MeshUploadData {
        MeshUploadData {
            asset_id,
            buffer: SharedMemoryBufferDescriptor {
                length: 0,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn empty_mesh_without_gpu_completes_host_callback_semantics() {
        let mut queue = AssetTransferQueue::new();
        let mut ipc = None;

        let result = complete_empty_mesh_upload(&mut queue, &empty_upload(42), None, &mut ipc);

        assert_eq!(result.asset_id, 42);
        assert!(result.instance_changed);
        assert!(queue.pools.mesh_pool.is_empty());
        assert_eq!(queue.integrator.total_queued(), 0);
        assert!(queue.pending.pending_mesh_uploads.is_empty());
    }

    #[test]
    fn failed_mesh_upload_completes_host_callback_semantics() {
        let mut ipc = None;

        let result = complete_failed_mesh_upload(43, "test failure", &mut ipc);

        assert_eq!(result.asset_id, 43);
        assert!(!result.instance_changed);
    }

    #[test]
    fn mesh_upload_result_without_ipc_reports_not_queued() {
        let mut ipc = None;

        let queued = send_mesh_upload_result(
            &mut ipc,
            MeshUploadResult {
                asset_id: 44,
                instance_changed: false,
            },
        );

        assert!(!queued);
    }

    #[test]
    fn missing_shared_memory_payload_is_rejected_before_gpu_upload() {
        let prefix = format!("renderide_test_missing_mesh_{}", std::process::id());
        let mut shm = SharedMemoryAccessor::new(prefix);
        let data = MeshUploadData {
            asset_id: 45,
            buffer: SharedMemoryBufferDescriptor {
                buffer_id: 45,
                buffer_capacity: 16,
                length: 16,
                ..Default::default()
            },
            ..Default::default()
        };

        let raw = MeshUploadTask::copy_mesh_payload(&mut shm, &data, 16);

        assert!(raw.is_none());
    }
}
