//! Cooperative [`MeshUploadData`] integration: layout validation then GPU upload from shared memory.

use std::sync::Arc;

use crate::assets::mesh::{
    GpuMesh, compute_and_validate_mesh_layout, mesh_upload_input_fingerprint,
    try_upload_mesh_from_raw,
};
use crate::gpu::GpuLimits;
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::shared::{MeshUploadData, MeshUploadResult, RendererCommand};

use super::AssetTransferQueue;
use super::integrator::StepResult;

/// Completes a host mesh upload that carries no geometry payload.
pub(super) fn complete_empty_mesh_upload(
    queue: &mut AssetTransferQueue,
    data: &MeshUploadData,
    device: Option<&wgpu::Device>,
    ipc: &mut Option<&mut DualQueueIpc>,
) -> MeshUploadResult {
    profiling::scope!("asset::mesh_empty_upload_finalize");
    let asset_id = data.asset_id;
    let (resident_replaced, resident_stored) = if let Some(device) = device {
        let mesh = GpuMesh::empty(device, data);
        (queue.pools.mesh_pool.insert(mesh), true)
    } else {
        (queue.pools.mesh_pool.remove(asset_id), false)
    };
    let result = MeshUploadResult {
        asset_id,
        instance_changed: !resident_replaced,
    };
    send_mesh_upload_result(ipc, result.clone());
    logger::trace!(
        "mesh {} completed empty upload (replaced={} resident_stored={} resident_bytes~={})",
        asset_id,
        resident_replaced,
        resident_stored,
        queue.pools.mesh_pool.accounting().total_resident_bytes()
    );
    result
}

fn send_mesh_upload_result(ipc: &mut Option<&mut DualQueueIpc>, result: MeshUploadResult) {
    if let Some(ipc) = ipc.as_mut() {
        let _ = ipc.send_background(RendererCommand::MeshUploadResult(result));
    }
}

/// Stage for a single mesh upload task ([`Renderite.Unity.MeshAsset.Upload`]-style splitting).
#[derive(Debug)]
enum MeshStage {
    /// Compute and cache [`MeshBufferLayout`] (CPU only).
    PendingLayout,
    /// Background thread extraction and GPU upload.
    Decoding {
        rx: crossbeam_channel::Receiver<Option<GpuMesh>>,
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

    /// [`MeshUploadData::high_priority`].
    pub fn high_priority(&self) -> bool {
        self.data.high_priority
    }

    /// Runs at most one stage (layout, then GPU upload).
    pub fn step(
        &mut self,
        queue: &mut AssetTransferQueue,
        device: &Arc<wgpu::Device>,
        gpu_limits: &Arc<GpuLimits>,
        gpu_queue: &Arc<wgpu::Queue>,
        shm: &mut SharedMemoryAccessor,
        ipc: &mut Option<&mut DualQueueIpc>,
    ) -> StepResult {
        let asset_id = self.data.asset_id;
        if matches!(self.stage, MeshStage::PendingLayout) {
            return self.start_pending_layout(queue, device, gpu_limits, gpu_queue, shm, ipc);
        }
        if let MeshStage::Decoding { rx } = &mut self.stage {
            return Self::poll_background_upload(asset_id, rx, queue, ipc);
        }
        StepResult::Done
    }

    /// Starts layout resolution, shared-memory capture, and background GPU upload.
    fn start_pending_layout(
        &mut self,
        queue: &mut AssetTransferQueue,
        device: &Arc<wgpu::Device>,
        gpu_limits: &Arc<GpuLimits>,
        gpu_queue: &Arc<wgpu::Queue>,
        shm: &mut SharedMemoryAccessor,
        ipc: &mut Option<&mut DualQueueIpc>,
    ) -> StepResult {
        profiling::scope!("asset::mesh_pending_layout");
        let asset_id = self.data.asset_id;
        if self.data.buffer.length <= 0 {
            complete_empty_mesh_upload(queue, &self.data, Some(device.as_ref()), ipc);
            return StepResult::Done;
        }
        let Some(layout) = self.resolve_layout(queue) else {
            return StepResult::Done;
        };

        let data = self.data.clone();
        let existing = queue.pools.mesh_pool.get(asset_id).cloned();
        let raw_len = data.buffer.length.max(0) as usize;
        let raw_arc = Self::copy_mesh_payload(shm, &data, raw_len);
        let Some(raw) = raw_arc else {
            return StepResult::Done;
        };

        let (tx, rx) = crossbeam_channel::bounded(1);
        let device_clone = Arc::clone(device);
        let gpu_limits_clone = Arc::clone(gpu_limits);
        let gpu_queue_clone = Arc::clone(gpu_queue);
        rayon::spawn(move || {
            profiling::scope!("asset::mesh_upload_background");
            let mesh = try_upload_mesh_from_raw(
                device_clone.as_ref(),
                gpu_limits_clone.as_ref(),
                Some(gpu_queue_clone.as_ref()),
                &raw,
                &data,
                existing,
                &layout,
            );
            let _ = tx.send(mesh);
        });

        self.stage = MeshStage::Decoding { rx };
        StepResult::YieldBackground
    }

    /// Resolves and caches the mesh buffer layout for the upload.
    fn resolve_layout(
        &self,
        queue: &mut AssetTransferQueue,
    ) -> Option<crate::assets::mesh::MeshBufferLayout> {
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

    /// Polls the background upload and integrates the resident mesh plus host callback.
    fn poll_background_upload(
        asset_id: i32,
        rx: &crossbeam_channel::Receiver<Option<GpuMesh>>,
        queue: &mut AssetTransferQueue,
        ipc: &mut Option<&mut DualQueueIpc>,
    ) -> StepResult {
        profiling::scope!("asset::mesh_upload_poll");
        match rx.try_recv() {
            Ok(upload_result) => {
                Self::finalize_background_upload(asset_id, upload_result, queue, ipc)
            }
            Err(crossbeam_channel::TryRecvError::Empty) => StepResult::YieldBackground,
            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                logger::error!("mesh {asset_id}: background decode thread panicked");
                StepResult::Done
            }
        }
    }

    /// Stores a completed background upload and sends the host result.
    fn finalize_background_upload(
        asset_id: i32,
        upload_result: Option<GpuMesh>,
        queue: &mut AssetTransferQueue,
        ipc: &mut Option<&mut DualQueueIpc>,
    ) -> StepResult {
        let Some(mesh) = upload_result else {
            logger::error!(
                "mesh {asset_id}: upload failed or rejected -- host callback not completed (no MeshUploadResult sent)"
            );
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
        logger::trace!(
            "mesh {} uploaded via integrator (replaced={} resident_bytes~={})",
            asset_id,
            existed_before,
            queue.pools.mesh_pool.accounting().total_resident_bytes()
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
}
