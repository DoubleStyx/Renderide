//! Mesh upload IPC: enqueue cooperative [`super::super::mesh_task::MeshUploadTask`] integration.

use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::shared::{MeshUnload, MeshUploadData};

use super::super::AssetTransferQueue;
use super::super::integrator::AssetTask;
use super::super::mesh_task::MeshUploadTask;
use super::MAX_PENDING_MESH_UPLOADS;

/// Remove a mesh from the pool.
pub fn on_mesh_unload(queue: &mut AssetTransferQueue, u: MeshUnload) {
    if queue.pools.mesh_pool.remove(u.asset_id) {
        logger::info!(
            "mesh {} unloaded (resident_bytes~={})",
            u.asset_id,
            queue.pools.mesh_pool.accounting().total_resident_bytes()
        );
    }
}

/// Enqueue mesh bytes from shared memory for time-sliced GPU integration ([`super::super::integrator::drain_asset_tasks`]).
pub fn try_process_mesh_upload(
    queue: &mut AssetTransferQueue,
    data: MeshUploadData,
    _shm: &mut SharedMemoryAccessor,
    _ipc: Option<&mut DualQueueIpc>,
) {
    if data.buffer.length <= 0 {
        return;
    }
    if queue.gpu.gpu_device.is_none() {
        if queue.pending.pending_mesh_uploads.len() >= MAX_PENDING_MESH_UPLOADS {
            logger::warn!(
                "mesh upload pending queue full; dropping asset {}",
                data.asset_id
            );
            return;
        }
        queue.pending.pending_mesh_uploads.push_back(data);
        return;
    }

    let high = data.high_priority;
    let asset_id = data.asset_id;
    let task = AssetTask::Mesh(MeshUploadTask::new(data));
    if !queue.integrator_mut().try_enqueue(task, high) {
        logger::warn!(
            "mesh {}: asset integration queue full; dropping upload",
            asset_id
        );
    }
}
