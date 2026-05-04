//! Mesh upload IPC: enqueue cooperative [`super::super::mesh_task::MeshUploadTask`] integration.

use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::shared::{MeshUnload, MeshUploadData};

use super::super::AssetTransferQueue;
use super::super::integrator::AssetTask;
use super::super::mesh_task::MeshUploadTask;
use super::MAX_PENDING_MESH_UPLOADS;

/// Remove a mesh from the pool.
pub fn on_mesh_unload(queue: &mut AssetTransferQueue, u: MeshUnload) {
    let pending_before = queue.pending.pending_mesh_uploads.len();
    queue
        .pending
        .pending_mesh_uploads
        .retain(|upload| upload.asset_id != u.asset_id);
    let pending_removed = pending_before.saturating_sub(queue.pending.pending_mesh_uploads.len());
    if pending_removed > 0 {
        logger::debug!(
            "mesh {} unload removed {} deferred upload(s)",
            u.asset_id,
            pending_removed
        );
    }
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
        let asset_id = data.asset_id;
        queue.pending.pending_mesh_uploads.push_back(data);
        log_pending_mesh_upload_pressure(queue, asset_id);
        return;
    }

    let high = data.high_priority;
    let task = AssetTask::Mesh(MeshUploadTask::new(data));
    queue.integrator_mut().enqueue(task, high);
}

fn log_pending_mesh_upload_pressure(queue: &AssetTransferQueue, asset_id: i32) {
    let pending = queue.pending.pending_mesh_uploads.len();
    if pending == MAX_PENDING_MESH_UPLOADS
        || (pending > MAX_PENDING_MESH_UPLOADS && pending.is_multiple_of(MAX_PENDING_MESH_UPLOADS))
    {
        logger::warn!(
            "mesh {asset_id}: deferred upload backlog high: pending={} threshold={} reason=gpu not attached",
            pending,
            MAX_PENDING_MESH_UPLOADS
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::shared::buffer::SharedMemoryBufferDescriptor;

    use super::*;

    fn upload(asset_id: i32) -> MeshUploadData {
        MeshUploadData {
            asset_id,
            buffer: SharedMemoryBufferDescriptor {
                length: 16,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn mesh_without_gpu_is_deferred_beyond_warning_threshold() {
        let mut queue = AssetTransferQueue::new();
        let mut shm = SharedMemoryAccessor::new(String::new());

        for i in 0..=MAX_PENDING_MESH_UPLOADS {
            try_process_mesh_upload(&mut queue, upload(i as i32), &mut shm, None);
        }

        assert_eq!(
            queue.pending.pending_mesh_uploads.len(),
            MAX_PENDING_MESH_UPLOADS + 1
        );
    }

    #[test]
    fn mesh_unload_removes_deferred_uploads_for_asset() {
        let mut queue = AssetTransferQueue::new();
        let mut shm = SharedMemoryAccessor::new(String::new());

        try_process_mesh_upload(&mut queue, upload(7), &mut shm, None);
        try_process_mesh_upload(&mut queue, upload(8), &mut shm, None);
        on_mesh_unload(&mut queue, MeshUnload { asset_id: 7 });

        assert_eq!(queue.pending.pending_mesh_uploads.len(), 1);
        assert_eq!(queue.pending.pending_mesh_uploads[0].asset_id, 8);
    }
}
