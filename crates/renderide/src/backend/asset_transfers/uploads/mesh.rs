//! Mesh upload IPC: enqueue cooperative [`super::super::mesh_task::MeshUploadTask`] integration.

use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::shared::{MeshUnload, MeshUploadData, MeshUploadResult};

use super::super::AssetTransferQueue;
use super::super::integrator::AssetTask;
use super::super::limits::admit_descriptor_payload_len;
use super::super::mesh_task::{
    MeshUploadTask, complete_empty_mesh_upload, complete_failed_mesh_upload,
};
use super::PENDING_MESH_UPLOAD_WARN_THRESHOLD;

/// Remove a mesh from the pool.
pub fn on_mesh_unload(queue: &mut AssetTransferQueue, u: MeshUnload) {
    queue.invalidate_mesh_upload_generation(u.asset_id);
    let pending_removed = remove_deferred_mesh_uploads_for_asset(queue, u.asset_id);
    if pending_removed > 0 {
        logger::debug!(
            "mesh {} unload removed {} deferred upload(s)",
            u.asset_id,
            pending_removed
        );
    }
    queue.retire_mesh_asset(u.asset_id);
}

/// Enqueue mesh bytes from shared memory for time-sliced GPU integration ([`super::super::integrator::drain_asset_tasks`]).
pub fn try_process_mesh_upload(
    queue: &mut AssetTransferQueue,
    data: MeshUploadData,
    shm: Option<&mut SharedMemoryAccessor>,
    ipc: Option<&mut DualQueueIpc>,
) -> Option<MeshUploadResult> {
    log_mesh_upload_received(&data);
    let mut ipc = ipc;
    if shm.is_none() {
        return Some(complete_failed_mesh_upload(
            data.asset_id,
            "shared memory unavailable",
            &mut ipc,
        ));
    }
    let asset_id = data.asset_id;
    let generation = queue.begin_mesh_upload_generation(asset_id);
    let pending_removed = remove_deferred_mesh_uploads_for_asset(queue, asset_id);
    if pending_removed > 0 {
        logger::trace!(
            "mesh {asset_id}: superseded {} deferred upload(s) with generation {}",
            pending_removed,
            generation
        );
    }
    if data.buffer.length <= 0 {
        let device = queue.gpu.gpu_device.clone();
        return Some(complete_empty_mesh_upload(
            queue,
            &data,
            device.as_deref(),
            &mut ipc,
        ));
    }
    if !admit_descriptor_payload_len("mesh", asset_id, data.buffer.length) {
        return Some(complete_failed_mesh_upload(
            data.asset_id,
            "mesh upload payload exceeds renderer admission cap",
            &mut ipc,
        ));
    }
    if queue.gpu.gpu_device.is_none() {
        if queue.pending.pending_mesh_uploads.len() >= PENDING_MESH_UPLOAD_WARN_THRESHOLD {
            return Some(complete_failed_mesh_upload(
                data.asset_id,
                "too many deferred mesh uploads",
                &mut ipc,
            ));
        }
        queue.pending.pending_mesh_uploads.push_back(data);
        logger::debug!(
            "mesh {asset_id}: deferred upload until GPU attach (pending={})",
            queue.pending.pending_mesh_uploads.len()
        );
        log_pending_mesh_upload_pressure(queue, asset_id);
        return None;
    }

    let high = data.high_priority;
    let asset_id = data.asset_id;
    let task = AssetTask::Mesh(MeshUploadTask::new(data, generation));
    if !queue.integrator_mut().enqueue(task, high) {
        return Some(complete_failed_mesh_upload(
            asset_id,
            "asset integration backlog full",
            &mut ipc,
        ));
    }
    None
}

fn remove_deferred_mesh_uploads_for_asset(queue: &mut AssetTransferQueue, asset_id: i32) -> usize {
    let pending_before = queue.pending.pending_mesh_uploads.len();
    queue
        .pending
        .pending_mesh_uploads
        .retain(|upload| upload.asset_id != asset_id);
    pending_before.saturating_sub(queue.pending.pending_mesh_uploads.len())
}

fn log_mesh_upload_received(data: &MeshUploadData) {
    logger::trace!(
        "mesh {} upload received: bytes={} high_priority={}",
        data.asset_id,
        data.buffer.length,
        data.high_priority
    );
}

fn log_pending_mesh_upload_pressure(queue: &AssetTransferQueue, asset_id: i32) {
    let pending = queue.pending.pending_mesh_uploads.len();
    if pending == PENDING_MESH_UPLOAD_WARN_THRESHOLD
        || (pending > PENDING_MESH_UPLOAD_WARN_THRESHOLD
            && pending.is_multiple_of(PENDING_MESH_UPLOAD_WARN_THRESHOLD))
    {
        logger::warn!(
            "mesh {asset_id}: deferred upload backlog high: pending={} threshold={} reason=gpu not attached",
            pending,
            PENDING_MESH_UPLOAD_WARN_THRESHOLD
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
    fn mesh_without_gpu_is_capped_at_pending_threshold() {
        let mut queue = AssetTransferQueue::new();
        let mut shm = SharedMemoryAccessor::new(String::new());

        for i in 0..=PENDING_MESH_UPLOAD_WARN_THRESHOLD {
            try_process_mesh_upload(&mut queue, upload(i as i32), Some(&mut shm), None);
        }

        assert_eq!(
            queue.pending.pending_mesh_uploads.len(),
            PENDING_MESH_UPLOAD_WARN_THRESHOLD
        );
    }

    #[test]
    fn mesh_rejects_descriptor_payload_above_cap() {
        let mut queue = AssetTransferQueue::new();
        let mut shm = SharedMemoryAccessor::new(String::new());
        let mut upload = upload(1);
        upload.buffer.length = super::super::super::limits::MAX_UPLOAD_PAYLOAD_BYTES + 1;

        let result = try_process_mesh_upload(&mut queue, upload, Some(&mut shm), None)
            .expect("oversized upload should complete as failed");

        assert_eq!(result.asset_id, 1);
        assert!(queue.pending.pending_mesh_uploads.is_empty());
    }

    #[test]
    fn mesh_unload_removes_deferred_uploads_for_asset() {
        let mut queue = AssetTransferQueue::new();
        let mut shm = SharedMemoryAccessor::new(String::new());

        try_process_mesh_upload(&mut queue, upload(7), Some(&mut shm), None);
        let generation = queue
            .current_mesh_upload_generation(7)
            .expect("mesh generation");
        try_process_mesh_upload(&mut queue, upload(8), Some(&mut shm), None);
        on_mesh_unload(&mut queue, MeshUnload { asset_id: 7 });

        assert_eq!(queue.pending.pending_mesh_uploads.len(), 1);
        assert_eq!(queue.pending.pending_mesh_uploads[0].asset_id, 8);
        assert!(!queue.mesh_upload_generation_is_current(7, generation));
    }

    #[test]
    fn mesh_without_gpu_keeps_only_latest_deferred_upload_per_asset() {
        let mut queue = AssetTransferQueue::new();
        let mut shm = SharedMemoryAccessor::new(String::new());
        let first = upload(7);
        let mut second = upload(7);
        second.high_priority = true;

        try_process_mesh_upload(&mut queue, first, Some(&mut shm), None);
        let first_generation = queue
            .current_mesh_upload_generation(7)
            .expect("first mesh generation");
        try_process_mesh_upload(&mut queue, second, Some(&mut shm), None);

        assert_eq!(queue.pending.pending_mesh_uploads.len(), 1);
        assert_eq!(queue.pending.pending_mesh_uploads[0].asset_id, 7);
        assert!(queue.pending.pending_mesh_uploads[0].high_priority);
        assert!(!queue.mesh_upload_generation_is_current(7, first_generation));
    }

    #[test]
    fn empty_mesh_without_gpu_is_completed_without_defer_or_enqueue() {
        let mut queue = AssetTransferQueue::new();
        let mut shm = SharedMemoryAccessor::new(String::new());
        let data = MeshUploadData {
            asset_id: 7,
            buffer: SharedMemoryBufferDescriptor {
                length: 0,
                ..Default::default()
            },
            ..Default::default()
        };

        let result = try_process_mesh_upload(&mut queue, data, Some(&mut shm), None);

        assert_eq!(result.as_ref().map(|result| result.asset_id), Some(7));
        assert_eq!(
            result.as_ref().map(|result| result.instance_changed),
            Some(true)
        );
        assert!(queue.pending.pending_mesh_uploads.is_empty());
        assert_eq!(queue.integrator.total_queued(), 0);
        assert!(queue.pools.mesh_pool.is_empty());
    }

    #[test]
    fn mesh_upload_without_shared_memory_completes_failure_without_enqueue() {
        let mut queue = AssetTransferQueue::new();

        let result = try_process_mesh_upload(&mut queue, upload(9), None, None);

        assert_eq!(result.as_ref().map(|result| result.asset_id), Some(9));
        assert_eq!(
            result.as_ref().map(|result| result.instance_changed),
            Some(false)
        );
        assert!(queue.pending.pending_mesh_uploads.is_empty());
        assert_eq!(queue.integrator.total_queued(), 0);
        assert!(queue.pools.mesh_pool.is_empty());
    }
}
