//! Point-render buffer ingestion and catalog lifetime.

use std::sync::Arc;

use crate::assets::asset_transfer_queue::AssetTransferQueue;
use crate::assets::asset_transfer_queue::catalogs::PointRenderBufferPayload;
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::shared::{PointRenderBufferConsumed, PointRenderBufferUnload, PointRenderBufferUpload};

/// Copies one [`PointRenderBufferUpload`] payload from shared memory into the catalog.
pub fn on_point_render_buffer_upload(
    queue: &mut AssetTransferQueue,
    upload: PointRenderBufferUpload,
    shm: Option<&mut SharedMemoryAccessor>,
    _ipc: Option<&mut DualQueueIpc>,
) {
    let Some(shm) = shm else {
        logger::warn!(
            "point render buffer {}: upload skipped because shared memory is unavailable",
            upload.asset_id
        );
        return;
    };
    let want = upload.buffer.length.max(0) as usize;
    let Some(payload) = shm.with_read_bytes(&upload.buffer, |raw| {
        if raw.len() < want {
            logger::warn!(
                "point render buffer {}: payload shorter than descriptor (need {}, got {})",
                upload.asset_id,
                want,
                raw.len()
            );
            return None;
        }
        Some(Arc::from(&raw[..want]))
    }) else {
        logger::warn!(
            "point render buffer {}: failed to read shared-memory payload",
            upload.asset_id
        );
        return;
    };
    queue.catalogs.point_render_buffers.insert(
        upload.asset_id,
        PointRenderBufferPayload {
            asset_id: upload.asset_id,
            count: upload.count,
            positions_offset: upload.positions_offset,
            rotations_offset: upload.rotations_offset,
            sizes_offset: upload.sizes_offset,
            colors_offset: upload.colors_offset,
            frame_indexes_offset: upload.frame_indexes_offset,
            frame_grid_size: upload.frame_grid_size,
            payload,
        },
    );
}

/// Marks one point-render buffer as consumed by the host.
pub fn on_point_render_buffer_consumed(
    _queue: &mut AssetTransferQueue,
    _consumed: PointRenderBufferConsumed,
) {
    // Host command is informational in the current pipeline.
}

/// Removes one point-render buffer payload from the catalog.
pub fn on_point_render_buffer_unload(
    queue: &mut AssetTransferQueue,
    unload: PointRenderBufferUnload,
) {
    queue.catalogs.point_render_buffers.remove(&unload.asset_id);
}
