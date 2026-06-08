//! Cubemap format/properties/data IPC and cooperative [`super::super::cubemap_task::CubemapUploadTask`] integration.

use crate::gpu_pools::GpuCubemap;
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::shared::{
    RendererCommand, SetCubemapData, SetCubemapFormat, SetCubemapProperties, SetCubemapResult,
    TextureUpdateResultType, UnloadCubemap,
};

use super::super::AssetTransferQueue;
use super::super::cubemap_task::CubemapUploadTask;
use super::super::integrator::{AssetTask, RetiredAssetResource};
use super::MAX_PENDING_CUBEMAP_UPLOADS;
use super::allocations::flush_pending_cubemap_allocations;
use super::texture_common::{TextureUploadAdmission, admit_texture_upload_data};

enum CubemapUploadEnqueueResult {
    Enqueued,
    Defer(SetCubemapData),
    QueueFull { asset_id: i32 },
}

fn send_cubemap_result(
    ipc: Option<&mut DualQueueIpc>,
    asset_id: i32,
    update: i32,
    instance_changed: bool,
) {
    let Some(ipc) = ipc else {
        return;
    };
    if !ipc.send_background_reliable(RendererCommand::SetCubemapResult(SetCubemapResult {
        asset_id,
        r#type: TextureUpdateResultType(update),
        instance_changed,
    })) {
        logger::warn!("cubemap {asset_id}: failed to enqueue reliable SetCubemapResult");
    }
}

/// Handle [`SetCubemapFormat`](crate::shared::SetCubemapFormat).
pub fn on_set_cubemap_format(
    queue: &mut AssetTransferQueue,
    f: SetCubemapFormat,
    ipc: Option<&mut DualQueueIpc>,
) {
    let id = f.asset_id;
    let mut ipc = ipc;
    queue.catalogs.cubemap_formats.insert(id, f.clone());
    let props = queue.catalogs.cubemap_properties.get(&id).cloned();
    let Some(device) = queue.gpu.gpu_device.clone() else {
        send_cubemap_result(
            ipc.as_deref_mut(),
            id,
            TextureUpdateResultType::FORMAT_SET,
            queue.pools.cubemap_pool.get(id).is_none(),
        );
        return;
    };
    let Some(limits) = queue.gpu.gpu_limits.as_ref() else {
        logger::warn!("cubemap {id}: gpu_limits missing; format deferred until attach");
        send_cubemap_result(
            ipc.as_deref_mut(),
            id,
            TextureUpdateResultType::FORMAT_SET,
            queue.pools.cubemap_pool.get(id).is_none(),
        );
        return;
    };
    if let Some(cubemap) = queue.pools.cubemap_pool.get_mut(id)
        && cubemap.allocation_matches_format(device.as_ref(), limits.as_ref(), &f)
    {
        cubemap.apply_format_metadata(&f, props.as_ref());
        replay_pending_cubemap_uploads_for_asset(queue, id, ipc.as_deref_mut());
        send_cubemap_result(
            ipc.as_deref_mut(),
            id,
            TextureUpdateResultType::FORMAT_SET,
            false,
        );
        logger::trace!(
            "cubemap {} format {:?} size={} mips={} reused resident allocation",
            id,
            f.format,
            f.size,
            f.mipmap_count
        );
        return;
    }
    let Some(tex) =
        GpuCubemap::new_from_format(device.as_ref(), limits.as_ref(), &f, props.as_ref())
    else {
        logger::warn!("cubemap {id}: SetCubemapFormat rejected (bad size or device)");
        send_cubemap_result(
            ipc.as_deref_mut(),
            id,
            TextureUpdateResultType::FORMAT_SET,
            false,
        );
        return;
    };
    let existed_before = queue.pools.cubemap_pool.insert(tex);
    replay_pending_cubemap_uploads_for_asset(queue, id, ipc.as_deref_mut());
    send_cubemap_result(
        ipc,
        id,
        TextureUpdateResultType::FORMAT_SET,
        !existed_before,
    );
    logger::trace!(
        "cubemap {} format {:?} size={} mips={} (resident_bytes~={})",
        id,
        f.format,
        f.size,
        f.mipmap_count,
        queue
            .pools
            .cubemap_pool
            .accounting()
            .texture_resident_bytes()
    );
}

/// Handle [`SetCubemapProperties`](crate::shared::SetCubemapProperties).
pub fn on_set_cubemap_properties(
    queue: &mut AssetTransferQueue,
    p: SetCubemapProperties,
    ipc: Option<&mut DualQueueIpc>,
) {
    let id = p.asset_id;
    queue.catalogs.cubemap_properties.insert(id, p.clone());
    if let Some(t) = queue.pools.cubemap_pool.get_mut(id) {
        t.apply_properties(&p);
    }
    send_cubemap_result(ipc, id, TextureUpdateResultType::PROPERTIES_SET, false);
}

/// Enqueue [`SetCubemapData`] for time-sliced GPU integration.
pub fn on_set_cubemap_data(
    queue: &mut AssetTransferQueue,
    d: SetCubemapData,
    _shm: Option<&mut SharedMemoryAccessor>,
    ipc: Option<&mut DualQueueIpc>,
) {
    let Some(d) = admit_texture_upload_data(TextureUploadAdmission {
        asset_id: d.asset_id,
        payload_len: d.data.length,
        data: d,
        kind: "cubemap",
        format_command: "SetCubemapData",
        pending_warn_threshold: MAX_PENDING_CUBEMAP_UPLOADS,
        queue,
        has_format: |queue, id| queue.catalogs.cubemap_formats.contains_key(&id),
        pending_len: |queue| queue.pending.pending_cubemap_uploads.len(),
        push_pending: |queue, data| queue.pending.pending_cubemap_uploads.push_back(data),
        has_resident: |queue, id| queue.pools.cubemap_pool.get(id).is_some(),
        flush_allocations: flush_pending_cubemap_allocations,
    }) else {
        return;
    };
    let asset_id = d.asset_id;
    logger::trace!(
        "cubemap_upload enqueue asset_id={} payload_bytes={} high_priority={}",
        asset_id,
        d.data.length.max(0),
        d.high_priority,
    );

    let enqueue_result = enqueue_cubemap_upload_task(queue, d);
    handle_live_cubemap_upload_enqueue_result(queue, enqueue_result, ipc);
}

/// Replay pending cubemap data after GPU attach.
pub fn try_cubemap_upload_with_device(
    queue: &mut AssetTransferQueue,
    data: SetCubemapData,
    _shm: &mut SharedMemoryAccessor,
    ipc: Option<&mut DualQueueIpc>,
    _consume_texture_upload_budget: bool,
) {
    let enqueue_result = enqueue_cubemap_upload_task(queue, data);
    handle_replayed_cubemap_upload_enqueue_result(queue, enqueue_result, ipc);
}

/// Remove a cubemap asset from CPU tables and the pool.
pub fn on_unload_cubemap(queue: &mut AssetTransferQueue, u: UnloadCubemap) {
    let id = u.asset_id;
    queue.catalogs.cubemap_formats.remove(&id);
    queue.catalogs.cubemap_properties.remove(&id);
    remove_pending_cubemap_uploads_for_asset(queue, id);
    if let Some(cubemap) = queue.pools.cubemap_pool.take(id) {
        queue
            .integrator_mut()
            .enqueue_delayed_removal(RetiredAssetResource::Cubemap(cubemap));
    }
}

fn enqueue_cubemap_upload_task(
    queue: &mut AssetTransferQueue,
    d: SetCubemapData,
) -> CubemapUploadEnqueueResult {
    let id = d.asset_id;
    let Some(fmt) = queue.catalogs.cubemap_formats.get(&id).cloned() else {
        logger::warn!("cubemap {id}: missing format");
        return CubemapUploadEnqueueResult::Defer(d);
    };
    let Some(wgpu_fmt) = queue.pools.cubemap_pool.get(id).map(|t| t.wgpu_format) else {
        logger::warn!("cubemap {id}: missing GPU texture");
        return CubemapUploadEnqueueResult::Defer(d);
    };
    let high = d.high_priority;
    let task = AssetTask::Cubemap(CubemapUploadTask::new(d, fmt, wgpu_fmt));
    if queue.integrator_mut().enqueue(task, high) {
        CubemapUploadEnqueueResult::Enqueued
    } else {
        CubemapUploadEnqueueResult::QueueFull { asset_id: id }
    }
}

fn handle_live_cubemap_upload_enqueue_result(
    queue: &mut AssetTransferQueue,
    result: CubemapUploadEnqueueResult,
    ipc: Option<&mut DualQueueIpc>,
) {
    match result {
        CubemapUploadEnqueueResult::Enqueued => {}
        CubemapUploadEnqueueResult::Defer(data) => {
            retain_deferred_cubemap_upload(queue, data, "live enqueue prerequisites changed");
        }
        CubemapUploadEnqueueResult::QueueFull { asset_id } => {
            logger::warn!(
                "cubemap {asset_id}: rejected data upload because asset integrator is full"
            );
            send_cubemap_result(ipc, asset_id, TextureUpdateResultType::DATA_UPLOAD, false);
        }
    }
}

fn handle_replayed_cubemap_upload_enqueue_result(
    queue: &mut AssetTransferQueue,
    result: CubemapUploadEnqueueResult,
    ipc: Option<&mut DualQueueIpc>,
) -> bool {
    match result {
        CubemapUploadEnqueueResult::Enqueued => true,
        CubemapUploadEnqueueResult::Defer(data) => {
            retain_deferred_cubemap_upload(queue, data, "replay prerequisites changed");
            false
        }
        CubemapUploadEnqueueResult::QueueFull { asset_id } => {
            logger::warn!(
                "cubemap {asset_id}: dropping replayed upload because asset integrator is full"
            );
            send_cubemap_result(ipc, asset_id, TextureUpdateResultType::DATA_UPLOAD, false);
            false
        }
    }
}

fn retain_deferred_cubemap_upload(
    queue: &mut AssetTransferQueue,
    data: SetCubemapData,
    reason: &'static str,
) -> bool {
    if queue.pending.pending_cubemap_uploads.len() >= MAX_PENDING_CUBEMAP_UPLOADS {
        logger::warn!(
            "cubemap {}: dropping deferred upload because pending queue reached cap {} ({reason})",
            data.asset_id,
            MAX_PENDING_CUBEMAP_UPLOADS
        );
        return false;
    }
    queue.pending.pending_cubemap_uploads.push_back(data);
    true
}

fn replay_pending_cubemap_uploads_for_asset(
    queue: &mut AssetTransferQueue,
    asset_id: i32,
    ipc: Option<&mut DualQueueIpc>,
) {
    let pending = std::mem::take(&mut queue.pending.pending_cubemap_uploads);
    let mut replayed = 0usize;
    let mut ipc = ipc;
    for data in pending {
        if data.asset_id == asset_id {
            let enqueue_result = enqueue_cubemap_upload_task(queue, data);
            if handle_replayed_cubemap_upload_enqueue_result(
                queue,
                enqueue_result,
                ipc.as_deref_mut(),
            ) {
                replayed += 1;
            }
        } else {
            retain_deferred_cubemap_upload(queue, data, "unrelated replay");
        }
    }
    if replayed > 0 {
        logger::debug!("cubemap {asset_id}: replayed {replayed} deferred data upload(s)");
    }
}

fn remove_pending_cubemap_uploads_for_asset(queue: &mut AssetTransferQueue, asset_id: i32) {
    let pending_before = queue.pending.pending_cubemap_uploads.len();
    queue
        .pending
        .pending_cubemap_uploads
        .retain(|upload| upload.asset_id != asset_id);
    let removed = pending_before.saturating_sub(queue.pending.pending_cubemap_uploads.len());
    if removed > 0 {
        logger::debug!("cubemap {asset_id}: removed {removed} deferred upload(s) on unload");
    }
}

#[cfg(test)]
mod tests {
    use crate::shared::buffer::SharedMemoryBufferDescriptor;
    use crate::shared::{TextureFilterMode, TextureFormat};

    use super::*;

    fn format(asset_id: i32) -> SetCubemapFormat {
        SetCubemapFormat {
            asset_id,
            size: 64,
            mipmap_count: 1,
            format: TextureFormat::RGBA32,
            ..Default::default()
        }
    }

    fn data(asset_id: i32) -> SetCubemapData {
        SetCubemapData {
            asset_id,
            data: SharedMemoryBufferDescriptor {
                length: 16,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn format_without_gpu_updates_catalog_but_not_resident_pool() {
        let mut queue = AssetTransferQueue::new();

        on_set_cubemap_format(&mut queue, format(11), None);

        assert_eq!(
            queue.catalogs.cubemap_formats.get(&11).map(|fmt| fmt.size),
            Some(64)
        );
        assert!(queue.pools.cubemap_pool.get(11).is_none());
    }

    #[test]
    fn properties_without_resident_cubemap_update_catalog() {
        let mut queue = AssetTransferQueue::new();

        on_set_cubemap_properties(
            &mut queue,
            SetCubemapProperties {
                asset_id: 11,
                filter_mode: TextureFilterMode::Trilinear,
                aniso_level: 8,
                ..Default::default()
            },
            None,
        );

        let props = queue
            .catalogs
            .cubemap_properties
            .get(&11)
            .expect("stored props");
        assert_eq!(props.filter_mode, TextureFilterMode::Trilinear);
        assert_eq!(props.aniso_level, 8);
    }

    #[test]
    fn data_without_format_is_deferred_until_format() {
        let mut queue = AssetTransferQueue::new();

        on_set_cubemap_data(&mut queue, data(11), None, None);

        assert_eq!(queue.pending.pending_cubemap_uploads.len(), 1);
        assert_eq!(queue.pending.pending_cubemap_uploads[0].asset_id, 11);
    }

    #[test]
    fn data_with_format_but_no_gpu_is_deferred() {
        let mut queue = AssetTransferQueue::new();
        on_set_cubemap_format(&mut queue, format(11), None);

        on_set_cubemap_data(&mut queue, data(11), None, None);

        assert_eq!(queue.pending.pending_cubemap_uploads.len(), 1);
        assert_eq!(queue.pending.pending_cubemap_uploads[0].asset_id, 11);
    }
}
