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
use super::super::pending::PendingTextureUpload;
use super::PENDING_CUBEMAP_UPLOAD_WARN_THRESHOLD;
use super::allocations::flush_pending_cubemap_allocations;
use super::texture_common::{TextureUploadAdmission, admit_texture_upload_data};

fn send_cubemap_result(
    ipc: Option<&mut DualQueueIpc>,
    asset_id: i32,
    update: i32,
    instance_changed: bool,
) {
    let Some(ipc) = ipc else {
        return;
    };
    let _ = ipc.send_background_reliable(RendererCommand::SetCubemapResult(SetCubemapResult {
        asset_id,
        r#type: TextureUpdateResultType(update),
        instance_changed,
    }));
}

/// Handle [`SetCubemapFormat`](crate::shared::SetCubemapFormat).
pub fn on_set_cubemap_format(
    queue: &mut AssetTransferQueue,
    f: SetCubemapFormat,
    ipc: Option<&mut DualQueueIpc>,
) {
    let id = f.asset_id;
    let format_generation_changed = queue
        .catalogs
        .cubemap_formats
        .get(&id)
        .is_none_or(|old| !cubemap_format_shape_matches(old, &f));
    if format_generation_changed {
        queue.begin_cubemap_upload_generation(id);
    }
    queue.catalogs.cubemap_formats.insert(id, f.clone());
    let props = queue.catalogs.cubemap_properties.get(&id).cloned();
    let Some(device) = queue.gpu.gpu_device.clone() else {
        send_cubemap_result(
            ipc,
            id,
            TextureUpdateResultType::FORMAT_SET,
            queue.pools.cubemap_pool.get(id).is_none(),
        );
        return;
    };
    let Some(limits) = queue.gpu.gpu_limits.as_ref() else {
        logger::warn!("cubemap {id}: gpu_limits missing; format deferred until attach");
        send_cubemap_result(
            ipc,
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
        replay_pending_cubemap_uploads_for_asset(queue, id);
        send_cubemap_result(ipc, id, TextureUpdateResultType::FORMAT_SET, false);
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
        send_cubemap_result(ipc, id, TextureUpdateResultType::FORMAT_SET, false);
        return;
    };
    let existed_before = queue.pools.cubemap_pool.insert(tex);
    replay_pending_cubemap_uploads_for_asset(queue, id);
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
    _ipc: Option<&mut DualQueueIpc>,
) {
    let Some(d) = admit_texture_upload_data(TextureUploadAdmission {
        asset_id: d.asset_id,
        payload_len: d.data.length,
        data: d,
        kind: "cubemap",
        format_command: "SetCubemapData",
        pending_warn_threshold: PENDING_CUBEMAP_UPLOAD_WARN_THRESHOLD,
        queue,
        has_format: |queue, id| queue.catalogs.cubemap_formats.contains_key(&id),
        pending_len: |queue| queue.pending.pending_cubemap_uploads.len(),
        push_pending: push_pending_cubemap_upload,
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

    enqueue_cubemap_upload_task(queue, d);
}

/// Replay pending cubemap data after GPU attach.
pub fn try_cubemap_upload_with_device(
    queue: &mut AssetTransferQueue,
    pending: PendingTextureUpload<SetCubemapData>,
    _shm: &mut SharedMemoryAccessor,
    _ipc: Option<&mut DualQueueIpc>,
    _consume_texture_upload_budget: bool,
) {
    if pending_cubemap_upload_is_stale(queue, &pending) {
        logger::trace!(
            "cubemap {}: dropped stale deferred upload generation {:?}",
            pending.data.asset_id,
            pending.generation
        );
        return;
    }
    if !enqueue_cubemap_upload_task(queue, pending.data.clone()) {
        queue.pending.pending_cubemap_uploads.push_back(pending);
    }
}

/// Remove a cubemap asset from CPU tables and the pool.
pub fn on_unload_cubemap(queue: &mut AssetTransferQueue, u: UnloadCubemap) {
    let id = u.asset_id;
    queue.catalogs.cubemap_formats.remove(&id);
    queue.catalogs.cubemap_properties.remove(&id);
    queue.invalidate_cubemap_upload_generation(id);
    remove_pending_cubemap_uploads_for_asset(queue, id);
    if let Some(cubemap) = queue.pools.cubemap_pool.take(id) {
        queue
            .integrator_mut()
            .enqueue_delayed_removal(RetiredAssetResource::Cubemap(cubemap));
    }
}

fn enqueue_cubemap_upload_task(queue: &mut AssetTransferQueue, d: SetCubemapData) -> bool {
    let id = d.asset_id;
    let Some(fmt) = queue.catalogs.cubemap_formats.get(&id).cloned() else {
        logger::warn!("cubemap {id}: missing format");
        return false;
    };
    let Some(wgpu_fmt) = queue.pools.cubemap_pool.get(id).map(|t| t.wgpu_format) else {
        logger::warn!("cubemap {id}: missing GPU texture");
        return false;
    };
    let Some(generation) = queue.current_cubemap_upload_generation(id) else {
        logger::warn!("cubemap {id}: missing upload generation");
        return false;
    };
    let high = d.high_priority;
    let task = AssetTask::Cubemap(CubemapUploadTask::new(d, fmt, wgpu_fmt, generation));
    queue.integrator_mut().enqueue(task, high);
    true
}

fn replay_pending_cubemap_uploads_for_asset(queue: &mut AssetTransferQueue, asset_id: i32) {
    let pending = std::mem::take(&mut queue.pending.pending_cubemap_uploads);
    let mut replayed = 0usize;
    let mut dropped_stale = 0usize;
    for pending_upload in pending {
        if pending_upload.data.asset_id == asset_id {
            if pending_cubemap_upload_is_stale(queue, &pending_upload) {
                dropped_stale += 1;
            } else if enqueue_cubemap_upload_task(queue, pending_upload.data.clone()) {
                replayed += 1;
            } else {
                queue
                    .pending
                    .pending_cubemap_uploads
                    .push_back(pending_upload);
            }
        } else {
            queue
                .pending
                .pending_cubemap_uploads
                .push_back(pending_upload);
        }
    }
    if replayed > 0 {
        logger::debug!("cubemap {asset_id}: replayed {replayed} deferred data upload(s)");
    }
    if dropped_stale > 0 {
        logger::debug!("cubemap {asset_id}: dropped {dropped_stale} stale deferred upload(s)");
    }
}

fn remove_pending_cubemap_uploads_for_asset(queue: &mut AssetTransferQueue, asset_id: i32) {
    let pending_before = queue.pending.pending_cubemap_uploads.len();
    queue
        .pending
        .pending_cubemap_uploads
        .retain(|upload| upload.data.asset_id != asset_id);
    let removed = pending_before.saturating_sub(queue.pending.pending_cubemap_uploads.len());
    if removed > 0 {
        logger::debug!("cubemap {asset_id}: removed {removed} deferred upload(s) on unload");
    }
}

fn push_pending_cubemap_upload(queue: &mut AssetTransferQueue, data: SetCubemapData) {
    let generation = queue.current_cubemap_upload_generation(data.asset_id);
    queue
        .pending
        .pending_cubemap_uploads
        .push_back(PendingTextureUpload::new(data, generation));
}

fn pending_cubemap_upload_is_stale(
    queue: &AssetTransferQueue,
    pending: &PendingTextureUpload<SetCubemapData>,
) -> bool {
    pending.generation.is_some_and(|generation| {
        !queue.cubemap_upload_generation_is_current(pending.data.asset_id, generation)
    })
}

fn cubemap_format_shape_matches(a: &SetCubemapFormat, b: &SetCubemapFormat) -> bool {
    a.size == b.size && a.mipmap_count == b.mipmap_count && a.format == b.format
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
        assert_eq!(queue.pending.pending_cubemap_uploads[0].data.asset_id, 11);
    }

    #[test]
    fn data_with_format_but_no_gpu_is_deferred() {
        let mut queue = AssetTransferQueue::new();
        on_set_cubemap_format(&mut queue, format(11), None);

        on_set_cubemap_data(&mut queue, data(11), None, None);

        assert_eq!(queue.pending.pending_cubemap_uploads.len(), 1);
        assert_eq!(queue.pending.pending_cubemap_uploads[0].data.asset_id, 11);
    }

    #[test]
    fn pending_upload_with_replaced_format_generation_is_dropped() {
        let mut queue = AssetTransferQueue::new();
        on_set_cubemap_format(&mut queue, format(11), None);
        on_set_cubemap_data(&mut queue, data(11), None, None);
        let first_generation = queue.current_cubemap_upload_generation(11);

        on_set_cubemap_format(
            &mut queue,
            SetCubemapFormat {
                size: 128,
                ..format(11)
            },
            None,
        );
        replay_pending_cubemap_uploads_for_asset(&mut queue, 11);

        assert_ne!(
            queue.current_cubemap_upload_generation(11),
            first_generation
        );
        assert!(queue.pending.pending_cubemap_uploads.is_empty());
    }
}
