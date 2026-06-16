//! Texture2D format/properties/data IPC and cooperative [`super::super::texture_task::TextureUploadTask`] integration.

use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::shared::{
    RendererCommand, SetTexture2DData, SetTexture2DFormat, SetTexture2DProperties,
    SetTexture2DResult, TextureUpdateResultType, UnloadTexture2D,
};

use super::super::AssetTransferQueue;
use super::super::integrator::{AssetTask, RetiredAssetResource};
use super::super::pending::PendingTextureUpload;
use super::super::texture_task::TextureUploadTask;
use super::PENDING_TEXTURE_UPLOAD_WARN_THRESHOLD;
use super::allocations::flush_pending_texture_allocations;
use super::texture_common::{TextureUploadAdmission, admit_texture_upload_data};

enum TextureUploadEnqueueResult {
    Enqueued,
    Defer(SetTexture2DData),
    QueueFull { asset_id: i32 },
}

fn send_texture_2d_result(
    ipc: Option<&mut DualQueueIpc>,
    asset_id: i32,
    update: i32,
    instance_changed: bool,
) {
    let Some(ipc) = ipc else {
        return;
    };
    if !ipc.send_background_reliable(RendererCommand::SetTexture2DResult(SetTexture2DResult {
        asset_id,
        r#type: TextureUpdateResultType(update),
        instance_changed,
    })) {
        logger::warn!("texture {asset_id}: failed to enqueue reliable SetTexture2DResult");
    }
}

/// Handle [`SetTexture2DFormat`](crate::shared::SetTexture2DFormat).
pub fn on_set_texture_2d_format(
    queue: &mut AssetTransferQueue,
    f: SetTexture2DFormat,
    ipc: Option<&mut DualQueueIpc>,
) {
    let id = f.asset_id;
    let mut ipc = ipc;
    let format_generation_changed = queue
        .catalogs
        .texture_formats
        .get(&id)
        .is_none_or(|old| !texture_2d_format_shape_matches(old, &f));
    if format_generation_changed {
        queue.begin_texture_upload_generation(id);
    }
    queue.catalogs.texture_formats.insert(id, f.clone());
    let props = queue.catalogs.texture_properties.get(&id).cloned();
    let Some(device) = queue.gpu.gpu_device.clone() else {
        send_texture_2d_result(
            ipc.as_deref_mut(),
            id,
            TextureUpdateResultType::FORMAT_SET,
            queue.pools.texture_pool.get(id).is_none(),
        );
        return;
    };
    let Some(limits) = queue.gpu.gpu_limits.as_ref() else {
        logger::warn!("texture {id}: gpu_limits missing; format deferred until attach");
        send_texture_2d_result(
            ipc.as_deref_mut(),
            id,
            TextureUpdateResultType::FORMAT_SET,
            queue.pools.texture_pool.get(id).is_none(),
        );
        return;
    };
    if let Some(texture) = queue.pools.texture_pool.get_mut(id)
        && texture.allocation_matches_format(device.as_ref(), limits.as_ref(), &f)
    {
        texture.apply_format_metadata(&f, props.as_ref());
        replay_pending_texture_uploads_for_asset(queue, id, ipc.as_deref_mut());
        send_texture_2d_result(
            ipc.as_deref_mut(),
            id,
            TextureUpdateResultType::FORMAT_SET,
            false,
        );
        logger::trace!(
            "texture {} format {:?} {}x{} mips={} reused resident allocation",
            id,
            f.format,
            f.width,
            f.height,
            f.mipmap_count
        );
        return;
    }
    let Some(tex) = crate::gpu_pools::GpuTexture2d::new_from_format(
        device.as_ref(),
        limits.as_ref(),
        &f,
        props.as_ref(),
    ) else {
        logger::warn!("texture {id}: SetTexture2DFormat rejected (bad size or device)");
        send_texture_2d_result(
            ipc.as_deref_mut(),
            id,
            TextureUpdateResultType::FORMAT_SET,
            false,
        );
        return;
    };
    let existed_before = queue.pools.texture_pool.insert(tex);
    replay_pending_texture_uploads_for_asset(queue, id, ipc.as_deref_mut());
    send_texture_2d_result(
        ipc,
        id,
        TextureUpdateResultType::FORMAT_SET,
        !existed_before,
    );
    logger::trace!(
        "texture {} format {:?} {}x{} mips={} (resident_bytes~={})",
        id,
        f.format,
        f.width,
        f.height,
        f.mipmap_count,
        queue
            .pools
            .texture_pool
            .accounting()
            .texture_resident_bytes()
    );
}

/// Handle [`SetTexture2DProperties`](crate::shared::SetTexture2DProperties).
pub fn on_set_texture_2d_properties(
    queue: &mut AssetTransferQueue,
    p: SetTexture2DProperties,
    ipc: Option<&mut DualQueueIpc>,
) {
    let id = p.asset_id;
    queue.catalogs.texture_properties.insert(id, p.clone());
    if let Some(t) = queue.pools.texture_pool.get_mut(id) {
        t.apply_properties(&p);
    }
    send_texture_2d_result(ipc, id, TextureUpdateResultType::PROPERTIES_SET, false);
}

/// Enqueue [`SetTexture2DData`] for time-sliced GPU integration ([`super::super::integrator::drain_asset_tasks`]).
pub fn on_set_texture_2d_data(
    queue: &mut AssetTransferQueue,
    d: SetTexture2DData,
    _shm: Option<&mut SharedMemoryAccessor>,
    ipc: Option<&mut DualQueueIpc>,
) {
    let Some(d) = admit_texture_upload_data(TextureUploadAdmission {
        asset_id: d.asset_id,
        payload_len: d.data.length,
        data: d,
        kind: "texture",
        format_command: "SetTexture2DData",
        pending_warn_threshold: PENDING_TEXTURE_UPLOAD_WARN_THRESHOLD,
        queue,
        has_format: |queue, id| queue.catalogs.texture_formats.contains_key(&id),
        pending_len: |queue| queue.pending.pending_texture_uploads.len(),
        push_pending: push_pending_texture_upload,
        has_resident: |queue, id| queue.pools.texture_pool.get(id).is_some(),
        flush_allocations: flush_pending_texture_allocations,
    }) else {
        return;
    };
    let asset_id = d.asset_id;
    logger::trace!(
        "texture_upload enqueue asset_id={} payload_bytes={} high_priority={} has_region={} mip_count={} start_mip={}",
        asset_id,
        d.data.length.max(0),
        d.high_priority,
        d.hint.has_region != 0,
        d.mip_map_sizes.len(),
        d.start_mip_level,
    );

    let enqueue_result = enqueue_texture_upload_task(queue, d);
    handle_live_texture_upload_enqueue_result(queue, enqueue_result, ipc);
}

/// Replay pending texture data after GPU attach (enqueue only; caller runs [`super::super::integrator::drain_asset_tasks_unbounded`]).
pub fn try_texture_upload_with_device(
    queue: &mut AssetTransferQueue,
    pending: PendingTextureUpload<SetTexture2DData>,
    _shm: &mut SharedMemoryAccessor,
    ipc: Option<&mut DualQueueIpc>,
    _consume_texture_upload_budget: bool,
) {
    if pending_texture_upload_is_stale(queue, &pending) {
        logger::trace!(
            "texture {}: dropped stale deferred upload generation {:?}",
            pending.data.asset_id,
            pending.generation
        );
        return;
    }
    let enqueue_result = enqueue_texture_upload_task(queue, pending.data.clone());
    handle_replayed_texture_upload_enqueue_result(queue, pending, enqueue_result, ipc);
}

/// Remove a texture asset from CPU tables and the pool.
pub fn on_unload_texture_2d(queue: &mut AssetTransferQueue, u: UnloadTexture2D) {
    let id = u.asset_id;
    queue.catalogs.texture_formats.remove(&id);
    queue.catalogs.texture_properties.remove(&id);
    queue.invalidate_texture_upload_generation(id);
    remove_pending_texture_uploads_for_asset(queue, id);
    if let Some(texture) = queue.pools.texture_pool.take(id) {
        queue
            .integrator_mut()
            .enqueue_delayed_removal(RetiredAssetResource::Texture2d(texture));
    }
}

fn enqueue_texture_upload_task(
    queue: &mut AssetTransferQueue,
    d: SetTexture2DData,
) -> TextureUploadEnqueueResult {
    let id = d.asset_id;
    let Some(fmt) = queue.catalogs.texture_formats.get(&id).cloned() else {
        logger::warn!("texture {id}: missing format");
        return TextureUploadEnqueueResult::Defer(d);
    };
    let Some(wgpu_fmt) = queue.pools.texture_pool.get(id).map(|t| t.wgpu_format) else {
        logger::warn!("texture {id}: missing GPU texture");
        return TextureUploadEnqueueResult::Defer(d);
    };
    let Some(generation) = queue.current_texture_upload_generation(id) else {
        logger::warn!("texture {id}: missing upload generation");
        return TextureUploadEnqueueResult::Defer(d);
    };
    let high = d.high_priority;
    let task = AssetTask::Texture(TextureUploadTask::new(d, fmt, wgpu_fmt, generation));
    if queue.integrator_mut().enqueue(task, high) {
        TextureUploadEnqueueResult::Enqueued
    } else {
        TextureUploadEnqueueResult::QueueFull { asset_id: id }
    }
}

fn handle_live_texture_upload_enqueue_result(
    queue: &mut AssetTransferQueue,
    result: TextureUploadEnqueueResult,
    ipc: Option<&mut DualQueueIpc>,
) {
    match result {
        TextureUploadEnqueueResult::Enqueued => {}
        TextureUploadEnqueueResult::Defer(data) => {
            retain_deferred_texture_upload(queue, data, "live enqueue prerequisites changed");
        }
        TextureUploadEnqueueResult::QueueFull { asset_id } => {
            logger::warn!(
                "texture {asset_id}: rejected data upload because asset integrator is full"
            );
            send_texture_2d_result(ipc, asset_id, TextureUpdateResultType::DATA_UPLOAD, false);
        }
    }
}

fn handle_replayed_texture_upload_enqueue_result(
    queue: &mut AssetTransferQueue,
    pending: PendingTextureUpload<SetTexture2DData>,
    result: TextureUploadEnqueueResult,
    ipc: Option<&mut DualQueueIpc>,
) -> bool {
    match result {
        TextureUploadEnqueueResult::Enqueued => true,
        TextureUploadEnqueueResult::Defer(_data) => {
            retain_deferred_texture_upload_record(queue, pending, "replay prerequisites changed");
            false
        }
        TextureUploadEnqueueResult::QueueFull { asset_id } => {
            logger::warn!(
                "texture {asset_id}: dropping replayed upload because asset integrator is full"
            );
            send_texture_2d_result(ipc, asset_id, TextureUpdateResultType::DATA_UPLOAD, false);
            false
        }
    }
}

fn retain_deferred_texture_upload(
    queue: &mut AssetTransferQueue,
    data: SetTexture2DData,
    reason: &'static str,
) -> bool {
    let generation = queue.current_texture_upload_generation(data.asset_id);
    retain_deferred_texture_upload_record(
        queue,
        PendingTextureUpload::new(data, generation),
        reason,
    )
}

fn retain_deferred_texture_upload_record(
    queue: &mut AssetTransferQueue,
    pending: PendingTextureUpload<SetTexture2DData>,
    reason: &'static str,
) -> bool {
    if queue.pending.pending_texture_uploads.len() >= PENDING_TEXTURE_UPLOAD_WARN_THRESHOLD {
        logger::warn!(
            "texture {}: dropping deferred upload because pending queue reached cap {} ({reason})",
            pending.data.asset_id,
            PENDING_TEXTURE_UPLOAD_WARN_THRESHOLD
        );
        return false;
    }
    queue.pending.pending_texture_uploads.push_back(pending);
    true
}

fn replay_pending_texture_uploads_for_asset(
    queue: &mut AssetTransferQueue,
    asset_id: i32,
    ipc: Option<&mut DualQueueIpc>,
) {
    let pending = std::mem::take(&mut queue.pending.pending_texture_uploads);
    let mut replayed = 0usize;
    let mut dropped_stale = 0usize;
    let mut ipc = ipc;
    for pending_upload in pending {
        if pending_upload.data.asset_id == asset_id {
            if pending_texture_upload_is_stale(queue, &pending_upload) {
                dropped_stale += 1;
            } else {
                let enqueue_result =
                    enqueue_texture_upload_task(queue, pending_upload.data.clone());
                if handle_replayed_texture_upload_enqueue_result(
                    queue,
                    pending_upload,
                    enqueue_result,
                    ipc.as_deref_mut(),
                ) {
                    replayed += 1;
                }
            }
        } else {
            retain_deferred_texture_upload_record(queue, pending_upload, "unrelated replay");
        }
    }
    if replayed > 0 {
        logger::debug!("texture {asset_id}: replayed {replayed} deferred data upload(s)");
    }
    if dropped_stale > 0 {
        logger::debug!("texture {asset_id}: dropped {dropped_stale} stale deferred upload(s)");
    }
}

fn remove_pending_texture_uploads_for_asset(queue: &mut AssetTransferQueue, asset_id: i32) {
    let pending_before = queue.pending.pending_texture_uploads.len();
    queue
        .pending
        .pending_texture_uploads
        .retain(|upload| upload.data.asset_id != asset_id);
    let removed = pending_before.saturating_sub(queue.pending.pending_texture_uploads.len());
    if removed > 0 {
        logger::debug!("texture {asset_id}: removed {removed} deferred upload(s) on unload");
    }
}

fn push_pending_texture_upload(queue: &mut AssetTransferQueue, data: SetTexture2DData) {
    let generation = queue.current_texture_upload_generation(data.asset_id);
    queue
        .pending
        .pending_texture_uploads
        .push_back(PendingTextureUpload::new(data, generation));
}

fn pending_texture_upload_is_stale(
    queue: &AssetTransferQueue,
    pending: &PendingTextureUpload<SetTexture2DData>,
) -> bool {
    pending.generation.is_some_and(|generation| {
        !queue.texture_upload_generation_is_current(pending.data.asset_id, generation)
    })
}

fn texture_2d_format_shape_matches(a: &SetTexture2DFormat, b: &SetTexture2DFormat) -> bool {
    a.width == b.width
        && a.height == b.height
        && a.mipmap_count == b.mipmap_count
        && a.format == b.format
}

#[cfg(test)]
mod tests {
    use crate::shared::buffer::SharedMemoryBufferDescriptor;
    use crate::shared::{TextureFilterMode, TextureFormat, TextureWrapMode};

    use super::*;

    fn format(asset_id: i32) -> SetTexture2DFormat {
        SetTexture2DFormat {
            asset_id,
            width: 64,
            height: 32,
            mipmap_count: 1,
            format: TextureFormat::RGBA32,
            ..Default::default()
        }
    }

    fn data(asset_id: i32) -> SetTexture2DData {
        SetTexture2DData {
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

        on_set_texture_2d_format(&mut queue, format(7), None);

        assert_eq!(
            queue.catalogs.texture_formats.get(&7).map(|fmt| fmt.width),
            Some(64)
        );
        assert!(queue.pools.texture_pool.get(7).is_none());
    }

    #[test]
    fn properties_without_resident_texture_update_catalog() {
        let mut queue = AssetTransferQueue::new();

        on_set_texture_2d_properties(
            &mut queue,
            SetTexture2DProperties {
                asset_id: 7,
                filter_mode: TextureFilterMode::Trilinear,
                wrap_u: TextureWrapMode::Mirror,
                wrap_v: TextureWrapMode::Clamp,
                ..Default::default()
            },
            None,
        );

        let props = queue
            .catalogs
            .texture_properties
            .get(&7)
            .expect("stored props");
        assert_eq!(props.filter_mode, TextureFilterMode::Trilinear);
        assert_eq!(props.wrap_u, TextureWrapMode::Mirror);
        assert_eq!(props.wrap_v, TextureWrapMode::Clamp);
    }

    #[test]
    fn data_without_format_is_deferred_until_format() {
        let mut queue = AssetTransferQueue::new();

        on_set_texture_2d_data(&mut queue, data(7), None, None);

        assert_eq!(queue.pending.pending_texture_uploads.len(), 1);
        assert_eq!(queue.pending.pending_texture_uploads[0].data.asset_id, 7);
    }

    #[test]
    fn data_with_format_but_no_gpu_is_deferred() {
        let mut queue = AssetTransferQueue::new();
        on_set_texture_2d_format(&mut queue, format(7), None);

        on_set_texture_2d_data(&mut queue, data(7), None, None);

        assert_eq!(queue.pending.pending_texture_uploads.len(), 1);
        assert_eq!(queue.pending.pending_texture_uploads[0].data.asset_id, 7);
    }

    #[test]
    fn data_with_format_but_no_gpu_is_capped_at_pending_threshold() {
        let mut queue = AssetTransferQueue::new();
        on_set_texture_2d_format(&mut queue, format(7), None);

        for _ in 0..=PENDING_TEXTURE_UPLOAD_WARN_THRESHOLD {
            on_set_texture_2d_data(&mut queue, data(7), None, None);
        }

        assert_eq!(
            queue.pending.pending_texture_uploads.len(),
            PENDING_TEXTURE_UPLOAD_WARN_THRESHOLD
        );
    }

    #[test]
    fn unload_removes_deferred_texture_uploads_for_asset() {
        let mut queue = AssetTransferQueue::new();
        on_set_texture_2d_data(&mut queue, data(7), None, None);
        on_set_texture_2d_data(&mut queue, data(8), None, None);

        on_unload_texture_2d(&mut queue, UnloadTexture2D { asset_id: 7 });

        assert_eq!(queue.pending.pending_texture_uploads.len(), 1);
        assert_eq!(queue.pending.pending_texture_uploads[0].data.asset_id, 8);
    }

    #[test]
    fn pending_upload_with_replaced_format_generation_is_dropped() {
        let mut queue = AssetTransferQueue::new();
        on_set_texture_2d_format(&mut queue, format(7), None);
        on_set_texture_2d_data(&mut queue, data(7), None, None);
        let first_generation = queue.current_texture_upload_generation(7);

        on_set_texture_2d_format(
            &mut queue,
            SetTexture2DFormat {
                width: 128,
                ..format(7)
            },
            None,
        );
        replay_pending_texture_uploads_for_asset(&mut queue, 7, None);

        assert_ne!(queue.current_texture_upload_generation(7), first_generation);
        assert!(queue.pending.pending_texture_uploads.is_empty());
    }
}
