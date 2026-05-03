//! Texture2D format/properties/data IPC and cooperative [`super::super::texture_task::TextureUploadTask`] integration.

use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::shared::{
    RendererCommand, SetTexture2DData, SetTexture2DFormat, SetTexture2DProperties,
    SetTexture2DResult, TextureUpdateResultType, UnloadTexture2D,
};

use super::super::AssetTransferQueue;
use super::super::integrator::AssetTask;
use super::super::texture_task::TextureUploadTask;
use super::MAX_PENDING_TEXTURE_UPLOADS;
use super::allocations::flush_pending_texture_allocations;
use super::texture_common::{TextureUploadAdmission, admit_texture_upload_data};

fn send_texture_2d_result(
    ipc: Option<&mut DualQueueIpc>,
    asset_id: i32,
    update: i32,
    instance_changed: bool,
) {
    let Some(ipc) = ipc else {
        return;
    };
    let _ = ipc.send_background(RendererCommand::SetTexture2DResult(SetTexture2DResult {
        asset_id,
        r#type: TextureUpdateResultType(update),
        instance_changed,
    }));
}

/// Handle [`SetTexture2DFormat`](crate::shared::SetTexture2DFormat).
pub fn on_set_texture_2d_format(
    queue: &mut AssetTransferQueue,
    f: SetTexture2DFormat,
    ipc: Option<&mut DualQueueIpc>,
) {
    let id = f.asset_id;
    queue.catalogs.texture_formats.insert(id, f.clone());
    let props = queue.catalogs.texture_properties.get(&id);
    let Some(device) = queue.gpu.gpu_device.clone() else {
        send_texture_2d_result(
            ipc,
            id,
            TextureUpdateResultType::FORMAT_SET,
            queue.pools.texture_pool.get(id).is_none(),
        );
        return;
    };
    let Some(limits) = queue.gpu.gpu_limits.as_ref() else {
        logger::warn!("texture {id}: gpu_limits missing; format deferred until attach");
        send_texture_2d_result(
            ipc,
            id,
            TextureUpdateResultType::FORMAT_SET,
            queue.pools.texture_pool.get(id).is_none(),
        );
        return;
    };
    let Some(tex) = crate::gpu_pools::GpuTexture2d::new_from_format(
        device.as_ref(),
        limits.as_ref(),
        &f,
        props,
    ) else {
        logger::warn!("texture {id}: SetTexture2DFormat rejected (bad size or device)");
        return;
    };
    let existed_before = queue.pools.texture_pool.insert(tex);
    queue.maybe_warn_texture_vram_budget();
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
    _ipc: Option<&mut DualQueueIpc>,
) {
    let Some(d) = admit_texture_upload_data(TextureUploadAdmission {
        asset_id: d.asset_id,
        payload_len: d.data.length,
        data: d,
        kind: "texture",
        format_command: "SetTexture2DData",
        max_pending: MAX_PENDING_TEXTURE_UPLOADS,
        queue,
        has_format: |queue, id| queue.catalogs.texture_formats.contains_key(&id),
        pending_len: |queue| queue.pending.pending_texture_uploads.len(),
        push_pending: |queue, data| queue.pending.pending_texture_uploads.push_back(data),
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

    if !enqueue_texture_upload_task(queue, d) {
        logger::warn!("texture {asset_id}: asset integration queue full; dropping data upload",);
    }
}

/// Replay pending texture data after GPU attach (enqueue only; caller runs [`super::super::integrator::drain_asset_tasks_unbounded`]).
pub fn try_texture_upload_with_device(
    queue: &mut AssetTransferQueue,
    data: SetTexture2DData,
    _shm: &mut SharedMemoryAccessor,
    _ipc: Option<&mut DualQueueIpc>,
    _consume_texture_upload_budget: bool,
) {
    let _ = enqueue_texture_upload_task(queue, data);
}

/// Remove a texture asset from CPU tables and the pool.
pub fn on_unload_texture_2d(queue: &mut AssetTransferQueue, u: UnloadTexture2D) {
    let id = u.asset_id;
    queue.catalogs.texture_formats.remove(&id);
    queue.catalogs.texture_properties.remove(&id);
    if queue.pools.texture_pool.remove(id) {
        logger::info!(
            "texture {id} unloaded (mesh~={} tex~={} total~={})",
            queue.pools.mesh_pool.accounting().mesh_resident_bytes(),
            queue
                .pools
                .texture_pool
                .accounting()
                .texture_resident_bytes(),
            queue.pools.mesh_pool.accounting().total_resident_bytes()
        );
    }
}

fn enqueue_texture_upload_task(queue: &mut AssetTransferQueue, d: SetTexture2DData) -> bool {
    let id = d.asset_id;
    let Some(fmt) = queue.catalogs.texture_formats.get(&id).cloned() else {
        logger::warn!("texture {id}: missing format");
        return false;
    };
    let Some(wgpu_fmt) = queue.pools.texture_pool.get(id).map(|t| t.wgpu_format) else {
        logger::warn!("texture {id}: missing GPU texture");
        return false;
    };
    let high = d.high_priority;
    let task = AssetTask::Texture(TextureUploadTask::new(d, fmt, wgpu_fmt));
    queue.integrator_mut().try_enqueue(task, high)
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
    fn data_without_format_is_dropped_before_pending_queue() {
        let mut queue = AssetTransferQueue::new();

        on_set_texture_2d_data(&mut queue, data(7), None, None);

        assert!(queue.pending.pending_texture_uploads.is_empty());
    }

    #[test]
    fn data_with_format_but_no_gpu_is_deferred() {
        let mut queue = AssetTransferQueue::new();
        on_set_texture_2d_format(&mut queue, format(7), None);

        on_set_texture_2d_data(&mut queue, data(7), None, None);

        assert_eq!(queue.pending.pending_texture_uploads.len(), 1);
        assert_eq!(queue.pending.pending_texture_uploads[0].asset_id, 7);
    }
}
