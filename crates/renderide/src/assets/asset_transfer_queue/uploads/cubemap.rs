//! Cubemap format/properties/data IPC and cooperative [`super::super::cubemap_task::CubemapUploadTask`] integration.

use crate::gpu_pools::GpuCubemap;
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::shared::{
    RendererCommand, SetCubemapData, SetCubemapFormat, SetCubemapProperties, SetCubemapResult,
    TextureUpdateResultType, UnloadCubemap,
};

use super::super::AssetTransferQueue;
use super::super::cubemap_task::CubemapUploadTask;
use super::super::integrator::AssetTask;
use super::MAX_PENDING_CUBEMAP_UPLOADS;
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
    let _ = ipc.send_background(RendererCommand::SetCubemapResult(SetCubemapResult {
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
    queue.catalogs.cubemap_formats.insert(id, f.clone());
    let props = queue.catalogs.cubemap_properties.get(&id);
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
    let Some(tex) = GpuCubemap::new_from_format(device.as_ref(), limits.as_ref(), &f, props) else {
        logger::warn!("cubemap {id}: SetCubemapFormat rejected (bad size or device)");
        return;
    };
    let existed_before = queue.pools.cubemap_pool.insert(tex);
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
        max_pending: MAX_PENDING_CUBEMAP_UPLOADS,
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

    if !enqueue_cubemap_upload_task(queue, d) {
        logger::warn!("cubemap {asset_id}: asset integration queue full; dropping data upload",);
    }
}

/// Replay pending cubemap data after GPU attach.
pub fn try_cubemap_upload_with_device(
    queue: &mut AssetTransferQueue,
    data: SetCubemapData,
    _shm: &mut SharedMemoryAccessor,
    _ipc: Option<&mut DualQueueIpc>,
    _consume_texture_upload_budget: bool,
) {
    let _ = enqueue_cubemap_upload_task(queue, data);
}

/// Remove a cubemap asset from CPU tables and the pool.
pub fn on_unload_cubemap(queue: &mut AssetTransferQueue, u: UnloadCubemap) {
    let id = u.asset_id;
    queue.catalogs.cubemap_formats.remove(&id);
    queue.catalogs.cubemap_properties.remove(&id);
    if queue.pools.cubemap_pool.remove(id) {
        logger::info!(
            "cubemap {id} unloaded (tex~={} total~={})",
            queue
                .pools
                .cubemap_pool
                .accounting()
                .texture_resident_bytes(),
            queue.pools.mesh_pool.accounting().total_resident_bytes()
        );
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
    let high = d.high_priority;
    let task = AssetTask::Cubemap(CubemapUploadTask::new(d, fmt, wgpu_fmt));
    queue.integrator_mut().try_enqueue(task, high)
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
    fn data_without_format_is_dropped_before_pending_queue() {
        let mut queue = AssetTransferQueue::new();

        on_set_cubemap_data(&mut queue, data(11), None, None);

        assert!(queue.pending.pending_cubemap_uploads.is_empty());
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
