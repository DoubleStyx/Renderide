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
    queue.cubemap_formats.insert(id, f.clone());
    let props = queue.cubemap_properties.get(&id);
    let Some(device) = queue.gpu_device.clone() else {
        send_cubemap_result(
            ipc,
            id,
            TextureUpdateResultType::FORMAT_SET,
            queue.cubemap_pool.get_texture(id).is_none(),
        );
        return;
    };
    let Some(limits) = queue.gpu_limits.as_ref() else {
        logger::warn!("cubemap {id}: gpu_limits missing; format deferred until attach");
        send_cubemap_result(
            ipc,
            id,
            TextureUpdateResultType::FORMAT_SET,
            queue.cubemap_pool.get_texture(id).is_none(),
        );
        return;
    };
    let Some(tex) = GpuCubemap::new_from_format(device.as_ref(), limits.as_ref(), &f, props) else {
        logger::warn!("cubemap {id}: SetCubemapFormat rejected (bad size or device)");
        return;
    };
    let existed_before = queue.cubemap_pool.insert_texture(tex);
    send_cubemap_result(
        ipc,
        id,
        TextureUpdateResultType::FORMAT_SET,
        !existed_before,
    );
    logger::trace!(
        "cubemap {} format {:?} size={} mips={} (resident_bytes≈{})",
        id,
        f.format,
        f.size,
        f.mipmap_count,
        queue.cubemap_pool.accounting().texture_resident_bytes()
    );
}

/// Handle [`SetCubemapProperties`](crate::shared::SetCubemapProperties).
pub fn on_set_cubemap_properties(
    queue: &mut AssetTransferQueue,
    p: SetCubemapProperties,
    ipc: Option<&mut DualQueueIpc>,
) {
    let id = p.asset_id;
    queue.cubemap_properties.insert(id, p.clone());
    if let Some(t) = queue.cubemap_pool.get_texture_mut(id) {
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
        has_format: |queue, id| queue.cubemap_formats.contains_key(&id),
        pending_len: |queue| queue.pending_cubemap_uploads.len(),
        push_pending: |queue, data| queue.pending_cubemap_uploads.push_back(data),
        has_resident: |queue, id| queue.cubemap_pool.get_texture(id).is_some(),
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
    queue.cubemap_formats.remove(&id);
    queue.cubemap_properties.remove(&id);
    if queue.cubemap_pool.remove_texture(id) {
        logger::info!(
            "cubemap {id} unloaded (tex≈{} total≈{})",
            queue.cubemap_pool.accounting().texture_resident_bytes(),
            queue.mesh_pool.accounting().total_resident_bytes()
        );
    }
}

fn enqueue_cubemap_upload_task(queue: &mut AssetTransferQueue, d: SetCubemapData) -> bool {
    let id = d.asset_id;
    let Some(fmt) = queue.cubemap_formats.get(&id).cloned() else {
        logger::warn!("cubemap {id}: missing format");
        return false;
    };
    let Some(wgpu_fmt) = queue.cubemap_pool.get_texture(id).map(|t| t.wgpu_format) else {
        logger::warn!("cubemap {id}: missing GPU texture");
        return false;
    };
    let high = d.high_priority;
    let task = AssetTask::Cubemap(CubemapUploadTask::new(d, fmt, wgpu_fmt));
    queue.integrator_mut().try_enqueue(task, high)
}
