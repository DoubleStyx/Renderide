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
    queue.texture_formats.insert(id, f.clone());
    let props = queue.texture_properties.get(&id);
    let Some(device) = queue.gpu_device.clone() else {
        send_texture_2d_result(
            ipc,
            id,
            TextureUpdateResultType::FORMAT_SET,
            queue.texture_pool.get_texture(id).is_none(),
        );
        return;
    };
    let Some(limits) = queue.gpu_limits.as_ref() else {
        logger::warn!("texture {id}: gpu_limits missing; format deferred until attach");
        send_texture_2d_result(
            ipc,
            id,
            TextureUpdateResultType::FORMAT_SET,
            queue.texture_pool.get_texture(id).is_none(),
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
    let existed_before = queue.texture_pool.insert_texture(tex);
    queue.maybe_warn_texture_vram_budget();
    send_texture_2d_result(
        ipc,
        id,
        TextureUpdateResultType::FORMAT_SET,
        !existed_before,
    );
    logger::trace!(
        "texture {} format {:?} {}×{} mips={} (resident_bytes≈{})",
        id,
        f.format,
        f.width,
        f.height,
        f.mipmap_count,
        queue.texture_pool.accounting().texture_resident_bytes()
    );
}

/// Handle [`SetTexture2DProperties`](crate::shared::SetTexture2DProperties).
pub fn on_set_texture_2d_properties(
    queue: &mut AssetTransferQueue,
    p: SetTexture2DProperties,
    ipc: Option<&mut DualQueueIpc>,
) {
    let id = p.asset_id;
    queue.texture_properties.insert(id, p.clone());
    if let Some(t) = queue.texture_pool.get_texture_mut(id) {
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
        has_format: |queue, id| queue.texture_formats.contains_key(&id),
        pending_len: |queue| queue.pending_texture_uploads.len(),
        push_pending: |queue, data| queue.pending_texture_uploads.push_back(data),
        has_resident: |queue, id| queue.texture_pool.get_texture(id).is_some(),
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
    queue.texture_formats.remove(&id);
    queue.texture_properties.remove(&id);
    if queue.texture_pool.remove_texture(id) {
        logger::info!(
            "texture {id} unloaded (mesh≈{} tex≈{} total≈{})",
            queue.mesh_pool.accounting().mesh_resident_bytes(),
            queue.texture_pool.accounting().texture_resident_bytes(),
            queue.mesh_pool.accounting().total_resident_bytes()
        );
    }
}

fn enqueue_texture_upload_task(queue: &mut AssetTransferQueue, d: SetTexture2DData) -> bool {
    let id = d.asset_id;
    let Some(fmt) = queue.texture_formats.get(&id).cloned() else {
        logger::warn!("texture {id}: missing format");
        return false;
    };
    let Some(wgpu_fmt) = queue.texture_pool.get_texture(id).map(|t| t.wgpu_format) else {
        logger::warn!("texture {id}: missing GPU texture");
        return false;
    };
    let high = d.high_priority;
    let task = AssetTask::Texture(TextureUploadTask::new(d, fmt, wgpu_fmt));
    queue.integrator_mut().try_enqueue(task, high)
}
