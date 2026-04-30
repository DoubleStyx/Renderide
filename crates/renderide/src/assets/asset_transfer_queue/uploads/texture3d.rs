//! Texture3D format/properties/data IPC and cooperative [`super::super::texture3d_task::Texture3dUploadTask`] integration.

use crate::gpu_pools::GpuTexture3d;
use crate::ipc::{DualQueueIpc, SharedMemoryAccessor};
use crate::shared::{
    RendererCommand, SetTexture3DData, SetTexture3DFormat, SetTexture3DProperties,
    SetTexture3DResult, TextureUpdateResultType, UnloadTexture3D,
};

use super::super::AssetTransferQueue;
use super::super::integrator::AssetTask;
use super::super::texture3d_task::Texture3dUploadTask;
use super::MAX_PENDING_TEXTURE3D_UPLOADS;
use super::allocations::flush_pending_texture3d_allocations;
use super::texture_common::{TextureUploadAdmission, admit_texture_upload_data};

fn send_texture_3d_result(
    ipc: Option<&mut DualQueueIpc>,
    asset_id: i32,
    update: i32,
    instance_changed: bool,
) {
    let Some(ipc) = ipc else {
        return;
    };
    let _ = ipc.send_background(RendererCommand::SetTexture3DResult(SetTexture3DResult {
        asset_id,
        r#type: TextureUpdateResultType(update),
        instance_changed,
    }));
}

/// Handle [`SetTexture3DFormat`](crate::shared::SetTexture3DFormat).
pub fn on_set_texture_3d_format(
    queue: &mut AssetTransferQueue,
    f: SetTexture3DFormat,
    ipc: Option<&mut DualQueueIpc>,
) {
    let id = f.asset_id;
    queue.texture3d_formats.insert(id, f.clone());
    let props = queue.texture3d_properties.get(&id);
    let Some(device) = queue.gpu_device.clone() else {
        send_texture_3d_result(
            ipc,
            id,
            TextureUpdateResultType::FORMAT_SET,
            queue.texture3d_pool.get_texture(id).is_none(),
        );
        return;
    };
    let Some(limits) = queue.gpu_limits.as_ref() else {
        logger::warn!("texture3d {id}: gpu_limits missing; format deferred until attach");
        send_texture_3d_result(
            ipc,
            id,
            TextureUpdateResultType::FORMAT_SET,
            queue.texture3d_pool.get_texture(id).is_none(),
        );
        return;
    };
    let Some(tex) = GpuTexture3d::new_from_format(device.as_ref(), limits.as_ref(), &f, props)
    else {
        logger::warn!("texture3d {id}: SetTexture3DFormat rejected (bad size or device)");
        return;
    };
    let existed_before = queue.texture3d_pool.insert_texture(tex);
    send_texture_3d_result(
        ipc,
        id,
        TextureUpdateResultType::FORMAT_SET,
        !existed_before,
    );
    logger::trace!(
        "texture3d {} format {:?} {}×{}×{} mips={} (resident_bytes≈{})",
        id,
        f.format,
        f.width,
        f.height,
        f.depth,
        f.mipmap_count,
        queue.texture3d_pool.accounting().texture_resident_bytes()
    );
}

/// Handle [`SetTexture3DProperties`](crate::shared::SetTexture3DProperties).
pub fn on_set_texture_3d_properties(
    queue: &mut AssetTransferQueue,
    p: SetTexture3DProperties,
    ipc: Option<&mut DualQueueIpc>,
) {
    let id = p.asset_id;
    queue.texture3d_properties.insert(id, p.clone());
    if let Some(t) = queue.texture3d_pool.get_texture_mut(id) {
        t.apply_properties(&p);
    }
    send_texture_3d_result(ipc, id, TextureUpdateResultType::PROPERTIES_SET, false);
}

/// Enqueue [`SetTexture3DData`] for time-sliced GPU integration.
pub fn on_set_texture_3d_data(
    queue: &mut AssetTransferQueue,
    d: SetTexture3DData,
    _shm: Option<&mut SharedMemoryAccessor>,
    _ipc: Option<&mut DualQueueIpc>,
) {
    let Some(d) = admit_texture_upload_data(TextureUploadAdmission {
        asset_id: d.asset_id,
        payload_len: d.data.length,
        data: d,
        kind: "texture3d",
        format_command: "SetTexture3DData",
        max_pending: MAX_PENDING_TEXTURE3D_UPLOADS,
        queue,
        has_format: |queue, id| queue.texture3d_formats.contains_key(&id),
        pending_len: |queue| queue.pending_texture3d_uploads.len(),
        push_pending: |queue, data| queue.pending_texture3d_uploads.push_back(data),
        has_resident: |queue, id| queue.texture3d_pool.get_texture(id).is_some(),
        flush_allocations: flush_pending_texture3d_allocations,
    }) else {
        return;
    };
    let asset_id = d.asset_id;
    logger::trace!(
        "texture3d_upload enqueue asset_id={} payload_bytes={} high_priority={}",
        asset_id,
        d.data.length.max(0),
        d.high_priority,
    );

    if !enqueue_texture3d_upload_task(queue, d) {
        logger::warn!("texture3d {asset_id}: asset integration queue full; dropping data upload",);
    }
}

/// Replay pending Texture3D data after GPU attach.
pub fn try_texture3d_upload_with_device(
    queue: &mut AssetTransferQueue,
    data: SetTexture3DData,
    _shm: &mut SharedMemoryAccessor,
    _ipc: Option<&mut DualQueueIpc>,
    _consume_texture_upload_budget: bool,
) {
    let _ = enqueue_texture3d_upload_task(queue, data);
}

/// Remove a Texture3D asset from CPU tables and the pool.
pub fn on_unload_texture_3d(queue: &mut AssetTransferQueue, u: UnloadTexture3D) {
    let id = u.asset_id;
    queue.texture3d_formats.remove(&id);
    queue.texture3d_properties.remove(&id);
    if queue.texture3d_pool.remove_texture(id) {
        logger::info!(
            "texture3d {id} unloaded (tex≈{} total≈{})",
            queue.texture3d_pool.accounting().texture_resident_bytes(),
            queue.mesh_pool.accounting().total_resident_bytes()
        );
    }
}

fn enqueue_texture3d_upload_task(queue: &mut AssetTransferQueue, d: SetTexture3DData) -> bool {
    let id = d.asset_id;
    let Some(fmt) = queue.texture3d_formats.get(&id).cloned() else {
        logger::warn!("texture3d {id}: missing format");
        return false;
    };
    let Some(wgpu_fmt) = queue.texture3d_pool.get_texture(id).map(|t| t.wgpu_format) else {
        logger::warn!("texture3d {id}: missing GPU texture");
        return false;
    };
    let high = d.high_priority;
    let task = AssetTask::Texture3d(Texture3dUploadTask::new(d, fmt, wgpu_fmt));
    queue.integrator_mut().try_enqueue(task, high)
}
