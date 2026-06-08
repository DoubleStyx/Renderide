//! Host render-texture format IPC -> [`crate::gpu_pools::GpuRenderTexture`] pool.

use crate::gpu_pools::GpuRenderTexture;
use crate::ipc::DualQueueIpc;
use crate::shared::{
    RenderTextureResult, RendererCommand, SetRenderTextureFormat, UnloadRenderTexture,
};

use super::super::AssetTransferQueue;
use super::super::integrator::RetiredAssetResource;

fn send_render_texture_result(
    ipc: Option<&mut DualQueueIpc>,
    asset_id: i32,
    instance_changed: bool,
) {
    let Some(ipc) = ipc else {
        return;
    };
    if !ipc.send_background_reliable(RendererCommand::RenderTextureResult(RenderTextureResult {
        asset_id,
        instance_changed,
    })) {
        logger::warn!("render texture {asset_id}: failed to enqueue reliable RenderTextureResult");
    }
}

/// Handle [`SetRenderTextureFormat`](crate::shared::SetRenderTextureFormat).
pub fn on_set_render_texture_format(
    queue: &mut AssetTransferQueue,
    f: SetRenderTextureFormat,
    ipc: Option<&mut DualQueueIpc>,
) {
    let id = f.asset_id;
    queue.catalogs.render_texture_formats.insert(id, f.clone());
    let Some(device) = queue.gpu.gpu_device.clone() else {
        send_render_texture_result(ipc, id, queue.pools.render_texture_pool.get(id).is_none());
        return;
    };
    let Some(limits) = queue.gpu.gpu_limits.as_ref() else {
        logger::warn!("render texture {id}: gpu_limits missing; format deferred until attach");
        send_render_texture_result(ipc, id, queue.pools.render_texture_pool.get(id).is_none());
        return;
    };
    let Some(tex) = GpuRenderTexture::new_from_format(device.as_ref(), limits.as_ref(), &f) else {
        logger::warn!("render texture {id}: SetRenderTextureFormat rejected (bad size or device)");
        send_render_texture_result(ipc, id, false);
        return;
    };
    queue.pools.render_texture_pool.insert(tex);
    send_render_texture_result(ipc, id, true);
    logger::trace!(
        "render texture {} {}x{} depth_bits={} (resident_bytes~={})",
        id,
        f.size.x,
        f.size.y,
        f.depth,
        queue
            .pools
            .render_texture_pool
            .accounting()
            .texture_resident_bytes()
    );
}

/// Remove a render texture asset from the CPU table and GPU pool.
pub fn on_unload_render_texture(queue: &mut AssetTransferQueue, u: UnloadRenderTexture) {
    let id = u.asset_id;
    queue.catalogs.render_texture_formats.remove(&id);
    if let Some(texture) = queue.pools.render_texture_pool.take(id) {
        queue
            .integrator_mut()
            .enqueue_delayed_removal(RetiredAssetResource::RenderTexture(texture));
    }
}
