//! Host render-texture format IPC → [`crate::gpu_pools::GpuRenderTexture`] pool.

use crate::gpu_pools::GpuRenderTexture;
use crate::ipc::DualQueueIpc;
use crate::shared::{
    RenderTextureResult, RendererCommand, SetRenderTextureFormat, UnloadRenderTexture,
};

use super::super::AssetTransferQueue;

fn send_render_texture_result(
    ipc: Option<&mut DualQueueIpc>,
    asset_id: i32,
    instance_changed: bool,
) {
    let Some(ipc) = ipc else {
        return;
    };
    let _ = ipc.send_background(RendererCommand::RenderTextureResult(RenderTextureResult {
        asset_id,
        instance_changed,
    }));
}

/// Handle [`SetRenderTextureFormat`](crate::shared::SetRenderTextureFormat).
pub fn on_set_render_texture_format(
    queue: &mut AssetTransferQueue,
    f: SetRenderTextureFormat,
    ipc: Option<&mut DualQueueIpc>,
) {
    let id = f.asset_id;
    queue.render_texture_formats.insert(id, f.clone());
    let Some(device) = queue.gpu_device.clone() else {
        send_render_texture_result(ipc, id, queue.render_texture_pool.get(id).is_none());
        return;
    };
    let Some(limits) = queue.gpu_limits.as_ref() else {
        logger::warn!("render texture {id}: gpu_limits missing; format deferred until attach");
        send_render_texture_result(ipc, id, queue.render_texture_pool.get(id).is_none());
        return;
    };
    let Some(tex) = GpuRenderTexture::new_from_format(
        device.as_ref(),
        limits.as_ref(),
        &f,
        queue.render_texture_hdr_color,
    ) else {
        logger::warn!("render texture {id}: SetRenderTextureFormat rejected (bad size or device)");
        return;
    };
    let existed_before = queue.render_texture_pool.insert_texture(tex);
    queue.maybe_warn_texture_vram_budget();
    send_render_texture_result(ipc, id, !existed_before);
    logger::trace!(
        "render texture {} {}×{} depth_bits={} (resident_bytes≈{})",
        id,
        f.size.x,
        f.size.y,
        f.depth,
        queue
            .render_texture_pool
            .accounting()
            .texture_resident_bytes()
    );
}

/// Remove a render texture asset from the CPU table and GPU pool.
pub fn on_unload_render_texture(queue: &mut AssetTransferQueue, u: UnloadRenderTexture) {
    let id = u.asset_id;
    queue.render_texture_formats.remove(&id);
    if queue.render_texture_pool.remove(id) {
        logger::info!(
            "render texture {id} unloaded (tex≈{} total≈{})",
            queue.texture_pool.accounting().texture_resident_bytes(),
            queue.mesh_pool.accounting().total_resident_bytes()
        );
    }
}
