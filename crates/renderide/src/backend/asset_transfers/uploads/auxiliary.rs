//! Correctness-level ingestion for asset families without full GPU rendering paths yet.

use crate::ipc::DualQueueIpc;
use crate::shared::{
    DesktopTexturePropertiesUpdate, GaussianSplatConfig, GaussianSplatResult,
    GaussianSplatUploadEncoded, GaussianSplatUploadRaw, PointRenderBufferUnload,
    PointRenderBufferUpload, RendererCommand, SetDesktopTextureProperties, TrailRenderBufferUnload,
    TrailRenderBufferUpload, UnloadDesktopTexture, UnloadGaussianSplat,
};

use super::super::AssetTransferQueue;
use super::super::catalogs::GaussianSplatUploadKind;
use super::super::integrator::{AssetTask, AssetTaskLane};
use super::super::limits::admit_descriptor_payload_len;
use super::super::particle_task::{
    PointRenderBufferTask, TrailRenderBufferTask, send_point_render_buffer_consumed,
    send_trail_render_buffer_consumed,
};

const MAX_POINT_RENDER_BUFFER_POINTS: i32 = 1_000_000;
const MAX_TRAIL_RENDER_BUFFER_TRAILS: i32 = 262_144;
const MAX_TRAIL_RENDER_BUFFER_POINTS: i32 = 1_000_000;

fn send_desktop_texture_update(
    ipc: Option<&mut DualQueueIpc>,
    update: DesktopTexturePropertiesUpdate,
) {
    let Some(ipc) = ipc else {
        return;
    };
    let asset_id = update.asset_id;
    if !ipc.send_background_reliable(RendererCommand::DesktopTexturePropertiesUpdate(update)) {
        logger::warn!(
            "desktop texture {asset_id}: failed to enqueue reliable DesktopTexturePropertiesUpdate"
        );
    }
}

fn send_gaussian_splat_result(
    ipc: Option<&mut DualQueueIpc>,
    asset_id: i32,
    instance_changed: bool,
) {
    let Some(ipc) = ipc else {
        return;
    };
    if !ipc.send_background_reliable(RendererCommand::GaussianSplatResult(GaussianSplatResult {
        asset_id,
        instance_changed,
    })) {
        logger::warn!("gaussian splat {asset_id}: failed to enqueue reliable GaussianSplatResult");
    }
}

/// Stores desktop texture properties and reports the currently known placeholder size.
pub fn on_set_desktop_texture_properties(
    queue: &mut AssetTransferQueue,
    properties: SetDesktopTextureProperties,
    ipc: Option<&mut DualQueueIpc>,
) {
    let asset_id = properties.asset_id;
    let display_index = properties.display_index;
    queue
        .catalogs
        .desktop_texture_properties
        .insert(asset_id, properties);
    let update = queue
        .catalogs
        .desktop_texture_updates
        .get(&asset_id)
        .cloned()
        .unwrap_or_else(|| DesktopTexturePropertiesUpdate {
            asset_id,
            ..Default::default()
        });
    send_desktop_texture_update(ipc, update);
    logger::debug!(
        "desktop texture {asset_id}: display_index={display_index} tracked with placeholder source"
    );
}

/// Tracks a desktop texture properties update.
pub fn on_desktop_texture_properties_update(
    queue: &mut AssetTransferQueue,
    update: DesktopTexturePropertiesUpdate,
) {
    logger::debug!(
        "desktop texture {}: properties update size={:?}",
        update.asset_id,
        update.size
    );
    queue
        .catalogs
        .desktop_texture_updates
        .insert(update.asset_id, update);
}

/// Removes a desktop texture placeholder record.
pub fn on_unload_desktop_texture(queue: &mut AssetTransferQueue, unload: UnloadDesktopTexture) {
    let asset_id = unload.asset_id;
    queue.catalogs.desktop_texture_properties.remove(&asset_id);
    queue.catalogs.desktop_texture_updates.remove(&asset_id);
    logger::debug!("desktop texture {asset_id}: unloaded placeholder source");
}

/// Queues a point render-buffer upload for PhotonDust mesh generation.
pub fn on_point_render_buffer_upload(
    queue: &mut AssetTransferQueue,
    upload: PointRenderBufferUpload,
    ipc: Option<&mut DualQueueIpc>,
) {
    let asset_id = upload.asset_id;
    let count = upload.count;
    let mut ipc = ipc;
    if !(0..=MAX_POINT_RENDER_BUFFER_POINTS).contains(&count) {
        logger::warn!(
            "point render buffer {asset_id}: rejected count={} cap={}",
            count,
            MAX_POINT_RENDER_BUFFER_POINTS
        );
        send_point_render_buffer_consumed(&mut ipc, asset_id);
        return;
    }
    if !admit_descriptor_payload_len("point render buffer", asset_id, upload.buffer.length) {
        send_point_render_buffer_consumed(&mut ipc, asset_id);
        return;
    }
    let coalesced = queue.retain_latest_point_render_buffer_upload(upload);
    if coalesced.replaced_pending_upload {
        send_point_render_buffer_consumed(&mut ipc, asset_id);
        #[cfg(feature = "tracy")]
        tracy_client::plot!("particle::point_pending_replacements", 1.0);
        logger::trace!(
            "point render buffer {asset_id}: coalesced superseded pending upload generation={}",
            coalesced.generation
        );
    } else if !queue.point_render_buffer_build_is_active(asset_id) {
        let enqueued = queue.integrator_mut().enqueue_lane(
            AssetTask::PointRenderBuffer(PointRenderBufferTask::new(asset_id)),
            AssetTaskLane::Particle,
        );
        if !enqueued {
            logger::warn!(
                "point render buffer {asset_id}: rejected upload because asset integrator is full"
            );
            if queue.cancel_point_render_buffer_generation(asset_id) {
                send_point_render_buffer_consumed(&mut ipc, asset_id);
            }
            return;
        }
    }
    logger::trace!(
        "point render buffer {asset_id}: retained upload count={count} generation={}",
        coalesced.generation
    );
}

/// Removes a resident point render-buffer upload and generated meshes.
pub fn on_point_render_buffer_unload(
    queue: &mut AssetTransferQueue,
    unload: PointRenderBufferUnload,
    ipc: Option<&mut DualQueueIpc>,
) {
    let asset_id = unload.asset_id;
    let mut ipc = ipc;
    let removed_pending_upload = queue.cancel_point_render_buffer_generation(asset_id);
    if removed_pending_upload {
        send_point_render_buffer_consumed(&mut ipc, asset_id);
    }
    queue.catalogs.point_render_buffers.remove(&asset_id);
    for mesh_id in crate::particles::point_render_buffer_generated_mesh_ids(asset_id) {
        queue.retire_mesh_asset(mesh_id);
    }
    logger::debug!("point render buffer {asset_id}: unloaded resident upload");
}

/// Queues a trail render-buffer upload for PhotonDust ribbon mesh generation.
pub fn on_trail_render_buffer_upload(
    queue: &mut AssetTransferQueue,
    upload: TrailRenderBufferUpload,
    ipc: Option<&mut DualQueueIpc>,
) {
    let asset_id = upload.asset_id;
    let trails_count = upload.trails_count;
    let trail_point_count = upload.trail_point_count;
    let mut ipc = ipc;
    if trails_count < 0
        || trail_point_count < 0
        || trails_count > MAX_TRAIL_RENDER_BUFFER_TRAILS
        || trail_point_count > MAX_TRAIL_RENDER_BUFFER_POINTS
    {
        logger::warn!(
            "trail render buffer {asset_id}: rejected trails={} points={} caps={}/{}",
            trails_count,
            trail_point_count,
            MAX_TRAIL_RENDER_BUFFER_TRAILS,
            MAX_TRAIL_RENDER_BUFFER_POINTS
        );
        send_trail_render_buffer_consumed(&mut ipc, asset_id);
        return;
    }
    if !admit_descriptor_payload_len("trail render buffer", asset_id, upload.buffer.length) {
        send_trail_render_buffer_consumed(&mut ipc, asset_id);
        return;
    }
    let coalesced = queue.retain_latest_trail_render_buffer_upload(upload);
    if coalesced.replaced_pending_upload {
        send_trail_render_buffer_consumed(&mut ipc, asset_id);
        #[cfg(feature = "tracy")]
        tracy_client::plot!("particle::trail_pending_replacements", 1.0);
        logger::trace!(
            "trail render buffer {asset_id}: coalesced superseded pending upload generation={}",
            coalesced.generation
        );
    } else if !queue.trail_render_buffer_build_is_active(asset_id) {
        let enqueued = queue.integrator_mut().enqueue_lane(
            AssetTask::TrailRenderBuffer(TrailRenderBufferTask::new(asset_id)),
            AssetTaskLane::Particle,
        );
        if !enqueued {
            logger::warn!(
                "trail render buffer {asset_id}: rejected upload because asset integrator is full"
            );
            if queue.cancel_trail_render_buffer_generation(asset_id) {
                send_trail_render_buffer_consumed(&mut ipc, asset_id);
            }
            return;
        }
    }
    logger::trace!(
        "trail render buffer {asset_id}: retained upload trails={trails_count} points={trail_point_count} generation={}",
        coalesced.generation
    );
}

/// Removes a resident trail render-buffer upload and generated meshes.
pub fn on_trail_render_buffer_unload(
    queue: &mut AssetTransferQueue,
    unload: TrailRenderBufferUnload,
    ipc: Option<&mut DualQueueIpc>,
) {
    let asset_id = unload.asset_id;
    let mut ipc = ipc;
    let removed_pending_upload = queue.cancel_trail_render_buffer_generation(asset_id);
    if removed_pending_upload {
        send_trail_render_buffer_consumed(&mut ipc, asset_id);
    }
    queue.catalogs.trail_render_buffers.remove(&asset_id);
    for mesh_id in crate::particles::trail_render_buffer_generated_mesh_ids(asset_id) {
        queue.retire_mesh_asset(mesh_id);
    }
    logger::debug!("trail render buffer {asset_id}: unloaded resident upload");
}

/// Stores the Gaussian splat renderer config.
pub fn on_gaussian_splat_config(queue: &mut AssetTransferQueue, config: GaussianSplatConfig) {
    logger::debug!(
        "gaussian splat config: sorting_mega_operations_per_camera={}",
        config.sorting_mega_operations_per_camera
    );
    queue.catalogs.gaussian_splat_config = config;
}

/// Stores and acknowledges a raw Gaussian splat upload.
pub fn on_gaussian_splat_upload_raw(
    queue: &mut AssetTransferQueue,
    upload: GaussianSplatUploadRaw,
    ipc: Option<&mut DualQueueIpc>,
) {
    let asset_id = upload.asset_id;
    let splat_count = upload.splat_count;
    let instance_changed = queue
        .catalogs
        .gaussian_splat_uploads
        .insert(asset_id, GaussianSplatUploadKind::Raw)
        .is_none();
    send_gaussian_splat_result(ipc, asset_id, instance_changed);
    logger::debug!(
        "gaussian splat {asset_id}: consumed raw placeholder upload splats={splat_count}"
    );
}

/// Stores and acknowledges an encoded Gaussian splat upload.
pub fn on_gaussian_splat_upload_encoded(
    queue: &mut AssetTransferQueue,
    upload: GaussianSplatUploadEncoded,
    ipc: Option<&mut DualQueueIpc>,
) {
    let asset_id = upload.asset_id;
    let splat_count = upload.splat_count;
    let chunk_count = upload.chunk_count;
    let instance_changed = queue
        .catalogs
        .gaussian_splat_uploads
        .insert(asset_id, GaussianSplatUploadKind::Encoded)
        .is_none();
    send_gaussian_splat_result(ipc, asset_id, instance_changed);
    logger::debug!(
        "gaussian splat {asset_id}: consumed encoded placeholder upload splats={splat_count} chunks={chunk_count}"
    );
}

/// Removes a tracked Gaussian splat upload.
pub fn on_unload_gaussian_splat(queue: &mut AssetTransferQueue, unload: UnloadGaussianSplat) {
    let asset_id = unload.asset_id;
    queue.catalogs.gaussian_splat_uploads.remove(&asset_id);
    logger::debug!("gaussian splat {asset_id}: unloaded placeholder upload");
}
