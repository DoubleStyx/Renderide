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

fn send_desktop_texture_update(
    ipc: Option<&mut DualQueueIpc>,
    update: DesktopTexturePropertiesUpdate,
) {
    let Some(ipc) = ipc else {
        return;
    };
    let _ = ipc.send_background_reliable(RendererCommand::DesktopTexturePropertiesUpdate(update));
}

fn send_gaussian_splat_result(
    ipc: Option<&mut DualQueueIpc>,
    asset_id: i32,
    instance_changed: bool,
) {
    let Some(ipc) = ipc else {
        return;
    };
    let _ =
        ipc.send_background_reliable(RendererCommand::GaussianSplatResult(GaussianSplatResult {
            asset_id,
            instance_changed,
        }));
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

/// Stores and acknowledges a point render buffer upload.
pub fn on_point_render_buffer_upload(
    queue: &mut AssetTransferQueue,
    upload: PointRenderBufferUpload,
    _ipc: Option<&mut DualQueueIpc>,
) {
    let asset_id = upload.asset_id;
    let count = upload.count;
    queue.integrator_mut().enqueue_lane(
        AssetTask::PointRenderBuffer(upload),
        AssetTaskLane::Particle,
    );
    logger::trace!("point render buffer {asset_id}: queued placeholder upload count={count}");
}

/// Removes a tracked point render buffer upload.
pub fn on_point_render_buffer_unload(
    queue: &mut AssetTransferQueue,
    unload: PointRenderBufferUnload,
) {
    let asset_id = unload.asset_id;
    queue.catalogs.point_render_buffer_uploads.remove(&asset_id);
    logger::debug!("point render buffer {asset_id}: unloaded placeholder upload");
}

/// Stores and acknowledges a trail render buffer upload.
pub fn on_trail_render_buffer_upload(
    queue: &mut AssetTransferQueue,
    upload: TrailRenderBufferUpload,
    _ipc: Option<&mut DualQueueIpc>,
) {
    let asset_id = upload.asset_id;
    let trails_count = upload.trails_count;
    let trail_point_count = upload.trail_point_count;
    queue.integrator_mut().enqueue_lane(
        AssetTask::TrailRenderBuffer(upload),
        AssetTaskLane::Particle,
    );
    logger::trace!(
        "trail render buffer {asset_id}: queued placeholder upload trails={trails_count} points={trail_point_count}"
    );
}

/// Removes a tracked trail render buffer upload.
pub fn on_trail_render_buffer_unload(
    queue: &mut AssetTransferQueue,
    unload: TrailRenderBufferUnload,
) {
    let asset_id = unload.asset_id;
    queue.catalogs.trail_render_buffer_uploads.remove(&asset_id);
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
