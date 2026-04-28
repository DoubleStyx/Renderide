use super::super::AssetTransferQueue;
use crate::assets::video::player::VideoPlayer;
use crate::resources::GpuVideoTexture;
use renderide_shared::{
    UnloadVideoTexture, VideoTextureLoad, VideoTextureProperties, VideoTextureStartAudioTrack,
    VideoTextureUpdate,
};

/// Handle [`VideoTextureLoad`].
pub fn on_video_texture_load(queue: &mut AssetTransferQueue, v: VideoTextureLoad) {
    let id = v.asset_id;
    if let Some(tex) = VideoPlayer::new(
        v,
        queue.gpu_device.clone().unwrap(),
        queue.gpu_queue.clone().unwrap(),
    ) {
        queue.video_players.insert(id, tex);
    }
}

/// Handle [`VideoTextureUpdate`].
pub fn on_video_texture_update(queue: &mut AssetTransferQueue, v: VideoTextureUpdate) {
    let id = v.asset_id;
    if let Some(t) = queue.video_players.get_mut(&id) {
        t.handle_update(v);
    }
}

/// Handle [`VideoTextureProperties`].
pub fn on_video_texture_properties(queue: &mut AssetTransferQueue, p: VideoTextureProperties) {
    let id = p.asset_id;

    let Some(device) = queue.gpu_device.clone() else {
        return;
    };

    if let Some(tex) = queue.video_texture_pool.get_mut(id) {
        tex.set_props(&p);
        return;
    }

    // create new dummy texture
    let tex = GpuVideoTexture::new(device.as_ref(), id, &p);

    queue.video_texture_pool.insert_texture(tex);
    queue.maybe_warn_texture_vram_budget();

    logger::trace!(
        "video texture {} (resident_bytes≈{})",
        id,
        queue
            .video_texture_pool
            .accounting()
            .texture_resident_bytes()
    );
}

/// Handle [`VideoTextureStartAudioTrack`].
pub fn on_video_texture_start_audio_track(
    queue: &mut AssetTransferQueue,
    s: VideoTextureStartAudioTrack,
) {
    let id = s.asset_id;
    if let Some(tex) = queue.video_players.get_mut(&id) {
        tex.handle_start_audio_track(s);
    }
}

/// /// Handle [`UnloadVideoTexture`].
pub fn on_unload_video_texture(queue: &mut AssetTransferQueue, u: UnloadVideoTexture) {
    let id = u.asset_id;
    queue.video_players.remove(&id);
    if queue.video_texture_pool.remove(id) {
        logger::info!(
            "video texture {id} unloaded (total≈{})",
            queue
                .video_texture_pool
                .accounting()
                .texture_resident_bytes(),
        );
    }
}
