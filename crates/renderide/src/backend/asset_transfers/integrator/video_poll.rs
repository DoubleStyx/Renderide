//! Video-texture event polling performed at the end of each asset-integration drain.

use crate::gpu::GpuQueueAccessMode;
use crate::ipc::DualQueueIpc;

use super::super::AssetTransferQueue;

/// Polls video texture players after upload integration.
///
/// Samples each player's clock error against the host's last-applied playback request and records
/// the latest result so the runtime can flush it into the next
/// [`crate::shared::FrameStartData`].
pub(super) fn poll_video_texture_events(
    asset: &mut AssetTransferQueue,
    ipc: &mut Option<&mut DualQueueIpc>,
    queue_access_mode: GpuQueueAccessMode,
) {
    // Idle scenes never instantiate a video player; skip the `mem::take` swap and the inner
    // profiling scope entirely so the tracy timeline doesn't carry a zero-work zone every frame.
    if asset.video.video_players.is_empty() {
        return;
    }
    profiling::scope!("asset::video_texture_poll_events");
    let mut video_textures = std::mem::take(&mut asset.video.video_players);
    {
        profiling::scope!("video::sample_clock_errors");
        for player in video_textures.values_mut() {
            player.process_events(asset, ipc, queue_access_mode);
            if let Some(state) = player.sample_clock_error() {
                asset.video.record_pending_clock_error(state);
            }
        }
    }
    asset.video.video_players = video_textures;
}
