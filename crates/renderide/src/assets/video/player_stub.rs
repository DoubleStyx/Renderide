//! Stub [`VideoPlayer`] used when the `video-textures` feature is disabled.
//!
//! Mirrors the public surface of the real GStreamer-backed player so that
//! [`crate::assets::asset_transfer_queue::AssetTransferQueue`] and the upload
//! handlers compile unchanged. Every method is a no-op; `new` always returns
//! `None`, so `video_players` stays empty and the integrator's polling loop
//! has nothing to drive.

use crate::assets::AssetTransferQueue;
use renderide_shared::ipc::DualQueueIpc;
use renderide_shared::{
    VideoTextureClockErrorState, VideoTextureLoad, VideoTextureStartAudioTrack, VideoTextureUpdate,
};
use std::sync::Arc;

/// Stand-in for the real GStreamer-backed player. Cannot be constructed.
pub struct VideoPlayer {
    /// Uninhabited marker; prevents instantiation outside `Self::new`.
    _never: std::convert::Infallible,
}

impl VideoPlayer {
    /// Always returns `None` because video playback is not compiled in.
    /// Logs a one-time-per-asset hint at debug level so production builds
    /// stay quiet while developers can still see why their video texture
    /// shows a black placeholder.
    pub fn new(
        load: VideoTextureLoad,
        _device: Arc<wgpu::Device>,
        _queue: Arc<wgpu::Queue>,
    ) -> Option<Self> {
        logger::debug!(
            "video texture {}: playback skipped (renderide built without `video-textures` feature)",
            load.asset_id
        );
        None
    }

    /// No-op stand-in for the GStreamer-backed implementation.
    pub fn handle_update(&mut self, _u: VideoTextureUpdate) {
        match self._never {}
    }

    /// No-op stand-in for the GStreamer-backed implementation.
    pub fn handle_start_audio_track(&mut self, _s: VideoTextureStartAudioTrack) {
        match self._never {}
    }

    /// No-op stand-in for the GStreamer-backed implementation.
    pub fn process_events(
        &mut self,
        _queue: &mut AssetTransferQueue,
        _ipc: &mut Option<&mut DualQueueIpc>,
    ) {
        match self._never {}
    }

    /// No-op stand-in for the GStreamer-backed implementation. The stub never produces samples.
    pub fn sample_clock_error(&self) -> Option<VideoTextureClockErrorState> {
        match self._never {}
    }
}
