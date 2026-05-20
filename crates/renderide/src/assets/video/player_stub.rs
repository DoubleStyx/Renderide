//! Stub [`VideoPlayer`] used when the `video-textures` feature is disabled.
//!
//! Mirrors the public surface of the real GStreamer-backed player so that
//! upload handlers compile unchanged. Every method is a no-op; `new` always returns
//! `None`, so `video_players` stays empty and the integrator's polling loop
//! has nothing to drive.

use crate::gpu::GpuQueueAccessGate;
use renderide_shared::ipc::DualQueueIpc;
use renderide_shared::{
    VideoTextureClockErrorState, VideoTextureLoad, VideoTextureStartAudioTrack, VideoTextureUpdate,
};
use std::sync::Arc;

/// Stand-in for the real GStreamer-backed player. Cannot be constructed.
pub enum VideoPlayer {}

impl VideoPlayer {
    /// Always returns `None` because video playback is not compiled in.
    /// Logs a one-time-per-asset hint at debug level so production builds
    /// stay quiet while developers can still see why their video texture
    /// shows a black placeholder.
    pub fn new(
        load: VideoTextureLoad,
        _device: Arc<wgpu::Device>,
        _queue: Arc<wgpu::Queue>,
        _queue_access_gate: GpuQueueAccessGate,
    ) -> Option<Self> {
        logger::debug!(
            "video texture {}: playback skipped (renderide built without `video-textures` feature)",
            load.asset_id
        );
        None
    }

    /// No-op stand-in for the GStreamer-backed implementation.
    pub fn handle_update(&self, _u: VideoTextureUpdate) {
        match self {}
    }

    /// No-op stand-in for the GStreamer-backed implementation.
    pub fn handle_start_audio_track(&self, _s: VideoTextureStartAudioTrack) {
        match self {}
    }

    /// No-op stand-in for the GStreamer-backed implementation.
    pub fn process_events(
        &self,
        _frame_sink: &mut impl Sized,
        _ipc: &mut Option<&mut DualQueueIpc>,
    ) {
        match self {}
    }

    /// No-op stand-in for starting cooperative shutdown.
    pub fn begin_shutdown(&self) {
        match self {}
    }

    /// No-op stand-in for polling cooperative shutdown completion.
    pub fn poll_shutdown_complete(&self) -> bool {
        match self {}
    }

    /// No-op stand-in for the GStreamer-backed implementation. The stub never produces samples.
    pub fn sample_clock_error(&self) -> Option<VideoTextureClockErrorState> {
        match self {}
    }
}
