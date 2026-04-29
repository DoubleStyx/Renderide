//! Video texture playback backend.
//!
//! Real GStreamer-backed implementation lives behind the `video-textures` Cargo feature; when
//! the feature is off, [`player::VideoPlayer`] is a stub whose `new` always returns `None`,
//! so video texture IPC commands resolve to a static black GPU placeholder.

#[cfg(feature = "video-textures")]
pub mod player;

#[cfg(not(feature = "video-textures"))]
#[path = "player_stub.rs"]
pub mod player;

#[cfg(feature = "video-textures")]
mod audio_sink;
#[cfg(feature = "video-textures")]
mod cpu_copy;

#[cfg(feature = "video-textures")]
pub(crate) use sink::WgpuGstVideoSink;

#[cfg(feature = "video-textures")]
mod sink {
    use glam::IVec2;
    use gstreamer_app::AppSink;
    use std::sync::Arc;

    /// Common trait for all video sink implementations used in [`super::player::VideoPlayer`].
    pub trait WgpuGstVideoSink: Send + Sync {
        /// Name of the video sink backend.
        fn name(&self) -> &str;

        /// Returns the underlying [`AppSink`] for passing to playbin.
        fn appsink(&self) -> &AppSink;

        /// Returns a new [`wgpu::TextureView`] if the sink allocated a new texture
        /// since the last call, along with its dimensions and resident byte count.
        /// Returns `None` if nothing changed.
        fn poll_texture_change(&mut self) -> Option<(Arc<wgpu::TextureView>, u32, u32, u64)>;

        /// Returns the current video frame size from negotiated caps,
        /// or `None` if caps are not yet available.
        fn size(&self) -> Option<IVec2>;
    }
}
