//! GStreamer-based video backend implementation

use glam::IVec2;
use gstreamer_app::AppSink;
use std::sync::Arc;

pub mod player;

mod cpu_copy;

/// Common trait for all video sink implementations used in [`player::VideoPlayer`].
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
