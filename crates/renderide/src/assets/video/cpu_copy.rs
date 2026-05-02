//! CPU-copy GStreamer sink that uploads decoded RGBA frames into wgpu textures.
//!
//! The pipeline inserts a `videoflip method=vertical-flip` element upstream of
//! this sink (see [`crate::assets::video::player::VideoPlayer::new`]) so frames
//! arrive in the Unity V=0-bottom convention shared by all sampled textures.
//! This sink therefore writes the mapped buffer directly with no orientation
//! transform.

use crate::assets::video::WgpuGstVideoSink;
use glam::IVec2;
use gstreamer::prelude::ElementExt;
use gstreamer_app::{AppSink, AppSinkCallbacks};
use std::sync::{Arc, Mutex};

/// Bytes per RGBA8 pixel.
const RGBA8_BYTES_PER_PIXEL_U32: u32 = 4;

/// Bytes per RGBA8 pixel as a `u64`.
const RGBA8_BYTES_PER_PIXEL_U64: u64 = 4;

/// Internal state shared between [`CpuCopyVideoSink`] and the appsink callback.
struct SinkState {
    /// Device used to allocate replacement textures when the decoded size changes.
    device: Arc<wgpu::Device>,
    /// Queue used for CPU-to-GPU frame uploads.
    queue: Arc<wgpu::Queue>,
    /// The texture currently being written into by the callback.
    write_texture: Option<Arc<wgpu::Texture>>,
    /// Width of [`Self::write_texture`].
    width: u32,
    /// Height of [`Self::write_texture`].
    height: u32,
    /// Set to `Some` when a new texture is created, consumed by [`CpuCopyVideoSink::poll_texture_change`].
    pending_view: Option<Arc<wgpu::TextureView>>,
}

impl SinkState {
    /// Reallocates the write texture when the video size changes.
    /// Returns `true` if a new texture was created.
    fn resize_if_needed(&mut self, asset_id: i32, width: u32, height: u32) -> bool {
        profiling::scope!("video::cpu_copy_resize_if_needed");
        if self.width == width && self.height == height && self.write_texture.is_some() {
            return false;
        }

        let max_dimension = self.device.limits().max_texture_dimension_2d;
        if width > max_dimension || height > max_dimension {
            logger::warn!(
                "CpuCopyVideoSink {asset_id}: dimensions {width}x{height} exceed device limit {max_dimension}"
            );
            self.write_texture = None;
            self.pending_view = None;
            self.width = 0;
            self.height = 0;
            return false;
        }

        let texture = Arc::new(self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("VideoTexture {asset_id}")),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        }));

        let view = Arc::new(texture.create_view(&wgpu::TextureViewDescriptor::default()));

        self.write_texture = Some(texture);
        self.width = width;
        self.height = height;
        self.pending_view = Some(view);

        true
    }
}

/// Owns the video [`AppSink`], the wgpu texture it writes into.
pub struct CpuCopyVideoSink {
    /// Host video texture asset id.
    asset_id: i32,
    /// GStreamer sink receiving decoded RGBA samples.
    sink: AppSink,
    /// Shared GPU upload state.
    state: Arc<Mutex<SinkState>>,
}

impl CpuCopyVideoSink {
    /// Creates a CPU-copy sink backed by the supplied wgpu device and queue.
    pub fn new(asset_id: i32, device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let sink = build_rgba_appsink();
        let state = Arc::new(Mutex::new(SinkState {
            device,
            queue,
            write_texture: None,
            width: 0,
            height: 0,
            pending_view: None,
        }));
        install_sample_callback(&sink, asset_id, Arc::clone(&state));
        Self {
            asset_id,
            sink,
            state,
        }
    }
}

/// Builds the RGBA appsink requested from GStreamer.
fn build_rgba_appsink() -> AppSink {
    AppSink::builder()
        .caps(
            &gstreamer::Caps::builder("video/x-raw")
                .field("format", "RGBA")
                .build(),
        )
        .max_buffers(1)
        .drop(true)
        .build()
}

/// Installs the decoded-frame callback on the appsink.
fn install_sample_callback(sink: &AppSink, asset_id: i32, state: Arc<Mutex<SinkState>>) {
    sink.set_callbacks(
        AppSinkCallbacks::builder()
            .new_sample(move |appsink| handle_sample(asset_id, &state, appsink))
            .build(),
    );
}

/// Handles one decoded sample from GStreamer.
fn handle_sample(
    asset_id: i32,
    state: &Arc<Mutex<SinkState>>,
    appsink: &AppSink,
) -> Result<gstreamer::FlowSuccess, gstreamer::FlowError> {
    profiling::scope!("video::cpu_copy_sample");
    let sample = {
        profiling::scope!("video::pull_sample");
        match appsink.pull_sample() {
            Ok(sample) => sample,
            Err(e) => {
                logger::warn!("CpuCopyVideoSink {asset_id}: failed to pull sample: {e}");
                return Err(gstreamer::FlowError::Eos);
            }
        }
    };
    let Some((width, height)) = sample_dimensions(asset_id, &sample) else {
        return Ok(gstreamer::FlowSuccess::Ok);
    };
    let Some(buffer) = sample.buffer() else {
        logger::warn!("CpuCopyVideoSink {asset_id}: sample without buffer");
        return Ok(gstreamer::FlowSuccess::Ok);
    };
    let map = {
        profiling::scope!("video::map_sample_buffer");
        match buffer.map_readable() {
            Ok(map) => map,
            Err(e) => {
                logger::warn!("CpuCopyVideoSink {asset_id}: failed to map buffer: {e}");
                return Ok(gstreamer::FlowSuccess::Ok);
            }
        }
    };
    upload_mapped_rgba_frame(asset_id, state, width, height, map.as_slice());
    Ok(gstreamer::FlowSuccess::Ok)
}

/// Extracts the video dimensions declared by a sample.
fn sample_dimensions(asset_id: i32, sample: &gstreamer::Sample) -> Option<(u32, u32)> {
    let Some(caps) = sample.caps() else {
        logger::warn!("CpuCopyVideoSink {asset_id}: sample without caps");
        return None;
    };
    let Some(structure) = caps.structure(0) else {
        logger::warn!("CpuCopyVideoSink {asset_id}: caps without structure");
        return None;
    };
    dimensions_from_structure(asset_id, structure)
}

/// Extracts positive width and height values from a GStreamer caps structure.
fn dimensions_from_structure(
    asset_id: i32,
    structure: &gstreamer::StructureRef,
) -> Option<(u32, u32)> {
    match (
        structure.get::<i32>("width"),
        structure.get::<i32>("height"),
    ) {
        (Ok(width), Ok(height)) if width > 0 && height > 0 => Some((width as u32, height as u32)),
        _ => {
            logger::warn!(
                "CpuCopyVideoSink {asset_id}: invalid dimensions in caps: {:?}",
                structure
            );
            None
        }
    }
}

/// Uploads mapped RGBA bytes into the current or newly-sized write texture.
fn upload_mapped_rgba_frame(
    asset_id: i32,
    state: &Arc<Mutex<SinkState>>,
    width: u32,
    height: u32,
    bytes: &[u8],
) {
    profiling::scope!("video::upload_mapped_rgba_frame");
    let Ok(mut state) = state.lock() else {
        return;
    };
    state.resize_if_needed(asset_id, width, height);
    let Some(texture) = state.write_texture.as_ref() else {
        logger::warn!("CpuCopyVideoSink {asset_id}: no texture available after resize");
        return;
    };
    let Some(bytes_per_row) = validated_frame_layout(asset_id, width, height, bytes.len()) else {
        return;
    };
    write_rgba_frame_to_texture(&state.queue, texture, bytes, width, height, bytes_per_row);
}

/// Validates the mapped frame size and returns `bytes_per_row`.
fn validated_frame_layout(asset_id: i32, width: u32, height: u32, byte_len: usize) -> Option<u32> {
    profiling::scope!("video::validate_frame_layout");
    let Some(expected) = rgba8_frame_bytes_usize(width, height) else {
        logger::warn!("CpuCopyVideoSink {asset_id}: frame dimensions overflow byte count");
        return None;
    };
    if byte_len != expected {
        logger::warn!(
            "CpuCopyVideoSink {asset_id}: frame size mismatch (got {byte_len} bytes, expected {expected})"
        );
        return None;
    }
    let Some(bytes_per_row) = width.checked_mul(RGBA8_BYTES_PER_PIXEL_U32) else {
        logger::warn!("CpuCopyVideoSink {asset_id}: row byte count overflow for width {width}");
        return None;
    };
    Some(bytes_per_row)
}

/// Writes one RGBA frame into a wgpu texture.
fn write_rgba_frame_to_texture(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    bytes: &[u8],
    width: u32,
    height: u32,
    bytes_per_row: u32,
) {
    profiling::scope!("video::write_rgba_frame_to_texture");
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(bytes_per_row),
            rows_per_image: None,
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
}

impl WgpuGstVideoSink for CpuCopyVideoSink {
    fn name(&self) -> &str {
        "CpuCopyVideoSink"
    }

    fn appsink(&self) -> &AppSink {
        &self.sink
    }

    fn poll_texture_change(&mut self) -> Option<(Arc<wgpu::TextureView>, u32, u32, u64)> {
        profiling::scope!("video::poll_texture_change");
        let (view, w, h) = {
            let mut state = self.state.lock().ok()?;
            let view = state.pending_view.take()?;
            (view, state.width, state.height)
        };
        let Some(bytes) = rgba8_frame_bytes_u64(w, h) else {
            logger::warn!(
                "CpuCopyVideoSink {}: frame dimensions overflow resident byte count",
                self.asset_id
            );
            return None;
        };
        Some((view, w, h, bytes))
    }

    fn size(&self) -> Option<IVec2> {
        use gstreamer::prelude::PadExt;
        let pad = self.sink.static_pad("sink")?;
        let caps = pad.current_caps()?;
        let structure = caps.structure(0)?;
        let width = structure.get::<i32>("width").ok()?;
        let height = structure.get::<i32>("height").ok()?;
        (width > 0 && height > 0).then_some(IVec2::new(width, height))
    }
}

/// Returns the byte size of an RGBA8 frame.
fn rgba8_frame_bytes_u64(width: u32, height: u32) -> Option<u64> {
    u64::from(width)
        .checked_mul(u64::from(height))?
        .checked_mul(RGBA8_BYTES_PER_PIXEL_U64)
}

/// Returns the byte size of an RGBA8 frame as a host slice length.
fn rgba8_frame_bytes_usize(width: u32, height: u32) -> Option<usize> {
    rgba8_frame_bytes_u64(width, height)?.try_into().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba8_frame_byte_count_matches_dimensions() {
        assert_eq!(rgba8_frame_bytes_u64(1920, 1080), Some(8_294_400));
        assert_eq!(rgba8_frame_bytes_usize(2, 3), Some(24));
    }

    #[test]
    fn rgba8_frame_byte_count_rejects_overflow() {
        assert_eq!(rgba8_frame_bytes_u64(u32::MAX, u32::MAX), None);
    }

    #[test]
    fn validated_frame_layout_returns_tight_rgba_row_stride() {
        assert_eq!(validated_frame_layout(7, 8, 4, 8 * 4 * 4), Some(32));
    }

    #[test]
    fn validated_frame_layout_rejects_size_mismatch() {
        assert_eq!(validated_frame_layout(7, 8, 4, 8 * 4 * 4 - 1), None);
    }

    #[test]
    fn validated_frame_layout_rejects_row_stride_overflow() {
        let exact_len = rgba8_frame_bytes_usize(u32::MAX, 1).unwrap();

        assert_eq!(validated_frame_layout(7, u32::MAX, 1, exact_len), None);
    }
}
