//! CPU-copy GStreamer sink that uploads decoded RGBA frames into wgpu textures.
//!
//! The pipeline inserts a `videoflip method=vertical-flip` element upstream of
//! this sink (see [`crate::assets::video::player::VideoPlayer::new`]) so frames
//! arrive in the Unity V=0-bottom convention shared by all sampled textures.
//! This sink therefore copies the mapped buffer directly with no orientation
//! transform. GStreamer callbacks only normalize decoded frames into a CPU
//! mailbox; GPU uploads run during renderer asset integration so queue-gate
//! contention never drops the only copy of a decoded frame.

use crate::assets::video::WgpuGstVideoSink;
use crate::gpu::{GpuQueueAccessGate, GpuQueueAccessMode};
use glam::IVec2;
use gstreamer::prelude::ElementExt;
use gstreamer_app::{AppSink, AppSinkCallbacks};
use gstreamer_video::{VideoFormat, VideoFrame, VideoFrameExt, VideoInfo};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Bytes per RGBA8 pixel.
const RGBA8_BYTES_PER_PIXEL_U32: u32 = 4;

/// Bytes per RGBA8 pixel as a `u64`.
const RGBA8_BYTES_PER_PIXEL_U64: u64 = 4;

/// CPU-owned decoded frame waiting for the next renderer-side upload.
struct PendingVideoFrame {
    /// Decoded frame width in pixels.
    width: u32,
    /// Decoded frame height in pixels.
    height: u32,
    /// Tightly packed RGBA8 texels.
    bytes: Vec<u8>,
}

/// GPU upload work prepared from a pending CPU frame.
struct PendingTextureUpload {
    /// CPU frame to upload.
    frame: PendingVideoFrame,
    /// Queue used for CPU-to-GPU frame uploads.
    queue: Arc<wgpu::Queue>,
    /// Texture receiving the frame.
    texture: Arc<wgpu::Texture>,
    /// Shared gate serializing texture uploads with renderer submits and OpenXR queue access.
    queue_access_gate: GpuQueueAccessGate,
}

/// Internal state shared between [`CpuCopyVideoSink`] and the appsink callback.
struct SinkState {
    /// Device used to allocate replacement textures when the decoded size changes.
    device: Option<Arc<wgpu::Device>>,
    /// Queue used for CPU-to-GPU frame uploads.
    queue: Option<Arc<wgpu::Queue>>,
    /// Shared gate serializing texture uploads with renderer submits and OpenXR queue access.
    queue_access_gate: GpuQueueAccessGate,
    /// The texture currently receiving renderer-side frame uploads.
    write_texture: Option<Arc<wgpu::Texture>>,
    /// Width of [`Self::write_texture`].
    width: u32,
    /// Height of [`Self::write_texture`].
    height: u32,
    /// Set to `Some` when a new texture is created, consumed by [`CpuCopyVideoSink::poll_texture_change`].
    pending_view: Option<Arc<wgpu::TextureView>>,
    /// Latest decoded frame waiting for renderer-side upload.
    pending_frame: Option<PendingVideoFrame>,
    /// Stops callbacks from allocating or uploading after renderer shutdown begins.
    shutdown: bool,
}

impl SinkState {
    /// Reallocates the write texture when the video size changes.
    /// Returns `true` if a new texture was created.
    fn resize_if_needed(&mut self, asset_id: i32, width: u32, height: u32) -> bool {
        profiling::scope!("video::cpu_copy_resize_if_needed");
        let Some(device) = self.device.as_ref() else {
            return false;
        };
        if self.width == width && self.height == height && self.write_texture.is_some() {
            return false;
        }

        let max_dimension = device.limits().max_texture_dimension_2d;
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

        let texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
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
        crate::profiling::note_resource_churn!(TextureView, "assets::video_cpu_copy_view");

        self.write_texture = Some(texture);
        self.width = width;
        self.height = height;
        self.pending_view = Some(view);

        true
    }

    /// Releases all GPU handles held by the callback state.
    fn begin_shutdown(&mut self) {
        self.shutdown = true;
        self.device = None;
        self.queue = None;
        self.write_texture = None;
        self.pending_view = None;
        self.pending_frame = None;
        self.width = 0;
        self.height = 0;
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
    /// Fast shutdown flag checked before callbacks pull or map samples.
    shutdown: Arc<AtomicBool>,
}

impl CpuCopyVideoSink {
    /// Creates a CPU-copy sink backed by the supplied wgpu device, queue, and queue gate.
    pub fn new(
        asset_id: i32,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        queue_access_gate: GpuQueueAccessGate,
    ) -> Self {
        let sink = build_rgba_appsink();
        let shutdown = Arc::new(AtomicBool::new(false));
        let state = Arc::new(Mutex::new(SinkState {
            device: Some(device),
            queue: Some(queue),
            queue_access_gate,
            write_texture: None,
            width: 0,
            height: 0,
            pending_view: None,
            pending_frame: None,
            shutdown: false,
        }));
        install_sample_callback(&sink, asset_id, Arc::clone(&state), Arc::clone(&shutdown));
        Self {
            asset_id,
            sink,
            state,
            shutdown,
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
fn install_sample_callback(
    sink: &AppSink,
    asset_id: i32,
    state: Arc<Mutex<SinkState>>,
    shutdown: Arc<AtomicBool>,
) {
    sink.set_callbacks(
        AppSinkCallbacks::builder()
            .new_sample(move |appsink| handle_sample(asset_id, &state, &shutdown, appsink))
            .build(),
    );
}

/// Handles one decoded sample from GStreamer.
fn handle_sample(
    asset_id: i32,
    state: &Arc<Mutex<SinkState>>,
    shutdown: &AtomicBool,
    appsink: &AppSink,
) -> Result<gstreamer::FlowSuccess, gstreamer::FlowError> {
    profiling::scope!("video::cpu_copy_sample");
    if shutdown.load(Ordering::Acquire) {
        return Ok(gstreamer::FlowSuccess::Ok);
    }
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
    if shutdown.load(Ordering::Acquire) {
        return Ok(gstreamer::FlowSuccess::Ok);
    }
    let Some(frame) = tight_rgba_frame_from_sample(asset_id, &sample) else {
        return Ok(gstreamer::FlowSuccess::Ok);
    };
    if shutdown.load(Ordering::Acquire) {
        return Ok(gstreamer::FlowSuccess::Ok);
    }
    store_pending_frame(asset_id, state, frame);
    Ok(gstreamer::FlowSuccess::Ok)
}

/// Extracts one tightly packed RGBA frame from a GStreamer sample.
fn tight_rgba_frame_from_sample(
    asset_id: i32,
    sample: &gstreamer::Sample,
) -> Option<PendingVideoFrame> {
    profiling::scope!("video::tight_rgba_frame_from_sample");
    let Some(caps) = sample.caps() else {
        logger::warn!("CpuCopyVideoSink {asset_id}: sample without caps");
        return None;
    };
    let video_info = match VideoInfo::from_caps(caps) {
        Ok(info) => info,
        Err(e) => {
            logger::warn!("CpuCopyVideoSink {asset_id}: failed to parse video caps: {e}");
            return None;
        }
    };
    if video_info.format() != VideoFormat::Rgba {
        logger::warn!(
            "CpuCopyVideoSink {asset_id}: negotiated unsupported format {}",
            video_info.name()
        );
        return None;
    }
    let Some(buffer) = sample.buffer_owned() else {
        logger::warn!("CpuCopyVideoSink {asset_id}: sample without buffer");
        return None;
    };
    let Ok(frame) = VideoFrame::from_buffer_readable(buffer, &video_info) else {
        logger::warn!("CpuCopyVideoSink {asset_id}: failed to map video frame");
        return None;
    };
    let Some(row_stride) = frame.info().stride().first().copied() else {
        logger::warn!("CpuCopyVideoSink {asset_id}: mapped frame without a color plane stride");
        return None;
    };
    let plane = match frame.plane_data(0) {
        Ok(plane) => plane,
        Err(e) => {
            logger::warn!("CpuCopyVideoSink {asset_id}: failed to read color plane: {e}");
            return None;
        }
    };
    let width = frame.width();
    let height = frame.height();
    let bytes = copy_tight_rgba_plane(asset_id, width, height, row_stride, plane)?;
    Some(PendingVideoFrame {
        width,
        height,
        bytes,
    })
}

/// Copies the visible RGBA pixels out of a potentially padded GStreamer plane.
fn copy_tight_rgba_plane(
    asset_id: i32,
    width: u32,
    height: u32,
    source_row_stride: i32,
    source: &[u8],
) -> Option<Vec<u8>> {
    profiling::scope!("video::copy_tight_rgba_plane");
    let Some(tight_row_bytes_u32) = width.checked_mul(RGBA8_BYTES_PER_PIXEL_U32) else {
        logger::warn!("CpuCopyVideoSink {asset_id}: row byte count overflow for width {width}");
        return None;
    };
    let tight_row_bytes = tight_row_bytes_u32 as usize;
    let height_usize = height as usize;
    let Some(tight_frame_bytes) = rgba8_frame_bytes_usize(width, height) else {
        logger::warn!("CpuCopyVideoSink {asset_id}: frame dimensions overflow byte count");
        return None;
    };
    let Ok(source_row_stride) = usize::try_from(source_row_stride) else {
        logger::warn!("CpuCopyVideoSink {asset_id}: negative row stride in decoded frame");
        return None;
    };
    if source_row_stride < tight_row_bytes {
        logger::warn!(
            "CpuCopyVideoSink {asset_id}: source row stride {source_row_stride} smaller than visible row {tight_row_bytes}"
        );
        return None;
    }
    let Some(required_source_bytes) =
        required_padded_plane_bytes(source_row_stride, tight_row_bytes, height_usize)
    else {
        logger::warn!("CpuCopyVideoSink {asset_id}: padded frame dimensions overflow byte count");
        return None;
    };
    if source.len() < required_source_bytes {
        logger::warn!(
            "CpuCopyVideoSink {asset_id}: source plane too short (got {} bytes, need {required_source_bytes})",
            source.len()
        );
        return None;
    }

    if source_row_stride == tight_row_bytes {
        return Some(source[..tight_frame_bytes].to_vec());
    }

    let mut tight = vec![0; tight_frame_bytes];
    for row in 0..height_usize {
        let source_start = row * source_row_stride;
        let source_end = source_start + tight_row_bytes;
        let target_start = row * tight_row_bytes;
        let target_end = target_start + tight_row_bytes;
        tight[target_start..target_end].copy_from_slice(&source[source_start..source_end]);
    }
    Some(tight)
}

/// Returns the required source byte count for a padded plane.
fn required_padded_plane_bytes(
    source_row_stride: usize,
    tight_row_bytes: usize,
    height: usize,
) -> Option<usize> {
    match height {
        0 => Some(0),
        _ => source_row_stride
            .checked_mul(height - 1)?
            .checked_add(tight_row_bytes),
    }
}

/// Stores the latest decoded CPU frame for renderer-side upload.
fn store_pending_frame(asset_id: i32, state: &Arc<Mutex<SinkState>>, frame: PendingVideoFrame) {
    let Ok(mut state) = state.lock() else {
        logger::warn!("CpuCopyVideoSink {asset_id}: state lock poisoned while storing frame");
        return;
    };
    if !state.shutdown {
        state.pending_frame = Some(frame);
    }
}

/// Prepares one pending frame for upload, allocating a replacement texture if needed.
fn prepare_pending_upload(
    asset_id: i32,
    state: &Arc<Mutex<SinkState>>,
) -> Option<PendingTextureUpload> {
    let Ok(mut state) = state.lock() else {
        logger::warn!("CpuCopyVideoSink {asset_id}: state lock poisoned while preparing upload");
        return None;
    };
    if state.shutdown {
        return None;
    }
    let frame = state.pending_frame.take()?;
    state.resize_if_needed(asset_id, frame.width, frame.height);
    let Some(queue) = state.queue.clone() else {
        state.pending_frame = Some(frame);
        return None;
    };
    let Some(texture) = state.write_texture.clone() else {
        logger::warn!("CpuCopyVideoSink {asset_id}: no texture available after resize");
        return None;
    };
    Some(PendingTextureUpload {
        frame,
        queue,
        texture,
        queue_access_gate: state.queue_access_gate.clone(),
    })
}

/// Restores a frame after a yielded upload unless a newer decoded frame already arrived.
fn restore_pending_frame_if_slot_empty(
    asset_id: i32,
    state: &Arc<Mutex<SinkState>>,
    frame: PendingVideoFrame,
) {
    let Ok(mut state) = state.lock() else {
        logger::warn!("CpuCopyVideoSink {asset_id}: state lock poisoned while restoring frame");
        return;
    };
    if !state.shutdown && state.pending_frame.is_none() {
        state.pending_frame = Some(frame);
    }
}

/// Attempts to acquire the queue gate for a decoded video frame upload.
fn acquire_video_upload_gate(
    gate: &GpuQueueAccessGate,
    mode: GpuQueueAccessMode,
) -> Option<parking_lot::MutexGuard<'_, ()>> {
    gate.lock_for(mode)
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

    fn poll_texture_change(
        &mut self,
        queue_access_mode: GpuQueueAccessMode,
    ) -> Option<(Arc<wgpu::TextureView>, u32, u32, u64)> {
        profiling::scope!("video::poll_texture_change");
        if self.shutdown.load(Ordering::Acquire) {
            return None;
        }
        let upload = prepare_pending_upload(self.asset_id, &self.state)?;
        let bytes_per_row = validated_frame_layout(
            self.asset_id,
            upload.frame.width,
            upload.frame.height,
            upload.frame.bytes.len(),
        )?;
        let Some(_gate) = acquire_video_upload_gate(&upload.queue_access_gate, queue_access_mode)
        else {
            restore_pending_frame_if_slot_empty(self.asset_id, &self.state, upload.frame);
            return None;
        };
        write_rgba_frame_to_texture(
            &upload.queue,
            &upload.texture,
            &upload.frame.bytes,
            upload.frame.width,
            upload.frame.height,
            bytes_per_row,
        );
        let (view, w, h) = {
            let mut state = self.state.lock().ok()?;
            if state.shutdown {
                return None;
            }
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

    fn begin_shutdown(&mut self) {
        profiling::scope!("video::cpu_copy_begin_shutdown");
        self.shutdown.store(true, Ordering::Release);
        self.sink.set_callbacks(AppSinkCallbacks::builder().build());
        match self.state.lock() {
            Ok(mut state) => state.begin_shutdown(),
            Err(poisoned) => {
                logger::warn!(
                    "CpuCopyVideoSink {}: state lock poisoned during shutdown",
                    self.asset_id
                );
                poisoned.into_inner().begin_shutdown();
            }
        }
    }

    fn size(&self) -> Option<IVec2> {
        if self.shutdown.load(Ordering::Acquire) {
            return None;
        }
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

    fn pending_frame(width: u32, height: u32, fill: u8) -> PendingVideoFrame {
        PendingVideoFrame {
            width,
            height,
            bytes: vec![fill; rgba8_frame_bytes_usize(width, height).unwrap()],
        }
    }

    fn state_without_gpu() -> Arc<Mutex<SinkState>> {
        Arc::new(Mutex::new(SinkState {
            device: None,
            queue: None,
            queue_access_gate: GpuQueueAccessGate::new(),
            write_texture: None,
            width: 0,
            height: 0,
            pending_view: None,
            pending_frame: None,
            shutdown: false,
        }))
    }

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

    #[test]
    fn tight_rgba_copy_accepts_tightly_packed_rows() {
        let source: Vec<u8> = (0..16).collect();

        assert_eq!(copy_tight_rgba_plane(7, 2, 2, 8, &source), Some(source));
    }

    #[test]
    fn tight_rgba_copy_strips_padded_rows() {
        let source = [
            1, 2, 3, 4, 5, 6, 7, 8, 90, 91, 92, 93, 9, 10, 11, 12, 13, 14, 15, 16, 94, 95, 96, 97,
        ];

        assert_eq!(
            copy_tight_rgba_plane(7, 2, 2, 12, &source),
            Some(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        );
    }

    #[test]
    fn tight_rgba_copy_rejects_negative_stride() {
        assert_eq!(copy_tight_rgba_plane(7, 2, 2, -8, &[0; 16]), None);
    }

    #[test]
    fn tight_rgba_copy_rejects_short_stride() {
        assert_eq!(copy_tight_rgba_plane(7, 2, 2, 7, &[0; 16]), None);
    }

    #[test]
    fn tight_rgba_copy_rejects_short_source_plane() {
        assert_eq!(copy_tight_rgba_plane(7, 2, 2, 12, &[0; 16]), None);
    }

    #[test]
    fn tight_rgba_copy_rejects_overflowing_dimensions() {
        assert_eq!(copy_tight_rgba_plane(7, u32::MAX, 2, 16, &[]), None);
    }

    #[test]
    fn required_padded_plane_bytes_counts_visible_tail_only() {
        assert_eq!(required_padded_plane_bytes(12, 8, 2), Some(20));
        assert_eq!(required_padded_plane_bytes(12, 8, 0), Some(0));
    }

    #[test]
    fn video_upload_gate_acquires_when_uncontended() {
        let gate = GpuQueueAccessGate::new();

        assert!(acquire_video_upload_gate(&gate, GpuQueueAccessMode::NonBlocking).is_some());
    }

    #[test]
    fn busy_upload_gate_keeps_frame_pending_when_slot_is_empty() {
        let state = state_without_gpu();
        let gate = GpuQueueAccessGate::new();
        let _held = gate.lock();
        let frame = pending_frame(1, 1, 5);

        assert!(acquire_video_upload_gate(&gate, GpuQueueAccessMode::NonBlocking).is_none());
        restore_pending_frame_if_slot_empty(7, &state, frame);

        let pending_bytes = state
            .lock()
            .unwrap()
            .pending_frame
            .as_ref()
            .map(|frame| frame.bytes.clone());
        assert_eq!(pending_bytes.as_deref(), Some(&[5, 5, 5, 5][..]));
    }

    #[test]
    fn failed_upload_restore_does_not_replace_newer_frame() {
        let state = state_without_gpu();
        store_pending_frame(7, &state, pending_frame(1, 1, 9));

        restore_pending_frame_if_slot_empty(7, &state, pending_frame(1, 1, 5));

        let pending_bytes = state
            .lock()
            .unwrap()
            .pending_frame
            .as_ref()
            .map(|frame| frame.bytes.clone());
        assert_eq!(pending_bytes.as_deref(), Some(&[9, 9, 9, 9][..]));
    }
}
