//! The [`VideoPlayer`] holds the GStreamer pipeline and handles incoming updates from host.

use super::WgpuGstVideoSink;
use super::audio_sink::ResoniteAudioSink;
use super::clock::{
    MediaDuration, adjusted_host_position, clock_time_from_seconds, should_seek_to_host_position,
    target_state_for_update, unix_nanos_now,
};
use super::cpu_copy::CpuCopyVideoSink;
use super::ready::{video_audio_track_eq, video_texture_ready_eq};
use super::source::source_uri;
use crate::assets::video::VideoTextureFrameSink;
use crate::gpu::{GpuQueueAccessGate, GpuQueueAccessMode};
use glam::IVec2;
use gstreamer::prelude::{ElementExt, ElementExtManual};

use gstreamer::StreamCollection;
use renderide_shared::ipc::DualQueueIpc;
use renderide_shared::{
    RendererCommand, VideoAudioTrack, VideoTextureClockErrorState, VideoTextureLoad,
    VideoTextureReady, VideoTextureStartAudioTrack, VideoTextureUpdate,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

/// Fallback audio rate used when the host sends an invalid sample rate.
const DEFAULT_AUDIO_SAMPLE_RATE: i32 = 48_000;

/// Poll interval for applying host playback updates to GStreamer.
const UPDATE_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(16);

/// Holds the GStreamer pipeline and handles incoming updates from host.
pub struct VideoPlayer {
    /// Host video texture asset id.
    asset_id: i32,
    /// GStreamer playbin pipeline for this video texture.
    pipeline: gstreamer::Element,
    /// AppSink used to forward decoded audio samples to the host.
    audio_sink: ResoniteAudioSink,
    /// Active decoded-video sink backend.
    video_sink: Box<dyn WgpuGstVideoSink + Send>,
    /// Audio sample rate requested by the host audio system.
    audio_sample_rate: i32,
    /// Stream metadata discovered from GStreamer stream-collection messages.
    streams: VideoStreamCatalog,
    /// Stores the latest [`VideoTextureUpdate`] until it gets processed by the update thread.
    pending_update: Arc<Mutex<Option<VideoTextureUpdate>>>,
    /// Most recently received host update, retained even after the worker thread applies it so
    /// [`Self::sample_clock_error`] can keep reporting drift each frame against
    /// [`VideoTextureUpdate::decoded_time`] until the host sends another update. Reusing the
    /// last-received instance keeps the drift calculation continuous between host updates.
    last_update: Arc<Mutex<Option<VideoTextureUpdate>>>,
    /// Last successfully queued ready message, used to avoid redundant background IPC.
    last_ready_message: Option<VideoTextureReady>,
    /// Shared shutdown flag checked by the update thread.
    shutdown: Arc<AtomicBool>,
    /// Worker applying GStreamer playback updates and driving the pipeline to `Null` on shutdown.
    update_thread: Option<JoinHandle<()>>,
}

/// Selectable audio stream metadata paired with the host-facing track descriptor.
struct AudioStreamMetadata {
    /// GStreamer stream id used in `select-streams` events.
    stream_id: String,
    /// Track descriptor sent to the host.
    track: VideoAudioTrack,
}

/// Stream metadata discovered for one video player.
#[derive(Default)]
struct VideoStreamCatalog {
    /// Selectable audio streams, ordered by host track index.
    audio_streams: Vec<AudioStreamMetadata>,
    /// GStreamer stream ids for video streams that should stay selected with audio.
    video_stream_ids: Vec<String>,
}

impl VideoStreamCatalog {
    /// Builds a stream catalog from a GStreamer stream collection.
    fn from_collection(collection: &StreamCollection, sample_rate: i32) -> Self {
        let mut audio_streams = Vec::new();
        let mut video_stream_ids = Vec::new();

        for stream in collection {
            let stream_type = stream.stream_type();
            if stream_type.contains(gstreamer::StreamType::VIDEO)
                && let Some(stream_id) = stream.stream_id()
            {
                video_stream_ids.push(stream_id.to_string());
            }

            if stream_type.contains(gstreamer::StreamType::AUDIO) {
                let Ok(track_index) = i32::try_from(audio_streams.len()) else {
                    break;
                };
                if let Some(metadata) =
                    AudioStreamMetadata::from_stream(track_index, &stream, sample_rate)
                {
                    audio_streams.push(metadata);
                }
            }
        }

        Self {
            audio_streams,
            video_stream_ids,
        }
    }

    /// Replaces this catalog with metadata from `collection` and returns whether it changed.
    fn replace_from_collection(&mut self, collection: &StreamCollection, sample_rate: i32) -> bool {
        let next = Self::from_collection(collection, sample_rate);
        if self.same_as(&next) {
            return false;
        }
        *self = next;
        true
    }

    /// Returns host-facing audio track descriptors.
    fn audio_tracks(&self) -> Vec<VideoAudioTrack> {
        self.audio_streams
            .iter()
            .map(|metadata| metadata.track.clone())
            .collect()
    }

    /// Returns the number of selectable audio streams.
    fn audio_track_count(&self) -> usize {
        self.audio_streams.len()
    }

    /// Builds stream ids for selecting one host audio track while retaining video streams.
    fn selected_stream_ids(&self, audio_track_index: usize) -> Option<Vec<&str>> {
        let audio_stream = self.audio_streams.get(audio_track_index)?;
        let mut stream_ids = Vec::with_capacity(self.video_stream_ids.len() + 1);
        stream_ids.extend(self.video_stream_ids.iter().map(String::as_str));
        stream_ids.push(audio_stream.stream_id.as_str());
        Some(stream_ids)
    }

    /// Returns whether this catalog has the same host-visible and GStreamer stream metadata.
    fn same_as(&self, other: &Self) -> bool {
        self.video_stream_ids == other.video_stream_ids
            && audio_stream_metadata_eq(&self.audio_streams, &other.audio_streams)
    }
}

impl AudioStreamMetadata {
    /// Builds selectable audio metadata from a GStreamer stream.
    fn from_stream(track_index: i32, stream: &gstreamer::Stream, sample_rate: i32) -> Option<Self> {
        let stream_id = stream.stream_id()?.to_string();
        let tags = stream.tags();
        Some(Self {
            stream_id,
            track: VideoAudioTrack {
                index: track_index,
                channel_count: channel_count_from_stream(stream),
                sample_rate: normalized_audio_sample_rate(sample_rate),
                language_code: tags
                    .as_ref()
                    .and_then(|t| t.get::<gstreamer::tags::LanguageCode>())
                    .map(|v| v.get().to_owned()),
                name: tags
                    .as_ref()
                    .and_then(|t| t.get::<gstreamer::tags::Title>())
                    .map(|v| v.get().to_owned()),
            },
        })
    }
}

/// Builds a `videoflip method=vertical-flip` element that converts GStreamer's
/// natural top-down V output to the Unity (V=0 bottom) convention shaders expect.
///
/// Returns `None` if `videoflip` is unavailable on the host system; the caller
/// then builds the pipeline without a `video-filter` so playback still proceeds
/// (frames will appear upside down until `gst-plugins-good` is installed).
fn build_vertical_flip_filter(asset_id: i32) -> Option<gstreamer::Element> {
    if gstreamer::ElementFactory::find("videoflip").is_none() {
        logger::warn!(
            "video texture {asset_id}: 'videoflip' element unavailable; \
             frames may appear vertically flipped (install gst-plugins-good)"
        );
        return None;
    }
    match gstreamer::ElementFactory::make("videoflip")
        .property_from_str("method", "vertical-flip")
        .build()
    {
        Ok(element) => Some(element),
        Err(e) => {
            logger::warn!("video texture {asset_id}: failed to build videoflip: {e}");
            None
        }
    }
}

impl VideoPlayer {
    /// Creates a new [`VideoPlayer`] using [`VideoTextureLoad`].
    pub fn new(
        l: VideoTextureLoad,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        queue_access_gate: GpuQueueAccessGate,
    ) -> Option<Self> {
        let id = l.asset_id;
        let audio_sample_rate = normalized_audio_sample_rate(l.audio_system_sample_rate);

        if let Err(e) = gstreamer::init() {
            logger::error!("gstreamer init failed: {e}");
            return None;
        }

        let audio_sink = ResoniteAudioSink::new(audio_sample_rate);

        // GStreamer-backed video textures currently use the CPU-copy sink on all platforms.
        let video_sink = Box::new(CpuCopyVideoSink::new(id, device, queue, queue_access_gate));

        let uri = match source_uri(l.source.as_deref()) {
            Ok(Some(uri)) => uri,
            Ok(None) => {
                logger::warn!("video texture {id}: load skipped because source is missing");
                return None;
            }
            Err(e) => {
                logger::error!("video texture {id}: failed to convert source path to URI: {e}");
                return None;
            }
        };

        let video_filter = build_vertical_flip_filter(id);
        let mut pipeline_builder = gstreamer::ElementFactory::make("playbin3")
            .property("uri", &uri)
            .property("audio-sink", audio_sink.appsink())
            .property("video-sink", video_sink.appsink());
        if let Some(flip) = video_filter.as_ref() {
            pipeline_builder = pipeline_builder.property("video-filter", flip);
        }
        let pipeline = match pipeline_builder.build() {
            Ok(p) => p,
            Err(e) => {
                logger::error!("video texture {}: failed to create playbin: {e}", id);
                return None;
            }
        };

        if let Err(e) = pipeline.set_state(gstreamer::State::Playing) {
            logger::error!("video texture {}: failed to start playbin: {e}", id);
            return None;
        }

        let pending_update: Arc<Mutex<Option<VideoTextureUpdate>>> = Arc::new(Mutex::new(None));
        let last_update: Arc<Mutex<Option<VideoTextureUpdate>>> = Arc::new(Mutex::new(None));
        let shutdown: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));

        let update_thread = Some(Self::spawn_update_thread(
            pipeline.clone(),
            Arc::clone(&pending_update),
            Arc::clone(&shutdown),
        ));

        Some(Self {
            asset_id: id,
            pipeline,
            audio_sink,
            video_sink,
            audio_sample_rate,
            streams: VideoStreamCatalog::default(),
            pending_update,
            last_update,
            last_ready_message: None,
            shutdown,
            update_thread,
        })
    }

    /// Handles [`VideoTextureStartAudioTrack`].
    /// Opens a shared memory queue to send audio back to host, and assigns the callback to the sink.
    pub fn handle_start_audio_track(&self, s: VideoTextureStartAudioTrack) {
        if self.shutdown.load(Ordering::Acquire) {
            return;
        }
        let id = self.asset_id;
        let Some(index) = validated_audio_track_index(s.audio_track_index) else {
            logger::warn!(
                "video texture {id}: invalid audio track index {}",
                s.audio_track_index
            );
            return;
        };

        let Some(stream_ids) = self.streams.selected_stream_ids(index) else {
            logger::warn!("video texture {id}: audio track index {index} out of range");
            return;
        };

        let event = gstreamer::event::SelectStreams::new(stream_ids);
        if !self.pipeline.send_event(event) {
            logger::warn!("video texture {id}: failed to select audio track {index}");
        }

        let Some(queue_name) = s.queue_name else {
            return;
        };

        if let Err(e) = self.audio_sink.attach_queue(&queue_name, s.queue_capacity) {
            logger::warn!("video texture {id}: failed to attach audio queue: {e}");
        }
    }

    /// Schedules a video player state update from [`VideoTextureUpdate`].
    ///
    /// Also stores the update as the latest snapshot for [`Self::sample_clock_error`]. The
    /// `decoded_time` field is set by the IPC unpack at receive time, so retaining the same
    /// update across multiple frames is correct: `(now - decoded_time)` keeps growing until the
    /// host sends a fresh update, matching the host video-clock contract.
    pub fn handle_update(&self, u: VideoTextureUpdate) {
        if self.shutdown.load(Ordering::Acquire) {
            return;
        }
        match self.last_update.lock() {
            Ok(mut slot) => *slot = Some(u.clone()),
            Err(_) => logger::warn!(
                "video texture {}: last-update lock poisoned; clock-error sample skipped",
                self.asset_id
            ),
        }
        match self.pending_update.lock() {
            Ok(mut pending_update) => *pending_update = Some(u),
            Err(_) => logger::warn!(
                "video texture {}: update state lock poisoned; dropping host update",
                self.asset_id
            ),
        }
    }

    /// Handles texture changes from the sink and running the GStreamer event loop,
    /// as well as sending of [`VideoTextureReady`].
    pub fn process_events(
        &mut self,
        frame_sink: &mut impl VideoTextureFrameSink,
        ipc: &mut Option<&mut DualQueueIpc>,
        queue_access_mode: GpuQueueAccessMode,
    ) {
        profiling::scope!("video::process_events");
        if self.shutdown.load(Ordering::Acquire) {
            return;
        }
        let Some(bus) = self.pipeline.bus() else {
            return;
        };

        // Forward any texture the sink created since last frame. Ensure the pool entry exists here
        // so a frame cannot be lost if video data arrives before sampler properties.
        let id = self.asset_id;
        if let Some((view, w, h, bytes)) = self.video_sink.poll_texture_change(queue_access_mode)
            && !frame_sink.set_video_texture_frame(id, view, w, h, bytes)
        {
            logger::warn!("video texture {id}: GPU placeholder unavailable for decoded frame");
        }

        while let Some(msg) = bus.timed_pop(gstreamer::ClockTime::ZERO) {
            match msg.view() {
                gstreamer::MessageView::AsyncDone(_) => self.handle_async_done(ipc),
                gstreamer::MessageView::StreamCollection(sc) => {
                    self.handle_stream_collection(sc.stream_collection(), ipc);
                }
                gstreamer::MessageView::Error(e) => {
                    logger::error!(
                        "video texture {}: gstreamer error: {}",
                        self.asset_id,
                        e.error()
                    );
                }
                _ => {}
            }
        }
    }

    /// Gets called when the pipeline is done prerolling and we have caps available.
    fn handle_async_done(&mut self, ipc: &mut Option<&mut DualQueueIpc>) {
        let size = self.video_sink.size();
        let duration = query_media_duration(&self.pipeline);

        self.send_ready(
            ipc,
            duration.ready_length_seconds(),
            size.unwrap_or_default(),
        );
    }

    /// Gets called when the video player becomes aware of audio/video streams.
    fn handle_stream_collection(
        &mut self,
        collection: StreamCollection,
        ipc: &mut Option<&mut DualQueueIpc>,
    ) {
        let id = self.asset_id;
        if !self
            .streams
            .replace_from_collection(&collection, self.audio_sample_rate)
        {
            return;
        }

        logger::info!(
            "video texture {id}: discovered {} audio tracks",
            self.streams.audio_track_count()
        );
        self.refresh_ready_if_already_sent(ipc);
    }

    /// Spawns the worker that applies playback state updates off the render thread.
    fn spawn_update_thread(
        pipeline: gstreamer::Element,
        pending_update: Arc<Mutex<Option<VideoTextureUpdate>>>,
        shutdown: Arc<AtomicBool>,
    ) -> JoinHandle<()> {
        thread::spawn(move || {
            while !shutdown.load(Ordering::Acquire) {
                thread::sleep(UPDATE_POLL_INTERVAL);

                let update = if let Ok(mut pending_update) = pending_update.lock() {
                    match pending_update.take() {
                        Some(update) => update,
                        None => continue,
                    }
                } else {
                    logger::warn!("video texture update thread: update lock poisoned");
                    break;
                };
                apply_update_to_pipeline(&pipeline, &update);
            }

            // GStreamer shutdown can block on damaged media; this work stays off the render thread.
            if let Err(e) = pipeline.set_state(gstreamer::State::Null) {
                logger::error!("failed to set pipeline to Null on shutdown: {e}");
            }
        })
    }

    /// Starts cooperative shutdown without blocking the render thread.
    pub fn begin_shutdown(&mut self) {
        profiling::scope!("video::begin_shutdown");
        if self.shutdown.swap(true, Ordering::AcqRel) {
            return;
        }

        if let Ok(mut pending_update) = self.pending_update.lock() {
            let _ = pending_update.take();
        } else {
            logger::warn!(
                "video texture {}: update lock poisoned during shutdown",
                self.asset_id
            );
        }

        self.audio_sink.begin_shutdown();
        self.video_sink.begin_shutdown();
    }

    /// Joins the update worker once it has completed pipeline shutdown.
    pub fn poll_shutdown_complete(&mut self) -> bool {
        if !self.shutdown.load(Ordering::Acquire) {
            self.begin_shutdown();
        }
        let Some(handle) = self.update_thread.as_ref() else {
            return true;
        };
        if !handle.is_finished() {
            return false;
        }
        let Some(handle) = self.update_thread.take() else {
            return true;
        };
        if handle.join().is_err() {
            logger::warn!(
                "video texture {}: update thread panicked during shutdown",
                self.asset_id
            );
        }
        true
    }

    /// Samples this player's clock error against the host's most recently received playback request.
    ///
    /// The error is `pipeline_position - adjusted_host_position`, where the adjusted position
    /// advances unconditionally at real-time from
    /// [`VideoTextureUpdate::decoded_time`] (set by the IPC unpack at receive time). Returns `None`
    /// until at least one update has arrived or when the pipeline position cannot be queried.
    pub fn sample_clock_error(&self) -> Option<VideoTextureClockErrorState> {
        if self.shutdown.load(Ordering::Acquire) {
            return None;
        }
        if !query_media_duration(&self.pipeline).reports_clock_error() {
            return None;
        }

        let update = if let Ok(slot) = self.last_update.lock() {
            slot.clone()?
        } else {
            logger::warn!(
                "video texture {}: last-update lock poisoned; skipping clock error sample",
                self.asset_id
            );
            return None;
        };
        let current = query_position_seconds(&self.pipeline)?;
        let now_nanos = unix_nanos_now();
        let adjusted = adjusted_host_position(&update, now_nanos);
        Some(VideoTextureClockErrorState {
            asset_id: self.asset_id,
            current_clock_error: (current - adjusted) as f32,
        })
    }

    /// Resends readiness after metadata changes when the host has already seen an initial ready.
    fn refresh_ready_if_already_sent(&mut self, ipc: &mut Option<&mut DualQueueIpc>) {
        if self.last_ready_message.is_none() {
            return;
        }

        let size = self.video_sink.size().unwrap_or_default();
        let duration = query_media_duration(&self.pipeline);
        self.send_ready(ipc, duration.ready_length_seconds(), size);
    }

    /// Sends a ready message when it differs from the last successfully queued one.
    fn send_ready(&mut self, ipc: &mut Option<&mut DualQueueIpc>, length: f64, size: IVec2) {
        let message = VideoTextureReady {
            length,
            size,
            has_alpha: false,
            asset_id: self.asset_id,
            instance_changed: true,
            playback_engine: Some(format!("GStreamer ({})", self.video_sink.name())),
            audio_tracks: self.streams.audio_tracks(),
        };

        if let Some(last) = self.last_ready_message.as_ref()
            && video_texture_ready_eq(last, &message)
        {
            return;
        }

        logger::info!(
            "video texture {}: loaded: size={:?}, length={}",
            self.asset_id,
            size,
            length
        );

        let Some(ipc) = ipc else {
            return;
        };

        if ipc.send_background_reliable(RendererCommand::VideoTextureReady(message.clone())) {
            self.last_ready_message = Some(message);
        } else {
            logger::warn!(
                "video texture {}: failed to queue ready message",
                self.asset_id
            );
        }
    }
}

impl Drop for VideoPlayer {
    fn drop(&mut self) {
        self.begin_shutdown();
        let _ = self.poll_shutdown_complete();
    }
}

/// Returns a positive host audio sample rate or a stable fallback.
fn normalized_audio_sample_rate(sample_rate: i32) -> i32 {
    if sample_rate > 0 {
        sample_rate
    } else {
        DEFAULT_AUDIO_SAMPLE_RATE
    }
}

/// Converts a host audio track index into a safe vector index.
fn validated_audio_track_index(index: i32) -> Option<usize> {
    usize::try_from(index).ok()
}

/// Queries current playback position in seconds.
fn query_position_seconds(pipeline: &gstreamer::Element) -> Option<f64> {
    let mut query = gstreamer::query::Position::new(gstreamer::Format::Time);
    if !pipeline.query(&mut query) {
        return None;
    }
    match query.result() {
        gstreamer::GenericFormattedValue::Time(Some(current)) => {
            Some(current.nseconds() as f64 / 1_000_000_000.0)
        }
        _ => None,
    }
}

/// Applies one host playback update to the GStreamer pipeline.
fn apply_update_to_pipeline(pipeline: &gstreamer::Element, update: &VideoTextureUpdate) {
    profiling::scope!("video::apply_update");
    if let Err(e) = pipeline.set_state(target_state_for_update(update)) {
        logger::warn!("video texture update: failed to set pipeline state: {e}");
    }
    if !query_media_duration(pipeline).supports_timeline_seeking() {
        return;
    }
    let Some(current_seconds) = query_position_seconds(pipeline) else {
        return;
    };
    if should_seek_to_host_position(current_seconds, update) {
        let target = clock_time_from_seconds(update.position);
        if let Err(e) = pipeline.seek_simple(
            gstreamer::SeekFlags::FLUSH | gstreamer::SeekFlags::KEY_UNIT,
            target,
        ) {
            logger::warn!("video texture update: failed to seek pipeline: {e}");
        }
    }
}

/// Queries the pipeline for the host-visible media duration.
fn query_media_duration(pipeline: &gstreamer::Element) -> MediaDuration {
    let mut query = gstreamer::query::Duration::new(gstreamer::Format::Time);
    if !pipeline.query(&mut query) {
        return MediaDuration::Stream;
    }
    match query.result() {
        gstreamer::GenericFormattedValue::Time(duration) => {
            MediaDuration::from_clock_time(duration)
        }
        _ => MediaDuration::Stream,
    }
}

/// Extracts the channel count out of a stream.
fn channel_count_from_stream(stream: &gstreamer::Stream) -> i32 {
    stream
        .caps()
        .and_then(|caps| {
            caps.structure(0)
                .and_then(|s| s.get::<i32>("channels").ok())
        })
        .unwrap_or(2)
}

/// Compares selectable audio stream metadata slices.
fn audio_stream_metadata_eq(a: &[AudioStreamMetadata], b: &[AudioStreamMetadata]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(a_stream, b_stream)| {
            a_stream.stream_id == b_stream.stream_id
                && video_audio_track_eq(&a_stream.track, &b_stream.track)
        })
}

#[cfg(test)]
mod tests;
