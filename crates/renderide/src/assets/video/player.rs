//! The [`VideoPlayer`] holds the GStreamer pipeline and handles incoming updates from host.

use super::audio_sink::ResoniteAudioSink;
use super::cpu_copy::CpuCopyVideoSink;
use super::WgpuGstVideoSink;
use crate::assets::AssetTransferQueue;
use glam::IVec2;
use gstreamer::prelude::{ElementExt, ElementExtManual};

use renderide_shared::ipc::DualQueueIpc;
use renderide_shared::{
    RendererCommand, VideoAudioTrack, VideoTextureClockErrorState, VideoTextureLoad,
    VideoTextureReady, VideoTextureStartAudioTrack, VideoTextureUpdate,
};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

/// Fallback audio rate used when the host sends an invalid sample rate.
pub(self) const DEFAULT_AUDIO_SAMPLE_RATE: i32 = 48_000;

/// Poll interval for applying host playback updates to GStreamer.
const UPDATE_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(16);

/// Maximum tolerated seek drift while video is actively playing.
const PLAYING_SEEK_DRIFT_SECONDS: f64 = 1.0;

/// Maximum tolerated seek drift while video is paused.
const PAUSED_SEEK_DRIFT_SECONDS: f64 = 0.01;

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
    /// Stores the latest [`VideoTextureUpdate`] until it gets processed by the update thread.
    pending_update: Arc<Mutex<Option<VideoTextureUpdate>>>,
    /// Most recently received host update, retained even after the worker thread applies it so
    /// [`Self::sample_clock_error`] can keep reporting drift each frame against
    /// [`VideoTextureUpdate::decoded_time`] until the host sends another update. Mirrors
    /// `_update` in `UnityVideoTextureBehaviour`, which is re-armed each frame with the same
    /// last-received instance.
    last_update: Arc<Mutex<Option<VideoTextureUpdate>>>,
    /// Shared shutdown flag checked by the update thread.
    shutdown: Arc<AtomicBool>,
}

impl VideoPlayer {
    /// Creates a new [`VideoPlayer`] using [`VideoTextureLoad`].
    pub fn new(
        l: VideoTextureLoad,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
    ) -> Option<Self> {
        let id = l.asset_id;
        let audio_sample_rate = normalized_audio_sample_rate(l.audio_system_sample_rate);

        if let Err(e) = gstreamer::init() {
            logger::error!("gstreamer init failed: {e}");
            return None;
        }

        let audio_sink = ResoniteAudioSink::new(id, audio_sample_rate);

        // GStreamer-backed video textures currently use the CPU-copy sink on all platforms.
        let video_sink = Box::new(CpuCopyVideoSink::new(id, device, queue));

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

        let pipeline = match gstreamer::ElementFactory::make("playbin")
            .property("uri", &uri)
            .property("audio-sink", audio_sink.appsink())
            .property("video-sink", video_sink.appsink())
            .build()
        {
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

        Self::spawn_update_thread(
            pipeline.clone(),
            Arc::clone(&pending_update),
            Arc::clone(&shutdown),
        );

        Some(Self {
            asset_id: id,
            pipeline,
            audio_sink,
            video_sink,
            audio_sample_rate,
            pending_update,
            last_update,
            shutdown,
        })
    }

    /// Handles [`VideoTextureStartAudioTrack`].
    /// Opens a shared memory queue to send audio back to host, and assigns the callback to the sink.
    pub fn handle_start_audio_track(&mut self, s: VideoTextureStartAudioTrack) {
        let id = self.asset_id;
        if s.audio_track_index != 0 {
            logger::warn!(
                "video texture {id}: unsupported audio track index {}",
                s.audio_track_index
            );
            return;
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
    /// host sends a fresh update, matching Renderite.Unity's `_update` reuse behaviour.
    pub fn handle_update(&mut self, u: VideoTextureUpdate) {
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
        queue: &mut AssetTransferQueue,
        ipc: &mut Option<&mut DualQueueIpc>,
    ) {
        profiling::scope!("video::process_events");
        let Some(bus) = self.pipeline.bus() else {
            return;
        };

        // Forward any texture the sink created since last frame. Ensure the pool entry exists here
        // so a frame cannot be lost if video data arrives before sampler properties.
        let id = self.asset_id;
        if let Some((view, w, h, bytes)) = self.video_sink.poll_texture_change() {
            let props = queue.video_texture_properties_or_default(id);
            if let Some(gpu_tex) = queue.ensure_video_texture_with_props(&props) {
                gpu_tex.set_view(view, w, h, bytes);
            } else {
                logger::warn!("video texture {id}: GPU placeholder unavailable for decoded frame");
            }
        }

        while let Some(msg) = bus.timed_pop(gstreamer::ClockTime::ZERO) {
            match msg.view() {
                gstreamer::MessageView::AsyncDone(_) => {
                    let size = self.video_sink.size();
                    let length = self.get_duration();
                    logger::info!(
                        "video texture {}: loaded: size={:?}, length={}",
                        id,
                        size,
                        length
                    );

                    self.send_ready(
                        ipc,
                        length,
                        size.unwrap_or_default(),
                        Some(format!("GStreamer ({})", self.video_sink.name())),
                    );
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

    fn get_duration(&self) -> f64 {
        let mut query = gstreamer::query::Duration::new(gstreamer::Format::Time);
        if !self.pipeline.query(&mut query) {
            return -1.0;
        }
        match query.result() {
            gstreamer::GenericFormattedValue::Time(Some(t)) if t != gstreamer::ClockTime::ZERO => {
                t.nseconds() as f64 / 1_000_000_000.0
            }
            _ => -1.0,
        }
    }

    fn spawn_update_thread(
        pipeline: gstreamer::Element,
        pending_update: Arc<Mutex<Option<VideoTextureUpdate>>>,
        shutdown: Arc<AtomicBool>,
    ) {
        thread::spawn(move || {
            while !shutdown.load(Ordering::Relaxed) {
                thread::sleep(UPDATE_POLL_INTERVAL);

                let update = match pending_update.lock() {
                    Ok(mut pending_update) => match pending_update.take() {
                        Some(update) => update,
                        None => continue,
                    },
                    Err(_) => {
                        logger::warn!("video texture update thread: update lock poisoned");
                        break;
                    }
                };
                apply_update_to_pipeline(&pipeline, &update);
            }

            // GStreamer shutdown can block on damaged media; this work stays off the render thread.
            if let Err(e) = pipeline.set_state(gstreamer::State::Null) {
                logger::error!("failed to set pipeline to Null on shutdown: {e}");
            }
        });
    }

    /// Samples this player's clock error against the host's most recently received playback request.
    ///
    /// Mirrors `UnityVideoTextureBehaviour`: the error is `pipeline_position − adjusted_host_position`,
    /// where the adjusted position advances unconditionally at real-time from
    /// [`VideoTextureUpdate::decoded_time`] (set by the IPC unpack at receive time). Returns `None`
    /// until at least one update has arrived or when the pipeline position cannot be queried.
    pub fn sample_clock_error(&self) -> Option<VideoTextureClockErrorState> {
        let update = match self.last_update.lock() {
            Ok(slot) => slot.clone()?,
            Err(_) => {
                logger::warn!(
                    "video texture {}: last-update lock poisoned; skipping clock error sample",
                    self.asset_id
                );
                return None;
            }
        };
        let current = query_position_seconds(&self.pipeline)?;
        let now_nanos = unix_nanos_now();
        let adjusted = adjusted_host_position(&update, now_nanos);
        Some(VideoTextureClockErrorState {
            asset_id: self.asset_id,
            current_clock_error: (current - adjusted) as f32,
        })
    }

    fn send_ready(
        &self,
        ipc: &mut Option<&mut DualQueueIpc>,
        length: f64,
        size: IVec2,
        playback_engine: Option<String>,
    ) {
        let Some(ipc) = ipc else {
            return;
        };

        ipc.send_background(RendererCommand::VideoTextureReady(VideoTextureReady {
            length,
            size,
            has_alpha: false,
            asset_id: self.asset_id,
            instance_changed: true,
            playback_engine,
            audio_tracks: vec![default_audio_track(self.audio_sample_rate)],
        }));
    }
}

impl Drop for VideoPlayer {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
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

/// Returns `true` when `source` already has a URI scheme.
fn is_uri_source(source: &str) -> bool {
    source.contains("://")
}

/// Converts a host source string into a playbin URI.
fn source_uri(source: Option<&str>) -> Result<Option<String>, gstreamer::glib::Error> {
    let Some(source) = source else {
        return Ok(None);
    };
    if is_uri_source(source) {
        return Ok(Some(source.to_owned()));
    }
    gstreamer::glib::filename_to_uri(local_source_path(source), None)
        .map(|uri| Some(uri.to_string()))
}

/// Returns an absolute local path for GLib URI conversion when possible.
fn local_source_path(source: &str) -> PathBuf {
    let path = Path::new(source);
    if path.is_absolute() {
        return path.to_path_buf();
    }
    match std::env::current_dir() {
        Ok(cwd) => cwd.join(path),
        Err(_) => path.to_path_buf(),
    }
}

/// Returns the pipeline state implied by the host update.
fn target_state_for_update(update: &VideoTextureUpdate) -> gstreamer::State {
    if update.play {
        gstreamer::State::Playing
    } else {
        gstreamer::State::Paused
    }
}

/// Returns how far the current playback position may drift before seeking.
fn max_seek_drift_seconds(update: &VideoTextureUpdate) -> f64 {
    if update.play {
        PLAYING_SEEK_DRIFT_SECONDS
    } else {
        PAUSED_SEEK_DRIFT_SECONDS
    }
}

/// Returns `true` when GStreamer should seek to the host clock position.
fn should_seek_to_host_position(current_seconds: f64, update: &VideoTextureUpdate) -> bool {
    (current_seconds - update.position).abs() > max_seek_drift_seconds(update)
}

/// Converts host seconds to a bounded GStreamer clock time.
fn clock_time_from_seconds(seconds: f64) -> gstreamer::ClockTime {
    if !seconds.is_finite() || seconds <= 0.0 {
        return gstreamer::ClockTime::ZERO;
    }
    let nanos = (seconds * 1_000_000_000.0).min(u64::MAX as f64) as u64;
    gstreamer::ClockTime::from_nseconds(nanos)
}

/// Returns the host-expected playback position right now, given the last received update.
///
/// Faithful port of `VideoTextureUpdate.AdjustedPosition` in `Renderite.Shared`:
/// `position + (now - decoded_time).total_seconds()`, with no play-state guard. Even while paused,
/// the host expects the renderer to keep refreshing this value off the most recent `decoded_time`,
/// which the IPC unpack stamps at receive time. `now_nanos` and `decoded_time` are both nanoseconds
/// since the UNIX epoch.
fn adjusted_host_position(update: &VideoTextureUpdate, now_nanos: i128) -> f64 {
    let elapsed_nanos = now_nanos - update.decoded_time;
    update.position + (elapsed_nanos as f64) / 1_000_000_000.0
}

/// Returns the current wall-clock time as nanoseconds since the UNIX epoch, matching the encoding
/// the IPC unpack uses for [`VideoTextureUpdate::decoded_time`].
fn unix_nanos_now() -> i128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as i128)
        .unwrap_or(0)
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

/// Builds the fallback track descriptor sent to the host until GStreamer track metadata is wired.
fn default_audio_track(sample_rate: i32) -> VideoAudioTrack {
    VideoAudioTrack {
        index: 0,
        channel_count: 2,
        sample_rate: normalized_audio_sample_rate(sample_rate),
        name: None,
        language_code: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn update(position: f64, play: bool) -> VideoTextureUpdate {
        VideoTextureUpdate {
            position,
            play,
            ..VideoTextureUpdate::default()
        }
    }

    #[test]
    fn invalid_audio_sample_rate_uses_default() {
        assert_eq!(normalized_audio_sample_rate(0), DEFAULT_AUDIO_SAMPLE_RATE);
        assert_eq!(
            normalized_audio_sample_rate(-44_100),
            DEFAULT_AUDIO_SAMPLE_RATE
        );
        assert_eq!(normalized_audio_sample_rate(44_100), 44_100);
    }

    #[test]
    fn seek_threshold_is_tighter_when_paused() {
        assert!(!should_seek_to_host_position(10.5, &update(10.0, true)));
        assert!(should_seek_to_host_position(10.5, &update(10.0, false)));
    }

    #[test]
    fn clock_time_from_seconds_clamps_invalid_values_to_zero() {
        assert_eq!(
            clock_time_from_seconds(f64::NAN),
            gstreamer::ClockTime::ZERO
        );
        assert_eq!(clock_time_from_seconds(-1.0), gstreamer::ClockTime::ZERO);
        assert_eq!(
            clock_time_from_seconds(1.25),
            gstreamer::ClockTime::from_nseconds(1_250_000_000)
        );
    }

    #[test]
    fn uri_sources_pass_through_without_file_conversion() {
        assert!(is_uri_source("https://example.invalid/video.mp4"));
        assert!(is_uri_source("file:///tmp/video.mp4"));
        assert!(!is_uri_source("/tmp/video.mp4"));
    }

    #[test]
    fn relative_local_sources_are_made_absolute_before_uri_conversion() {
        let path = local_source_path("video.mp4");
        assert!(path.is_absolute());
        assert!(path.ends_with("video.mp4"));
    }

    #[test]
    fn default_audio_track_uses_normalized_sample_rate() {
        let track = default_audio_track(-1);
        assert_eq!(track.sample_rate, DEFAULT_AUDIO_SAMPLE_RATE);
        assert_eq!(track.index, 0);
        assert_eq!(track.channel_count, 2);
        assert_eq!(track.name, None);
    }

    fn update_decoded_at(position: f64, play: bool, decoded_nanos: i128) -> VideoTextureUpdate {
        VideoTextureUpdate {
            position,
            play,
            decoded_time: decoded_nanos,
            ..VideoTextureUpdate::default()
        }
    }

    const HALF_SECOND_NS: i128 = 500_000_000;
    const ONE_SECOND_NS: i128 = 1_000_000_000;

    #[test]
    fn adjusted_host_position_advances_unconditionally_when_playing() {
        let u = update_decoded_at(10.0, true, 0);
        assert!((adjusted_host_position(&u, HALF_SECOND_NS) - 10.5).abs() < 1e-9);
    }

    #[test]
    fn adjusted_host_position_advances_even_when_paused() {
        // Mirrors C# `VideoTextureUpdate.AdjustedPosition`, which has no play-state guard. The
        // host re-sends paused updates frequently so elapsed-since-decoded stays bounded.
        let u = update_decoded_at(10.0, false, 0);
        assert!((adjusted_host_position(&u, HALF_SECOND_NS) - 10.5).abs() < 1e-9);
    }

    #[test]
    fn adjusted_host_position_zero_elapsed_returns_position() {
        let u = update_decoded_at(7.25, true, ONE_SECOND_NS);
        assert_eq!(adjusted_host_position(&u, ONE_SECOND_NS), 7.25);
    }

    #[test]
    fn adjusted_host_position_handles_negative_elapsed() {
        // If wall-clock goes backwards, elapsed becomes negative and the adjusted position retreats,
        // matching the C# `(DateTime.UtcNow - decodedTime).TotalSeconds` literal port.
        let u = update_decoded_at(4.0, true, ONE_SECOND_NS);
        let earlier = ONE_SECOND_NS - HALF_SECOND_NS;
        assert!((adjusted_host_position(&u, earlier) - 3.5).abs() < 1e-9);
    }
}
