//! The [`VideoPlayer`] holds the GStreamer pipeline and handles incoming updates from host.

use crate::assets::video::cpu_copy::CpuCopyVideoSink;
use crate::assets::video::WgpuGstVideoSink;
use crate::assets::AssetTransferQueue;
use glam::IVec2;
use gstreamer::prelude::{ElementExt, ElementExtManual};
use gstreamer_app::AppSink;
use interprocess::{QueueFactory, QueueOptions};
use renderide_shared::ipc::DualQueueIpc;
use renderide_shared::{
    RendererCommand, VideoAudioTrack, VideoTextureLoad, VideoTextureReady,
    VideoTextureStartAudioTrack, VideoTextureUpdate,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

/// Holds the GStreamer pipeline and handles incoming updates from host.
pub struct VideoPlayer {
    asset_id: i32,
    pipeline: gstreamer::Element,
    audio_sink: AppSink,
    video_sink: Box<dyn WgpuGstVideoSink + Send>,
    /// Stores the latest [`VideoTextureUpdate`] until it gets processed by the update thread.
    pending_update: Arc<Mutex<Option<VideoTextureUpdate>>>,
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

        if let Err(e) = gstreamer::init() {
            logger::error!("gstreamer init failed: {e}");
            return None;
        }

        let audio_sink = AppSink::builder()
            .caps(
                &gstreamer::Caps::builder("audio/x-raw")
                    .field("format", "F32LE")
                    .field("rate", l.audio_system_sample_rate)
                    .field("channels", 2i32)
                    .field("layout", "interleaved")
                    .build(),
            )
            .sync(true)
            .build();

        // for now there is only one video sink backend
        let video_sink = Box::new(CpuCopyVideoSink::new(id, device, queue));

        let uri = match l.source {
            Some(src) if src.contains("://") => src,
            // playbin needs the file:// scheme for accessing the local filesystem
            Some(src) => format!("file://{}", src),
            None => return None,
        };

        let pipeline = match gstreamer::ElementFactory::make("playbin")
            .property("uri", &uri)
            .property("audio-sink", &audio_sink)
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
            pending_update,
            shutdown,
        })
    }

    /// Handles [`VideoTextureStartAudioTrack`].
    /// Opens a shared memory queue to send audio back to host, and assigns the callback to the sink.
    pub fn handle_start_audio_track(&mut self, s: VideoTextureStartAudioTrack) {
        let id = self.asset_id;

        let queue_name = match s.queue_name {
            Some(name) => name,
            None => {
                // TODO: we still want to handle switching audio tracks here
                return;
            }
        };

        let options = match QueueOptions::new(&queue_name, s.queue_capacity as i64) {
            Ok(o) => o,
            Err(e) => {
                logger::error!("video texture {}: failed to build QueueOptions: {e}", id);
                return;
            }
        };

        let mut publisher = match QueueFactory::new().create_publisher(options) {
            Ok(p) => p,
            Err(e) => {
                logger::error!("video texture {}: failed to create publisher: {e}", id);
                return;
            }
        };

        use gstreamer_app::AppSinkCallbacks;
        self.audio_sink.set_callbacks(
            AppSinkCallbacks::builder()
                .new_sample(move |appsink| {
                    let sample = match appsink.pull_sample() {
                        Ok(s) => s,
                        Err(_) => return Err(gstreamer::FlowError::Eos),
                    };
                    let buffer = match sample.buffer() {
                        Some(b) => b,
                        None => return Ok(gstreamer::FlowSuccess::Ok),
                    };
                    let map = match buffer.map_readable() {
                        Ok(m) => m,
                        Err(_) => return Ok(gstreamer::FlowSuccess::Ok),
                    };
                    let _ = publisher.try_enqueue(map.as_slice());
                    Ok(gstreamer::FlowSuccess::Ok)
                })
                .build(),
        );
    }

    /// Schedules a video player state update from [`VideoTextureUpdate`].
    pub fn handle_update(&mut self, u: VideoTextureUpdate) {
        *self.pending_update.lock().unwrap() = Some(u);
    }

    /// Handles texture changes from the sink and running the GStreamer event loop,
    /// as well as sending of [`VideoTextureReady`].
    pub fn process_events(
        &mut self,
        queue: &mut AssetTransferQueue,
        ipc: &mut Option<&mut DualQueueIpc>,
    ) {
        let Some(bus) = self.pipeline.bus() else {
            return;
        };

        // forward any texture the sink created since last frame
        let id = self.asset_id;
        if let Some((view, w, h, bytes)) = self.video_sink.poll_texture_change() {
            if let Some(gpu_tex) = queue.video_texture_pool.get_mut(id) {
                gpu_tex.set_view(view, w, h, bytes);
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
                thread::sleep(std::time::Duration::from_millis(16));

                let update = match pending_update.lock().unwrap().take() {
                    Some(u) => u,
                    None => continue,
                };

                let target_state = if update.play {
                    gstreamer::State::Playing
                } else {
                    gstreamer::State::Paused
                };

                let _ = pipeline.set_state(target_state);

                let mut query = gstreamer::query::Position::new(gstreamer::Format::Time);
                if pipeline.query(&mut query) {
                    if let gstreamer::GenericFormattedValue::Time(Some(current)) = query.result() {
                        let current_secs = current.nseconds() as f64 / 1_000_000_000.0;
                        let drift = (current_secs - update.position).abs();
                        let max_error = if update.play { 1.0 } else { 0.01 };

                        if drift > max_error {
                            let ns = (update.position * 1_000_000_000.0) as u64;
                            let _ = pipeline.seek_simple(
                                gstreamer::SeekFlags::FLUSH | gstreamer::SeekFlags::KEY_UNIT,
                                gstreamer::ClockTime::from_nseconds(ns),
                            );
                        }
                    }
                }
            }

            // handle this here because apparently gstreamer can just freeze if the video is evil
            // it's still not ideal, because it keeps on running, but at least the renderer won't die
            if let Err(e) = pipeline.set_state(gstreamer::State::Null) {
                logger::error!("failed to set pipeline to Null on shutdown: {e}");
            }
        });
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
            // TODO: retrieve audio tracks from gstreamer
            audio_tracks: vec![VideoAudioTrack {
                index: 0,
                channel_count: 2,
                sample_rate: 48000,
                name: Some("test".into()),
                language_code: None,
            }],
        }));
    }
}

impl Drop for VideoPlayer {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }
}
