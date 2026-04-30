//! [`ResoniteAudioSink`] wraps a GStreamer [`AppSink`] configured for raw F32LE audio output
//! and handles attaching it to a shared-memory publisher.

use gstreamer_app::AppSink;
use interprocess::{QueueFactory, QueueOptions};
use thiserror::Error;

/// Wraps a GStreamer [`AppSink`] configured for interleaved F32LE stereo audio.
///
/// Call [`ResoniteAudioSink::attach_queue`] to connect a shared-memory publisher
/// and start forwarding decoded samples to the host.
pub struct ResoniteAudioSink {
    /// GStreamer app sink receiving decoded raw audio buffers.
    sink: AppSink,
}

impl ResoniteAudioSink {
    /// Builds an [`AppSink`] capped to the requested sample rate.
    pub fn new(sample_rate: i32) -> Self {
        let sink = AppSink::builder()
            .caps(
                &gstreamer::Caps::builder("audio/x-raw")
                    .field("format", "F32LE")
                    .field("rate", sample_rate)
                    .field("channels", 2i32)
                    .field("layout", "interleaved")
                    .build(),
            )
            .max_buffers(1)
            .drop(true)
            .sync(true)
            .build();

        Self { sink }
    }

    /// Returns a reference to the inner [`AppSink`] for use as a playbin property.
    pub fn appsink(&self) -> &AppSink {
        &self.sink
    }

    /// Connects a shared-memory publisher identified by `queue_name` and `queue_capacity`.
    ///
    /// When the queue cannot be opened, the sink is left without a new callback and no audio is
    /// forwarded to that queue.
    pub fn attach_queue(
        &self,
        queue_name: &str,
        queue_capacity: i32,
    ) -> Result<(), ResoniteAudioSinkError> {
        let queue_capacity = positive_queue_capacity(queue_capacity)?;

        let options = QueueOptions::new(queue_name, queue_capacity)
            .map_err(ResoniteAudioSinkError::QueueOptions)?;

        let mut publisher = QueueFactory::new()
            .create_publisher(options)
            .map_err(ResoniteAudioSinkError::Publisher)?;

        use gstreamer_app::AppSinkCallbacks;

        self.sink.set_callbacks(
            AppSinkCallbacks::builder()
                .new_sample(move |appsink| {
                    let Ok(sample) = appsink.pull_sample() else {
                        return Err(gstreamer::FlowError::Eos);
                    };

                    let Some(buffer) = sample.buffer() else {
                        return Ok(gstreamer::FlowSuccess::Ok);
                    };

                    let Ok(map) = buffer.map_readable() else {
                        return Ok(gstreamer::FlowSuccess::Ok);
                    };

                    // Backpressure is expected when the host falls behind; keep this path quiet.
                    let _ = publisher.try_enqueue(map.as_slice());

                    Ok(gstreamer::FlowSuccess::Ok)
                })
                .build(),
        );

        Ok(())
    }
}

/// Errors returned while attaching a host audio queue to [`ResoniteAudioSink`].
#[derive(Debug, Error)]
pub enum ResoniteAudioSinkError {
    /// Host provided a queue capacity that cannot hold any audio payloads.
    #[error("invalid audio queue capacity: {0}")]
    InvalidQueueCapacity(i32),

    /// Queue options could not be constructed for the host queue name.
    #[error("failed to build QueueOptions: {0}")]
    QueueOptions(String),

    /// Shared-memory publisher creation failed.
    #[error("failed to create publisher: {0}")]
    Publisher(#[source] interprocess::OpenError),
}

/// Converts host audio queue capacity to the signed queue API type.
pub fn positive_queue_capacity(queue_capacity: i32) -> Result<i64, ResoniteAudioSinkError> {
    if queue_capacity > 0 {
        Ok(queue_capacity as i64)
    } else {
        Err(ResoniteAudioSinkError::InvalidQueueCapacity(queue_capacity))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_audio_queue_capacity_is_rejected() {
        assert!(positive_queue_capacity(0).is_err());
        assert!(positive_queue_capacity(-1).is_err());
        assert!(matches!(positive_queue_capacity(64), Ok(64)));
    }
}
