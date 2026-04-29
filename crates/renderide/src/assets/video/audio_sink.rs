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
    sink: AppSink,
    /// The asset id is kept solely for diagnostic log messages.
    asset_id: i32,
}

impl ResoniteAudioSink {
    /// Builds an [`AppSink`] capped to the requested sample rate.
    pub fn new(asset_id: i32, sample_rate: i32) -> Self {
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

        Self { sink, asset_id }
    }

    /// Returns a reference to the inner [`AppSink`] for use as a playbin property.
    pub fn appsink(&self) -> &AppSink {
        &self.sink
    }

    /// Connects a shared-memory publisher identified by `queue_name` and `queue_capacity`.
    ///
    /// Returns `false` when the queue cannot be opened; the sink is then left without a callback
    /// and no audio will be forwarded.
    pub fn attach_queue(
        &self,
        queue_name: &str,
        queue_capacity: i32,
    ) -> Result<(), ResoniteAudioSinkError> {
        let id = self.asset_id;

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

                    // Silent failure (caller decides how to handle backpressure)
                    let _ = publisher.try_enqueue(map.as_slice());

                    Ok(gstreamer::FlowSuccess::Ok)
                })
                .build(),
        );

        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum ResoniteAudioSinkError {
    #[error("invalid audio queue capacity: {0}")]
    InvalidQueueCapacity(i32),

    #[error("failed to build QueueOptions: {0}")]
    QueueOptions(String),

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
        assert_eq!(positive_queue_capacity(64).unwrap(), 64);
    }
}
