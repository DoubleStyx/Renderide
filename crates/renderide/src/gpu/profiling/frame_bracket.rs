//! Frame-bracket GPU timing: real `TIMESTAMP_QUERY` writes that surround a tracked submit's command
//! buffers, giving the debug HUD a `gpu_frame_ms` value drawn from the GPU's own clock rather
//! than from `Queue::on_submitted_work_done` callback latency.
//!
//! # Lifecycle
//!
//! 1. Main thread: [`FrameBracket::open_session`] returns a [`FrameBracketSession`] when the
//!    adapter advertises both [`wgpu::Features::TIMESTAMP_QUERY`] and
//!    [`wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS`]. Returns [`None`] otherwise; callers
//!    fall back to the existing callback-latency path.
//! 2. Main thread: [`FrameBracketSession::begin_command_buffer`] / `end_command_buffer` produce
//!    two short [`wgpu::CommandBuffer`]s that bracket tracked submit work. The begin CB writes
//!    timestamp 0; the end CB writes timestamp 1, resolves both into a GPU-side buffer, and
//!    copies that into a CPU-mappable readback buffer.
//! 3. Main thread folds those CBs into the [`crate::gpu::driver_thread::SubmitBatch`] passed to
//!    the driver thread. The session is converted into a [`FrameBracketReadback`].
//! 4. Driver thread (or any later poll site): once the submit's GPU work completes, the
//!    readback callback fires with `(end_ticks - begin_ticks) * timestamp_period / 1e6` as the
//!    `gpu_frame_ms` value. The callback owns all its [`wgpu::Buffer`] / [`wgpu::QuerySet`]
//!    references so the GPU resources stay alive until the read completes.
//!
//! Each tracked submit uses fresh resources rather than a ring of pre-allocated slots -- the per-frame
//! cost is one 2-entry timestamp query set plus two 16-byte buffers, which is negligible
//! compared to the rest of the renderer's per-frame allocation.

use std::sync::Arc;

use super::super::sync::mapped_buffer_health::GpuMappedBufferHealth;

/// Number of bytes a 2-entry `Timestamp` query set resolves into (`u64 x 2`).
const TIMESTAMP_PAIR_BYTES: u64 = 16;

/// Factory for per-submit frame-bracket sessions.
///
/// Cheap to construct and to clone the held [`Arc`] handles. Held by [`super::GpuContext`]'s
/// submission state.
pub struct FrameBracket {
    /// Logical device used to create per-session query sets and buffers.
    device: Arc<wgpu::Device>,
    /// Queue used to read [`wgpu::Queue::get_timestamp_period`] when finishing a session.
    queue: Arc<wgpu::Queue>,
    /// Shared mapped-buffer invalidation generation for stale readback suppression.
    mapped_buffer_health: Arc<GpuMappedBufferHealth>,
    /// Whether the adapter's feature set permits encoder-level `write_timestamp` calls.
    enabled: bool,
}

impl FrameBracket {
    /// Builds a bracket factory bound to `device` / `queue`.
    ///
    /// The factory is "enabled" only when the device features include both
    /// [`wgpu::Features::TIMESTAMP_QUERY`] and
    /// [`wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS`]. When either is missing,
    /// [`Self::open_session`] returns [`None`] and the HUD falls back to relabeling the GPU row
    /// as "GPU latency" (callback-fire wall-clock, not real compute time).
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        mapped_buffer_health: Arc<GpuMappedBufferHealth>,
    ) -> Self {
        let features = device.features();
        let enabled = features.contains(wgpu::Features::TIMESTAMP_QUERY)
            && features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
        Self {
            device,
            queue,
            mapped_buffer_health,
            enabled,
        }
    }

    /// Allocates a fresh per-submit session, or [`None`] when the adapter does not support the
    /// required timestamp features.
    ///
    /// Each session carries its own query set / resolve buffer / readback buffer; resources are
    /// dropped when the readback callback completes, so there is no slot bookkeeping to manage.
    pub fn open_session(&self) -> Option<FrameBracketSession> {
        if !self.enabled {
            return None;
        }
        let query_set = self.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("frame_bracket_timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let resolve_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame_bracket_resolve"),
            size: TIMESTAMP_PAIR_BYTES,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        crate::profiling::note_resource_churn!(Buffer, "gpu::frame_bracket_resolve_buffer");
        let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame_bracket_readback"),
            size: TIMESTAMP_PAIR_BYTES,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        crate::profiling::note_resource_churn!(Buffer, "gpu::frame_bracket_readback_buffer");
        Some(FrameBracketSession {
            device: Arc::clone(&self.device),
            queue: Arc::clone(&self.queue),
            mapped_buffer_health: Arc::clone(&self.mapped_buffer_health),
            mapped_buffer_generation: self.mapped_buffer_health.generation(),
            query_set,
            resolve_buffer,
            readback_buffer,
        })
    }
}

/// Per-submit state used by the main thread to wrap tracked command buffers.
///
/// Produced by [`FrameBracket::open_session`]; consumed by [`Self::into_readback`] once the
/// begin / end command buffers have been folded into the submit batch.
pub struct FrameBracketSession {
    /// Logical device, retained so the begin / end encoders can be created on the main thread.
    device: Arc<wgpu::Device>,
    /// Queue, retained so [`Self::into_readback`] can capture `get_timestamp_period`.
    queue: Arc<wgpu::Queue>,
    /// Shared mapped-buffer invalidation generation for stale readback suppression.
    mapped_buffer_health: Arc<GpuMappedBufferHealth>,
    /// Invalidation generation captured when this readback's buffers were created.
    mapped_buffer_generation: u64,
    /// Query set written into by the begin / end command buffers.
    query_set: wgpu::QuerySet,
    /// GPU-side resolve target for the query pair.
    resolve_buffer: wgpu::Buffer,
    /// CPU-mappable readback target the driver thread polls for completed timestamps.
    readback_buffer: wgpu::Buffer,
}

impl FrameBracketSession {
    /// Builds the command buffer that opens the bracket -- writes timestamp index 0.
    ///
    /// Submit this **before** any other tracked command buffer in the submit so its timestamp
    /// reflects the GPU clock right before the renderer's real work begins.
    pub fn begin_command_buffer(&self) -> wgpu::CommandBuffer {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame_bracket_begin"),
            });
        enc.write_timestamp(&self.query_set, 0);
        {
            profiling::scope!("CommandEncoder::finish::frame_bracket_begin");
            enc.finish()
        }
    }

    /// Builds the command buffer that closes the bracket -- writes timestamp 1, resolves both
    /// timestamps into the resolve buffer, and copies the result into the mappable readback.
    ///
    /// Submit this **after** every other tracked command buffer in the submit so its timestamp
    /// reflects the GPU clock right after the renderer's real work completes.
    pub fn end_command_buffer(&self) -> wgpu::CommandBuffer {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame_bracket_end"),
            });
        enc.write_timestamp(&self.query_set, 1);
        enc.resolve_query_set(&self.query_set, 0..2, &self.resolve_buffer, 0);
        enc.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.readback_buffer,
            0,
            TIMESTAMP_PAIR_BYTES,
        );
        {
            profiling::scope!("CommandEncoder::finish::frame_bracket_end");
            enc.finish()
        }
    }

    /// Consumes the session and returns the readback handle the driver thread polls after submit.
    pub fn into_readback(self) -> FrameBracketReadback {
        let timestamp_period = self.queue.get_timestamp_period();
        FrameBracketReadback {
            device: self.device,
            mapped_buffer_health: self.mapped_buffer_health,
            mapped_buffer_generation: self.mapped_buffer_generation,
            readback_buffer: self.readback_buffer,
            query_set: self.query_set,
            resolve_buffer: self.resolve_buffer,
            timestamp_period,
        }
    }
}

/// GPU resources kept alive while a submit's frame-bracket timestamps are flying.
///
/// Held by the closure passed to [`Self::schedule_readback`]; dropped when the closure runs to
/// completion (or is dropped without running, e.g. on shutdown).
pub struct FrameBracketReadback {
    /// Logical device used to locally scope validation errors from best-effort readback mapping.
    device: Arc<wgpu::Device>,
    /// Shared mapped-buffer invalidation generation for stale readback suppression.
    mapped_buffer_health: Arc<GpuMappedBufferHealth>,
    /// Invalidation generation captured when this readback's buffers were created.
    mapped_buffer_generation: u64,
    /// CPU-mappable buffer the resolve copies finish into.
    readback_buffer: wgpu::Buffer,
    /// Held until readback completes so the underlying query set is not dropped early.
    query_set: wgpu::QuerySet,
    /// Held until readback completes for the same reason.
    resolve_buffer: wgpu::Buffer,
    /// Captured at session-finish time; multiplies u64 ticks into nanoseconds.
    timestamp_period: f32,
}

impl FrameBracketReadback {
    /// Registers a `map_async` callback on the readback buffer.
    ///
    /// `on_gpu_ms` is invoked exactly once with `Some(gpu_frame_ms)` on success, or [`None`] if
    /// the map fails (e.g. device loss). Successful maps are unmapped after bytes are read; failed
    /// maps are dropped without unmapping because the buffer never entered the mapped state.
    ///
    /// The callback fires on whatever thread next polls the device after the GPU has finished
    /// the submit; in practice that is the main thread, since the renderer drives
    /// [`wgpu::Device::poll`] from its frame loop.
    pub fn schedule_readback<F>(self, on_gpu_ms: F)
    where
        F: FnOnce(Option<f64>) + Send + 'static,
    {
        let Self {
            device,
            mapped_buffer_health,
            mapped_buffer_generation,
            readback_buffer,
            query_set,
            resolve_buffer,
            timestamp_period,
        } = self;
        let buffer_for_callback = readback_buffer.clone();
        let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        readback_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _keep_query_set_alive = query_set;
                let _keep_resolve_buffer_alive = resolve_buffer;
                let gpu_ms = match result {
                    Ok(()) => {
                        if frame_bracket_readback_generation_current(
                            mapped_buffer_health.as_ref(),
                            mapped_buffer_generation,
                        ) {
                            let gpu_ms = read_gpu_ms(&buffer_for_callback, timestamp_period);
                            buffer_for_callback.unmap();
                            gpu_ms
                        } else {
                            None
                        }
                    }
                    Err(_) => None,
                };
                on_gpu_ms(gpu_ms);
            });
        if let Some(error) = pollster::block_on(error_scope.pop()) {
            logger::debug!("frame bracket readback map validation suppressed: {error}");
        }
    }
}

/// Reads the two `u64` timestamps from `readback`, returning the elapsed milliseconds.
fn read_gpu_ms(readback: &wgpu::Buffer, timestamp_period: f32) -> Option<f64> {
    let view = readback.slice(..).get_mapped_range();
    let gpu_ms = timestamp_pair_bytes_to_ms(&view, timestamp_period);
    drop(view);
    gpu_ms
}

/// Returns whether a mapped readback callback still belongs to the active buffer generation.
fn frame_bracket_readback_generation_current(
    health: &GpuMappedBufferHealth,
    expected_generation: u64,
) -> bool {
    health.generation() == expected_generation
}

fn timestamp_pair_bytes_to_ms(bytes: &[u8], timestamp_period: f32) -> Option<f64> {
    if bytes.len() < TIMESTAMP_PAIR_BYTES as usize {
        return None;
    }
    let begin = u64::from_le_bytes(bytes[0..8].try_into().ok()?);
    let end = u64::from_le_bytes(bytes[8..16].try_into().ok()?);
    let ticks = end.saturating_sub(begin);
    let ns = (ticks as f64) * f64::from(timestamp_period);
    Some(ns / 1_000_000.0)
}

#[cfg(test)]
mod tests {
    use super::{frame_bracket_readback_generation_current, timestamp_pair_bytes_to_ms};
    use crate::gpu::sync::mapped_buffer_health::GpuMappedBufferHealth;

    #[test]
    fn timestamp_pair_bytes_convert_to_ms() {
        let bytes = timestamp_bytes(1_000, 3_500);

        let ms = timestamp_pair_bytes_to_ms(&bytes, 2.0).unwrap();

        assert_eq!(ms, 0.005);
    }

    #[test]
    fn timestamp_pair_bytes_reject_short_input() {
        let bytes = [0_u8; 15];

        assert_eq!(timestamp_pair_bytes_to_ms(&bytes, 1.0), None);
    }

    #[test]
    fn timestamp_pair_bytes_saturate_reversed_ticks() {
        let bytes = timestamp_bytes(3_500, 1_000);

        let ms = timestamp_pair_bytes_to_ms(&bytes, 2.0).unwrap();

        assert_eq!(ms, 0.0);
    }

    #[test]
    fn readback_generation_check_rejects_stale_callbacks() {
        let health = GpuMappedBufferHealth::new();
        let generation = health.generation();

        assert!(frame_bracket_readback_generation_current(
            &health, generation
        ));

        health.mark_invalid("test invalidation");

        assert!(!frame_bracket_readback_generation_current(
            &health, generation
        ));
    }

    fn timestamp_bytes(begin: u64, end: u64) -> [u8; 16] {
        let mut bytes = [0_u8; 16];
        bytes[0..8].copy_from_slice(&begin.to_le_bytes());
        bytes[8..16].copy_from_slice(&end.to_le_bytes());
        bytes
    }
}
