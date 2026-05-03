//! Frame-bracket GPU timing: real `TIMESTAMP_QUERY` writes that surround a tick's command
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
//!    two short [`wgpu::CommandBuffer`]s that bracket the tick's real work. The begin CB writes
//!    timestamp 0; the end CB writes timestamp 1, resolves both into a GPU-side buffer, and
//!    copies that into a CPU-mappable readback buffer.
//! 3. Main thread folds those CBs into the [`crate::gpu::driver_thread::SubmitBatch`] passed to
//!    the driver thread. The session is converted into a [`FrameBracketReadback`].
//! 4. Driver thread (or any later poll site): once the submit's GPU work completes, the
//!    readback callback fires with `(end_ticks - begin_ticks) * timestamp_period / 1e6` as the
//!    `gpu_frame_ms` value. The callback owns all its [`wgpu::Buffer`] / [`wgpu::QuerySet`]
//!    references so the GPU resources stay alive until the read completes.
//!
//! Each tick uses fresh resources rather than a ring of pre-allocated slots -- the per-frame
//! cost is one 2-entry timestamp query set plus two 16-byte buffers, which is negligible
//! compared to the rest of the renderer's per-frame allocation.

use std::sync::Arc;

/// Number of bytes a 2-entry `Timestamp` query set resolves into (`u64 x 2`).
const TIMESTAMP_PAIR_BYTES: u64 = 16;

/// Factory for per-tick frame-bracket sessions.
///
/// Cheap to construct and to clone the held [`Arc`] handles. Held by [`super::GpuContext`]'s
/// submission state.
pub struct FrameBracket {
    /// Logical device used to create per-session query sets and buffers.
    device: Arc<wgpu::Device>,
    /// Queue used to read [`wgpu::Queue::get_timestamp_period`] when finishing a session.
    queue: Arc<wgpu::Queue>,
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
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let features = device.features();
        let enabled = features.contains(wgpu::Features::TIMESTAMP_QUERY)
            && features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
        Self {
            device,
            queue,
            enabled,
        }
    }

    /// Allocates a fresh per-tick session, or [`None`] when the adapter does not support the
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
        let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame_bracket_readback"),
            size: TIMESTAMP_PAIR_BYTES,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Some(FrameBracketSession {
            device: Arc::clone(&self.device),
            queue: Arc::clone(&self.queue),
            query_set,
            resolve_buffer,
            readback_buffer,
        })
    }
}

/// Per-tick state used by the main thread to wrap a frame's command buffers.
///
/// Produced by [`FrameBracket::open_session`]; consumed by [`Self::into_readback`] once the
/// begin / end command buffers have been folded into the submit batch.
pub struct FrameBracketSession {
    /// Logical device, retained so the begin / end encoders can be created on the main thread.
    device: Arc<wgpu::Device>,
    /// Queue, retained so [`Self::into_readback`] can capture `get_timestamp_period`.
    queue: Arc<wgpu::Queue>,
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
    /// Submit this **before** any other tracked command buffer in the tick so its timestamp
    /// reflects the GPU clock right before the renderer's real work begins.
    pub fn begin_command_buffer(&self) -> wgpu::CommandBuffer {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame_bracket_begin"),
            });
        enc.write_timestamp(&self.query_set, 0);
        enc.finish()
    }

    /// Builds the command buffer that closes the bracket -- writes timestamp 1, resolves both
    /// timestamps into the resolve buffer, and copies the result into the mappable readback.
    ///
    /// Submit this **after** every other tracked command buffer in the tick so its timestamp
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
        enc.finish()
    }

    /// Consumes the session and returns the readback handle the driver thread polls after submit.
    pub fn into_readback(self) -> FrameBracketReadback {
        let timestamp_period = self.queue.get_timestamp_period();
        FrameBracketReadback {
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
    /// the map fails (e.g. device loss). After invocation, the buffer is unmapped and all GPU
    /// resources owned by this readback are released.
    ///
    /// The callback fires on whatever thread next polls the device after the GPU has finished
    /// the submit; in practice that is the main thread, since the renderer drives
    /// [`wgpu::Device::poll`] from its frame loop.
    pub fn schedule_readback<F>(self, on_gpu_ms: F)
    where
        F: FnOnce(Option<f64>) + Send + 'static,
    {
        let Self {
            readback_buffer,
            query_set,
            resolve_buffer,
            timestamp_period,
        } = self;
        let buffer_for_callback = readback_buffer.clone();
        readback_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _keep_query_set_alive = query_set;
                let _keep_resolve_buffer_alive = resolve_buffer;
                let gpu_ms = match result {
                    Ok(()) => read_gpu_ms(&buffer_for_callback, timestamp_period),
                    Err(_) => None,
                };
                buffer_for_callback.unmap();
                on_gpu_ms(gpu_ms);
            });
    }
}

/// Reads the two `u64` timestamps from `readback`, returning the elapsed milliseconds.
fn read_gpu_ms(readback: &wgpu::Buffer, timestamp_period: f32) -> Option<f64> {
    let view = readback.slice(..).get_mapped_range();
    if view.len() < TIMESTAMP_PAIR_BYTES as usize {
        return None;
    }
    let begin = u64::from_ne_bytes(view[0..8].try_into().ok()?);
    let end = u64::from_ne_bytes(view[8..16].try_into().ok()?);
    drop(view);
    let ticks = end.saturating_sub(begin);
    let ns = (ticks as f64) * f64::from(timestamp_period);
    Some(ns / 1_000_000.0)
}
