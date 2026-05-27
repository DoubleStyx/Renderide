//! Frame timing for the debug HUD: CPU per-frame work, GPU per-frame work, and the wall-clock
//! roundtrip between consecutive frame starts.
//!
//! Three numbers are tracked, each chosen so its label matches what the user expects from a
//! mainstream profiler overlay:
//!
//! - **CPU frame ms** -- wall-clock between [`Self::begin_frame`] and the matching
//!   [`Self::record_main_thread_cpu_end`] call from the runtime tick epilogue, both on the
//!   *main thread*. This is the time the renderer's main thread spends building the frame
//!   (asset integration, scene snapshot, draw collection, encoder recording, submit dispatch).
//!   It excludes FPS-gating sleeps, lockstep waits, event-loop idles, and explicit GPU/display
//!   pacing waits recorded through [`Self::record_excluded_wait`]. It does **not** cross the
//!   driver-thread queue boundary.
//! - **GPU frame ms** -- when the adapter advertises [`wgpu::Features::TIMESTAMP_QUERY`] +
//!   [`wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS`], the sum of every non-empty tracked
//!   submit's GPU timestamp bracket for the tick, computed from the GPU's own clock via
//!   [`wgpu::Queue::get_timestamp_period`]. When those features are unavailable, falls back to
//!   the wall-clock latency from the tick's first tracked `Queue::submit` return on the driver
//!   thread to the tick's last [`wgpu::Queue::on_submitted_work_done`] callback --
//!   [`GpuMsSource`] records which path produced the value so the HUD can relabel the row
//!   honestly.
//! - **Roundtrip ms** -- wall-clock between consecutive winit ticks. Tracked outside this
//!   struct ([`crate::diagnostics::FrameTimingHudSnapshot::wall_frame_time_ms`]).
//!
//! GPU values are populated **on the driver thread or on a `map_async` callback**, so they may
//! arrive after the originating winit tick has already ended its [`Self::end_frame`]. The HUD
//! reads [`Self::last_completed_paired_frame_ms`], which is updated only when a CPU value and a
//! GPU value have *both* arrived for the same frame generation -- that way the two numbers
//! shown to the user always belong to the same frame, so the relationship
//! `Frame >= max(CPU, GPU)` (in steady state) is observable on the overlay.
//!
//! `submit_latency_ms` is retained as a backend-cost measurement (frame_start ->
//! `Queue::submit` returning on the driver thread) but is not displayed in the default HUD; reach
//! for it when investigating driver-thread back-pressure.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use hashbrown::HashMap;

/// Maximum number of pending submit->completion pairs retained while waiting for a matching
/// GPU completion. The driver ring is bounded to a couple of frames; this generous cap
/// covers transient spikes where multiple frames' submits land before any of them complete.
const MAX_PENDING_PAIRS: usize = 16;

/// Origin of the most recently published `gpu_frame_ms` value.
///
/// The HUD uses this to label the GPU row: real timestamp queries get the standard "GPU"
/// label; the callback-latency fallback is relabelled to "GPU latency" so users do not mistake
/// it for actual compute time.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuMsSource {
    /// Computed from real GPU `WriteTimestamp` queries that bracket the tick's command buffers.
    FrameBracket,
    /// Wall-clock between driver-thread `Queue::submit` returning and
    /// `Queue::on_submitted_work_done` firing. Fallback used when the adapter lacks the
    /// timestamp-query features needed for [`Self::FrameBracket`].
    CallbackLatency,
}

/// Pending GPU timing aggregate for one renderer tick.
#[derive(Debug, Default)]
struct PendingFrameGpuTiming {
    /// Number of tracked submits expected for this tick, populated when the tick ends.
    expected_submits: Option<u32>,
    /// Number of submit completions recorded so far.
    completed_submits: u32,
    /// Sum of per-submit GPU timestamp brackets for timestamp-backed timing.
    frame_bracket_gpu_ms: f64,
    /// Earliest driver-thread submit-return instant for callback-latency fallback timing.
    first_submit_at: Option<Instant>,
    /// Latest callback completion instant for callback-latency fallback timing.
    last_callback_done_at: Option<Instant>,
    /// Completion source for the frame aggregate.
    source: Option<GpuMsSource>,
}

impl PendingFrameGpuTiming {
    /// Records that the tick expects exactly `expected` tracked submits.
    fn set_expected_submits(&mut self, expected: u32) {
        self.expected_submits = Some(expected);
    }

    /// Records the driver-thread submit-return instant for callback-latency fallback timing.
    fn record_submit_at(&mut self, submitted_at: Instant) {
        self.first_submit_at = Some(
            self.first_submit_at
                .map_or(submitted_at, |first| first.min(submitted_at)),
        );
    }

    /// Records one timestamp-backed submit completion.
    fn record_frame_bracket_done(&mut self, gpu_ms: f64) {
        self.completed_submits = self.completed_submits.saturating_add(1);
        self.frame_bracket_gpu_ms += gpu_ms;
        if self.source.is_none() {
            self.source = Some(GpuMsSource::FrameBracket);
        }
    }

    /// Records one callback-latency fallback submit completion.
    fn record_callback_done(&mut self, real_submit_at: Instant, completed_at: Instant) {
        self.completed_submits = self.completed_submits.saturating_add(1);
        self.record_submit_at(real_submit_at);
        self.last_callback_done_at = Some(
            self.last_callback_done_at
                .map_or(completed_at, |last| last.max(completed_at)),
        );
        self.source = Some(GpuMsSource::CallbackLatency);
    }

    /// Returns the completed frame GPU duration and source once every tracked submit has finished.
    fn completed_gpu_ms(&self) -> Option<(f64, GpuMsSource)> {
        let expected = self.expected_submits?;
        if self.completed_submits < expected {
            return None;
        }
        match self.source? {
            GpuMsSource::FrameBracket => {
                Some((self.frame_bracket_gpu_ms, GpuMsSource::FrameBracket))
            }
            GpuMsSource::CallbackLatency => {
                let first = self.first_submit_at?;
                let last = self.last_callback_done_at?;
                Some((
                    last.saturating_duration_since(first).as_secs_f64() * 1000.0,
                    GpuMsSource::CallbackLatency,
                ))
            }
        }
    }
}

/// Per-tick state for AAA-style CPU/GPU frame metrics (see [`GpuContext`](super::GpuContext)).
#[derive(Debug, Default)]
pub struct FrameCpuGpuTiming {
    /// Monotonic id; callbacks ignore stale generations after a new [`Self::begin_frame`].
    generation: u64,
    /// Start of the winit app-driver frame tick.
    frame_start: Option<Instant>,
    /// Main-thread waits that pace the tick on GPU, display, or compositor readiness rather than
    /// active renderer CPU work.
    excluded_wait: Duration,
    /// Number of tracked submits this tick (1-based).
    submit_seq: u32,
    /// Set in [`Self::end_frame`] to the last submit index for this tick.
    finalized_seq: Option<u32>,
    /// Driver-thread post-submit instants for this tick, keyed by submit index. Used to
    /// compute the optional `submit_latency_ms` diagnostic, not the HUD's CPU value.
    pending_real_submit_by_seq: HashMap<u32, Instant>,
    /// GPU timing aggregates keyed by frame generation. Kept across `begin_frame` so late
    /// timestamp readbacks can still publish the completed frame value.
    pending_gpu_by_generation: HashMap<u64, PendingFrameGpuTiming>,
    /// CPU ms keyed by `(generation, final_submit_seq)` for frames whose
    /// `record_main_thread_cpu_end` has already published the main-thread tick duration but whose
    /// GPU aggregate has not yet
    /// arrived. Kept ordered by insertion so the oldest entry is evicted first when
    /// [`MAX_PENDING_PAIRS`] is exceeded.
    pending_paired_cpu_ms: VecDeque<((u64, u32), f64)>,
    /// CPU ms (frame_start -> main-thread tick end) for the current tick once
    /// [`Self::record_main_thread_cpu_end`] fires.
    pub(crate) cpu_frame_ms: Option<f64>,
    /// Driver-thread submit-return latency for the current tick when the driver thread reported
    /// the last submit before [`Self::end_frame`] picked it up. Diagnostic, not displayed by
    /// default in the HUD.
    pub(crate) submit_latency_ms: Option<f64>,
    /// GPU ms for the current tick once every tracked submit's GPU completion has arrived.
    pub(crate) gpu_frame_ms: Option<f64>,
    /// Most recent `(cpu_ms, gpu_ms)` pair where both values describe the same completed frame.
    /// The HUD uses this so its CPU and GPU columns always belong to the same frame.
    /// Survives [`Self::begin_frame`] so the overlay never goes blank.
    pub(crate) last_completed_paired_frame_ms: Option<(f64, f64)>,
    /// Origin of the most recent `gpu_frame_ms` value, surfaced to the HUD label.
    pub(crate) last_gpu_source: Option<GpuMsSource>,
    /// Most recent completed whole-frame GPU ms. Used by
    /// [`crate::gpu::GpuContext::last_completed_gpu_render_time_seconds`] for the IPC
    /// `PerformanceState::render_time` field consumed by FrooxEngine.
    pub(crate) last_completed_gpu_frame_ms: Option<f64>,
}

/// Identifying info for one tracked submit, attached to a driver-thread batch.
///
/// The driver thread uses the embedded [`FrameCpuGpuTimingHandle`] to record the post-submit
/// instant (for diagnostics) and to publish GPU ms back to the main thread once the
/// frame-bracket readback or completion callback fires.
#[derive(Clone)]
pub struct FrameTimingTrack {
    /// Shared handle to the [`FrameCpuGpuTiming`] state.
    pub handle: FrameCpuGpuTimingHandle,
    /// Generation captured at [`FrameCpuGpuTiming::on_before_tracked_submit`] time.
    pub generation: u64,
    /// 1-based submit index within the originating tick.
    pub seq: u32,
    /// Winit tick start instant, used to compute `submit_latency_ms` once the real submit returns.
    pub frame_start: Instant,
}

impl FrameCpuGpuTiming {
    /// Starts tracking for a new winit tick; clears prior tick metrics.
    pub fn begin_frame(&mut self, frame_start: Instant) {
        self.generation = self.generation.wrapping_add(1);
        self.frame_start = Some(frame_start);
        self.excluded_wait = Duration::ZERO;
        self.submit_seq = 0;
        self.finalized_seq = None;
        self.pending_real_submit_by_seq.clear();
        let oldest_retained_generation = self.generation.saturating_sub(MAX_PENDING_PAIRS as u64);
        self.pending_gpu_by_generation
            .retain(|generation, _| *generation >= oldest_retained_generation);
        self.cpu_frame_ms = None;
        self.submit_latency_ms = None;
        self.gpu_frame_ms = None;
        // Intentionally keep `last_completed_*_frame_ms` / `last_gpu_source` for HUD display
        // without blocking -- the previous tick's values stand in until the next pairing lands.
    }

    /// Records main-thread pacing time that should not count as renderer CPU work.
    ///
    /// This is for explicit waits on GPU/display/compositor readiness that happen inside the
    /// app-driver tick after [`Self::begin_frame`] and before [`Self::end_frame`]. Calls outside
    /// an active frame are ignored so late cleanup and startup synchronization do not leak into
    /// the next HUD sample.
    pub(crate) fn record_excluded_wait(&mut self, wait: Duration) {
        if wait.is_zero() || self.frame_start.is_none() || self.finalized_seq.is_some() {
            return;
        }
        self.excluded_wait = self.excluded_wait.saturating_add(wait);
    }

    /// Publishes the main-thread CPU frame duration synchronously from the runtime tick
    /// epilogue, after the last `Queue::submit` dispatch but before the event-loop yield.
    ///
    /// This is the value the HUD's "CPU" row reflects. It excludes driver-thread queue
    /// overhead, FPS-gating sleeps, lockstep waits, and waits recorded through
    /// [`Self::record_excluded_wait`] -- those remain visible in the wall-clock "Frame" row.
    pub fn record_main_thread_cpu_end(&mut self, cpu_end: Instant) {
        let Some(frame_start) = self.frame_start else {
            return;
        };
        let gross_cpu = cpu_end.saturating_duration_since(frame_start);
        let cpu_ms = gross_cpu.saturating_sub(self.excluded_wait).as_secs_f64() * 1000.0;
        self.cpu_frame_ms = Some(cpu_ms);
        if self.submit_seq > 0 {
            let key = (self.generation, self.submit_seq);
            if self.pending_paired_cpu_ms.len() >= MAX_PENDING_PAIRS {
                self.pending_paired_cpu_ms.pop_front();
            }
            self.pending_paired_cpu_ms.push_back((key, cpu_ms));
        }
    }

    /// Call after all render graph submits for this tick (last submit index is known).
    ///
    /// Picks up the per-tick GPU value when the driver thread / readback already reported it;
    /// the GPU number may still arrive later, in which case
    /// [`Self::last_completed_paired_frame_ms`] is what the HUD renders.
    pub fn end_frame(&mut self) {
        if self.frame_start.is_none() {
            return;
        }
        self.finalized_seq = Some(self.submit_seq);
        if self.submit_seq > 0 {
            if let (Some(start), Some(real_submit_at)) = (
                self.frame_start,
                self.pending_real_submit_by_seq.remove(&self.submit_seq),
            ) {
                self.submit_latency_ms =
                    Some(real_submit_at.duration_since(start).as_secs_f64() * 1000.0);
            }
            self.pending_gpu_by_generation
                .entry(self.generation)
                .or_default()
                .set_expected_submits(self.submit_seq);
            self.publish_completed_frame_gpu_ms(self.generation);
        }
    }

    /// Records that the driver thread has finished `Queue::submit` for `seq`.
    ///
    /// Folds the value into the per-tick `submit_latency_ms` diagnostic when the tick is still
    /// current. Does **not** influence the HUD's CPU column -- that comes from
    /// [`Self::record_main_thread_cpu_end`].
    fn record_real_submit(
        &mut self,
        submitted_generation: u64,
        seq: u32,
        _frame_start: Instant,
        real_submit_at: Instant,
    ) {
        if submitted_generation != self.generation {
            return;
        }
        self.pending_real_submit_by_seq.insert(seq, real_submit_at);
        self.pending_gpu_by_generation
            .entry(submitted_generation)
            .or_default()
            .record_submit_at(real_submit_at);
        if self.finalized_seq == Some(seq) {
            self.submit_latency_ms = self
                .frame_start
                .map(|fs| real_submit_at.duration_since(fs).as_secs_f64() * 1000.0);
        }
    }

    /// Records a timestamp-backed tracked submit when its readback callback fires.
    ///
    /// If this completes the frame aggregate, publishes the GPU value used by the HUD and the
    /// host-facing `PerformanceState::render_time` field.
    fn record_frame_bracket_done(&mut self, submitted_generation: u64, seq: u32, gpu_ms: f64) {
        self.pending_gpu_by_generation
            .entry(submitted_generation)
            .or_default()
            .record_frame_bracket_done(gpu_ms);
        self.publish_completed_frame_gpu_ms(submitted_generation);
        if submitted_generation != self.generation {
            return;
        }
        if self.finalized_seq == Some(seq) {
            self.publish_completed_frame_gpu_ms(submitted_generation);
        }
    }

    /// Records a callback-latency fallback tracked submit when its completion callback fires.
    ///
    /// The frame fallback value spans from the tick's first tracked submit return to the last
    /// tracked completion callback, so a later tiny submit cannot replace the frame sample.
    fn record_callback_done(
        &mut self,
        submitted_generation: u64,
        seq: u32,
        real_submit_at: Instant,
        completed_at: Instant,
    ) {
        self.pending_gpu_by_generation
            .entry(submitted_generation)
            .or_default()
            .record_callback_done(real_submit_at, completed_at);
        self.publish_completed_frame_gpu_ms(submitted_generation);
        if submitted_generation != self.generation {
            return;
        }
        if self.finalized_seq == Some(seq) {
            self.publish_completed_frame_gpu_ms(submitted_generation);
        }
    }

    /// Publishes a completed frame GPU aggregate when the expected submit count is known and all
    /// tracked submits have finished.
    fn publish_completed_frame_gpu_ms(&mut self, generation: u64) {
        let Some((gpu_ms, source, expected_seq)) = self
            .pending_gpu_by_generation
            .get(&generation)
            .and_then(|frame| {
                let expected_seq = frame.expected_submits?;
                let (gpu_ms, source) = frame.completed_gpu_ms()?;
                Some((gpu_ms, source, expected_seq))
            })
        else {
            return;
        };
        self.pending_gpu_by_generation.remove(&generation);
        self.last_completed_gpu_frame_ms = Some(gpu_ms);
        self.last_gpu_source = Some(source);
        let key = (generation, expected_seq);
        if let Some(pos) = self
            .pending_paired_cpu_ms
            .iter()
            .position(|(k, _)| *k == key)
        {
            // Drain everything up to and including this entry: any older still-pending submits
            // have effectively been overtaken by this completion, so dropping their staged CPU
            // ms keeps the deque from growing across stalls.
            let mut last_cpu_ms = None;
            for _ in 0..=pos {
                last_cpu_ms = self.pending_paired_cpu_ms.pop_front().map(|(_, ms)| ms);
            }
            if let Some(cpu_ms) = last_cpu_ms {
                self.last_completed_paired_frame_ms = Some((cpu_ms, gpu_ms));
            }
        }
        if generation == self.generation && self.finalized_seq == Some(expected_seq) {
            self.gpu_frame_ms = Some(gpu_ms);
        }
    }

    /// Reserves the next submit index for the current tick.
    ///
    /// Returns [`None`] before the first [`Self::begin_frame`] or after a [`Self::end_frame`]
    /// without a follow-up `begin_frame`.
    pub fn on_before_tracked_submit(&mut self) -> Option<(u64, u32, Instant)> {
        let frame_start = self.frame_start?;
        self.submit_seq = self.submit_seq.saturating_add(1);
        Some((self.generation, self.submit_seq, frame_start))
    }
}

/// Shared timing state held by [`super::GpuContext`].
pub type FrameCpuGpuTimingHandle = Arc<Mutex<FrameCpuGpuTiming>>;

/// Records the driver-thread post-submit instant for a tracked batch.
///
/// Call from the driver thread immediately after [`wgpu::Queue::submit`] returns so the captured
/// instant reflects "CPU finished handing this frame to the driver." Returns the same instant so
/// callers can reuse it as the baseline for the callback-latency GPU fallback.
pub fn record_real_submit(track: &FrameTimingTrack) -> Instant {
    let real_submit_at = Instant::now();
    let mut g = track
        .handle
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    g.record_real_submit(
        track.generation,
        track.seq,
        track.frame_start,
        real_submit_at,
    );
    real_submit_at
}

/// Builds a callback that records one submit for the **callback-latency fallback** frame span.
///
/// Use this only when [`crate::gpu::frame_bracket::FrameBracket`] cannot produce a session for
/// the active adapter. `real_submit_at` must be captured on the driver thread after
/// `Queue::submit` returns, so the completed frame aggregate measures from the tick's first
/// submit return to its final `on_submitted_work_done` callback rather than including
/// driver-ring wait time.
pub fn make_gpu_done_callback(
    handle: FrameCpuGpuTimingHandle,
    generation: u64,
    seq: u32,
    real_submit_at: Instant,
) -> impl FnOnce() + Send + 'static {
    move || {
        let completed_at = Instant::now();
        let mut g = handle
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        g.record_callback_done(generation, seq, real_submit_at, completed_at);
    }
}

/// Publishes one frame-bracket-derived submit duration into the timing accumulator.
///
/// Call from the readback completion path; `gpu_ms` should already be `(end - begin) *
/// timestamp_period / 1e6`. The frame aggregate publishes after every tracked submit in the
/// same generation has completed.
pub fn record_frame_bracket_gpu_ms(
    handle: &FrameCpuGpuTimingHandle,
    generation: u64,
    seq: u32,
    gpu_ms: f64,
) {
    let mut g = handle
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    g.record_frame_bracket_done(generation, seq, gpu_ms);
}

#[cfg(test)]
mod tests {
    use super::{FrameCpuGpuTiming, GpuMsSource, MAX_PENDING_PAIRS};
    use std::time::{Duration, Instant};

    #[test]
    fn cpu_and_gpu_frame_ms_populated_when_main_thread_and_readback_arrive_in_time() {
        let mut t = FrameCpuGpuTiming::default();
        let frame_start = Instant::now();
        t.begin_frame(frame_start);
        let (generation, seq, fs) = t.on_before_tracked_submit().expect("tracked");
        assert_eq!(seq, 1);
        assert_eq!(fs, frame_start);
        let real_submit_at = frame_start + Duration::from_millis(3);
        t.record_real_submit(generation, seq, frame_start, real_submit_at);
        // Main-thread tick ends after the submit dispatch.
        let cpu_end = frame_start + Duration::from_millis(4);
        t.record_main_thread_cpu_end(cpu_end);
        t.record_frame_bracket_done(generation, seq, 5.0);
        t.end_frame();
        let cpu = t.cpu_frame_ms.expect("cpu_frame_ms");
        assert!((3.5..4.5).contains(&cpu), "cpu={cpu}");
        let submit_latency = t.submit_latency_ms.expect("submit_latency_ms");
        assert!(
            (2.5..3.5).contains(&submit_latency),
            "submit_latency={submit_latency}"
        );
        assert_eq!(t.gpu_frame_ms, Some(5.0));
        let (paired_cpu, paired_gpu) = t.last_completed_paired_frame_ms.expect("paired");
        assert!((3.5..4.5).contains(&paired_cpu), "paired_cpu={paired_cpu}");
        assert_eq!(paired_gpu, 5.0);
        assert_eq!(t.last_completed_gpu_frame_ms, Some(5.0));
        assert_eq!(t.last_gpu_source, Some(GpuMsSource::FrameBracket));
    }

    #[test]
    fn paired_frame_ms_survives_begin_frame_when_readback_arrives_late() {
        let mut t = FrameCpuGpuTiming::default();
        let start = Instant::now();
        t.begin_frame(start);
        let (generation, seq, fs) = t.on_before_tracked_submit().expect("tracked");
        let real_submit_at = fs + Duration::from_millis(2);
        t.record_real_submit(generation, seq, fs, real_submit_at);
        t.record_main_thread_cpu_end(start + Duration::from_millis(3));
        t.end_frame();
        // Next tick has already started by the time the GPU readback completes.
        t.begin_frame(start + Duration::from_millis(16));
        t.record_frame_bracket_done(generation, seq, 2.5);
        assert!(t.gpu_frame_ms.is_none());
        let (cpu, gpu) = t.last_completed_paired_frame_ms.expect("paired");
        assert!((2.5..3.5).contains(&cpu), "cpu={cpu}");
        assert_eq!(gpu, 2.5);
        assert_eq!(t.last_completed_gpu_frame_ms, Some(2.5));
        assert_eq!(t.last_gpu_source, Some(GpuMsSource::FrameBracket));
    }

    #[test]
    fn multi_submit_frame_render_time_sums_timestamp_brackets() {
        let mut t = FrameCpuGpuTiming::default();
        let start = Instant::now();
        t.begin_frame(start);
        let (generation, first_seq, _fs) = t.on_before_tracked_submit().expect("first tracked");
        let (_, second_seq, _) = t.on_before_tracked_submit().expect("second tracked");
        t.record_main_thread_cpu_end(start + Duration::from_millis(5));
        t.end_frame();

        t.record_frame_bracket_done(generation, second_seq, 0.0372);
        assert_eq!(t.last_completed_gpu_frame_ms, None);

        t.record_frame_bracket_done(generation, first_seq, 6.25);
        assert_ms_close(t.gpu_frame_ms, 6.2872);
        assert_ms_close(t.last_completed_gpu_frame_ms, 6.2872);
        let (paired_cpu, paired_gpu) = t.last_completed_paired_frame_ms.expect("paired");
        assert!((4.5..5.5).contains(&paired_cpu), "paired_cpu={paired_cpu}");
        assert!((paired_gpu - 6.2872).abs() < f64::EPSILON);
        assert_eq!(t.last_gpu_source, Some(GpuMsSource::FrameBracket));
    }

    #[test]
    fn late_multi_submit_readbacks_publish_previous_frame_aggregate() {
        let mut t = FrameCpuGpuTiming::default();
        let start = Instant::now();
        t.begin_frame(start);
        let (generation, first_seq, _fs) = t.on_before_tracked_submit().expect("first tracked");
        let (_, second_seq, _) = t.on_before_tracked_submit().expect("second tracked");
        t.record_main_thread_cpu_end(start + Duration::from_millis(4));
        t.end_frame();

        t.begin_frame(start + Duration::from_millis(16));
        t.record_frame_bracket_done(generation, first_seq, 3.5);
        assert!(t.last_completed_gpu_frame_ms.is_none());

        t.record_frame_bracket_done(generation, second_seq, 1.25);
        assert_eq!(t.gpu_frame_ms, None);
        assert_eq!(t.last_completed_gpu_frame_ms, Some(4.75));
        let (paired_cpu, paired_gpu) = t.last_completed_paired_frame_ms.expect("paired");
        assert!((3.5..4.5).contains(&paired_cpu), "paired_cpu={paired_cpu}");
        assert_eq!(paired_gpu, 4.75);
    }

    #[test]
    fn incomplete_multi_submit_frame_does_not_publish_partial_gpu_time() {
        let mut t = FrameCpuGpuTiming::default();
        let start = Instant::now();
        t.begin_frame(start);
        let (generation, first_seq, _fs) = t.on_before_tracked_submit().expect("first tracked");
        let (_, second_seq, _) = t.on_before_tracked_submit().expect("second tracked");
        t.record_main_thread_cpu_end(start + Duration::from_millis(5));
        t.end_frame();

        t.record_frame_bracket_done(generation, second_seq, 0.04);
        assert_eq!(t.gpu_frame_ms, None);
        assert_eq!(t.last_completed_gpu_frame_ms, None);
        assert_eq!(t.last_completed_paired_frame_ms, None);

        t.record_frame_bracket_done(generation, first_seq, 4.0);
        assert_ms_close(t.last_completed_gpu_frame_ms, 4.04);
    }

    #[test]
    fn callback_latency_fallback_labels_source_correctly() {
        let mut t = FrameCpuGpuTiming::default();
        let start = Instant::now();
        t.begin_frame(start);
        let (generation, seq, _fs) = t.on_before_tracked_submit().expect("tracked");
        t.record_main_thread_cpu_end(start + Duration::from_millis(2));
        t.record_callback_done(
            generation,
            seq,
            start + Duration::from_millis(1),
            start + Duration::from_millis(5),
        );
        t.end_frame();
        assert_eq!(t.last_gpu_source, Some(GpuMsSource::CallbackLatency));
        assert_eq!(t.gpu_frame_ms, Some(4.0));
    }

    #[test]
    fn multi_submit_callback_latency_publishes_one_frame_span() {
        let mut t = FrameCpuGpuTiming::default();
        let start = Instant::now();
        t.begin_frame(start);
        let (generation, first_seq, fs) = t.on_before_tracked_submit().expect("first tracked");
        let (_, second_seq, _) = t.on_before_tracked_submit().expect("second tracked");
        let first_submit_at = fs + Duration::from_millis(1);
        let second_submit_at = fs + Duration::from_millis(3);
        t.record_real_submit(generation, first_seq, fs, first_submit_at);
        t.record_real_submit(generation, second_seq, fs, second_submit_at);
        t.record_main_thread_cpu_end(start + Duration::from_millis(5));
        t.end_frame();

        t.record_callback_done(
            generation,
            first_seq,
            first_submit_at,
            fs + Duration::from_millis(7),
        );
        assert_eq!(t.last_completed_gpu_frame_ms, None);

        t.record_callback_done(
            generation,
            second_seq,
            second_submit_at,
            fs + Duration::from_millis(9),
        );
        assert_eq!(t.gpu_frame_ms, Some(8.0));
        assert_eq!(t.last_completed_gpu_frame_ms, Some(8.0));
        assert_eq!(t.last_gpu_source, Some(GpuMsSource::CallbackLatency));
    }

    #[test]
    fn cpu_frame_ms_subtracts_excluded_wait() {
        let mut t = FrameCpuGpuTiming::default();
        let start = Instant::now();
        t.begin_frame(start);
        let (generation, seq, _fs) = t.on_before_tracked_submit().expect("tracked");
        t.record_excluded_wait(Duration::from_millis(2));
        t.record_main_thread_cpu_end(start + Duration::from_millis(7));
        t.record_frame_bracket_done(generation, seq, 3.0);
        t.end_frame();

        let cpu = t.cpu_frame_ms.expect("cpu_frame_ms");
        assert!((4.5..5.5).contains(&cpu), "cpu={cpu}");
        let (paired_cpu, paired_gpu) = t.last_completed_paired_frame_ms.expect("paired");
        assert!((4.5..5.5).contains(&paired_cpu), "paired_cpu={paired_cpu}");
        assert_eq!(paired_gpu, 3.0);
    }

    #[test]
    fn excluded_wait_saturates_cpu_frame_ms_at_zero() {
        let mut t = FrameCpuGpuTiming::default();
        let start = Instant::now();
        t.begin_frame(start);
        let _ = t.on_before_tracked_submit().expect("tracked");
        t.record_excluded_wait(Duration::from_millis(10));
        t.record_main_thread_cpu_end(start + Duration::from_millis(4));

        assert_eq!(t.cpu_frame_ms, Some(0.0));
    }

    #[test]
    fn unmatched_gpu_done_does_not_publish_a_pair() {
        let mut t = FrameCpuGpuTiming::default();
        let start = Instant::now();
        t.begin_frame(start);
        let (generation, seq, _fs) = t.on_before_tracked_submit().expect("tracked");
        t.end_frame();
        // No record_main_thread_cpu_end ever fires; HUD pair must stay None.
        t.record_frame_bracket_done(generation, seq, 7.0);
        assert!(t.last_completed_paired_frame_ms.is_none());
        assert_eq!(t.last_completed_gpu_frame_ms, Some(7.0));
        assert_eq!(t.last_gpu_source, Some(GpuMsSource::FrameBracket));
    }

    #[test]
    fn pending_paired_cpu_evicts_oldest_at_capacity() {
        let mut t = FrameCpuGpuTiming::default();
        let start = Instant::now();
        // Push more pending CPU records than the cap, all without a matching GPU done. The
        // deque must stay at exactly MAX_PENDING_PAIRS, dropping the oldest entries.
        for i in 0..(MAX_PENDING_PAIRS as u32 + 5) {
            t.begin_frame(start + Duration::from_millis(u64::from(i)));
            let _ = t.on_before_tracked_submit().expect("tracked");
            t.record_main_thread_cpu_end(start + Duration::from_millis(u64::from(i) + 1));
            t.end_frame();
        }
        assert_eq!(t.pending_paired_cpu_ms.len(), MAX_PENDING_PAIRS);
    }

    fn assert_ms_close(actual: Option<f64>, expected: f64) {
        let actual = actual.expect("gpu ms");
        assert!(
            (actual - expected).abs() < 1e-9,
            "actual={actual} expected={expected}",
        );
    }
}
