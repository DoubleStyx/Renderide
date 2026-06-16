//! Frame-timing and GPU-profiler facade methods on [`GpuContext`].
//!
//! Couples the wall-clock CPU/GPU intervals consumed by the debug HUD with the wgpu
//! profiler's pass-level timestamp queries; both feed the same `submission` bundle so the
//! main tick reads them without blocking.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::gpu::driver_thread::SubmitToken;
#[cfg(feature = "tracy")]
use crate::log_throttle::LogThrottle;

use super::GpuContext;

/// Throttles logs when every profiler handle is waiting for submit completion.
#[cfg(feature = "tracy")]
static GPU_PROFILER_POOL_EXHAUSTED_LOG: LogThrottle = LogThrottle::new();
/// Throttles logs when replacement profiler creation unexpectedly fails.
#[cfg(feature = "tracy")]
static GPU_PROFILER_REPLACEMENT_FAILED_LOG: LogThrottle = LogThrottle::new();

/// Drains completed timestamp trees from one Tracy-enabled profiler handle.
#[cfg(feature = "tracy")]
fn drain_profiler_handle_results(
    profiler: &mut crate::profiling::GpuProfilerHandle,
    timestamp_period: f32,
    latest_published_frame_order: u64,
    latest_snapshot: &mut Option<crate::profiling::GpuProfilerSnapshot>,
) {
    while let Some(snapshot) = profiler.process_finished_frame(timestamp_period) {
        record_newer_gpu_profiler_snapshot(snapshot, latest_published_frame_order, latest_snapshot);
    }
}

/// Drains completed timestamp trees from the no-Tracy profiler stub.
#[cfg(not(feature = "tracy"))]
fn drain_profiler_handle_results(
    profiler: &crate::profiling::GpuProfilerHandle,
    timestamp_period: f32,
    latest_published_frame_order: u64,
    latest_snapshot: &mut Option<crate::profiling::GpuProfilerSnapshot>,
) {
    while let Some(snapshot) = profiler.process_finished_frame(timestamp_period) {
        record_newer_gpu_profiler_snapshot(snapshot, latest_published_frame_order, latest_snapshot);
    }
}

/// Keeps the newest unpublished GPU profiler snapshot encountered during a drain pass.
fn record_newer_gpu_profiler_snapshot(
    snapshot: crate::profiling::GpuProfilerSnapshot,
    latest_published_frame_order: u64,
    latest_snapshot: &mut Option<crate::profiling::GpuProfilerSnapshot>,
) {
    let frame_order = snapshot.stats.frame_order;
    if frame_order <= latest_published_frame_order {
        return;
    }
    let replace = latest_snapshot
        .as_ref()
        .is_none_or(|current| frame_order > current.stats.frame_order);
    if replace {
        *latest_snapshot = Some(snapshot);
    }
}

impl GpuContext {
    /// Call at the start of each winit frame tick (same instant as [`crate::runtime::RendererRuntime::tick_frame_wall_clock_begin`]).
    pub fn begin_frame_timing(&mut self, frame_start: Instant) {
        profiling::scope!("gpu::begin_frame_timing");
        self.finish_deferred_gpu_profiler_frames_if_ready();
        self.drain_gpu_profiler_results();
        self.ensure_active_gpu_profiler();
        self.refresh_gpu_profiler_tracy_bridge();
        self.submission
            .last_frame_submit_token
            .store(0, Ordering::Release);
        self.submission
            .frame_timing
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .begin_frame(frame_start);
    }

    /// Call after all tracked queue submits for this tick (before reading HUD metrics).
    ///
    /// Folds in this tick's CPU/GPU values when the driver thread already reported them; both
    /// numbers are updated asynchronously on the driver thread / completion-callback thread, so
    /// the HUD reads `last_completed_*_frame_ms` instead of blocking on
    /// [`wgpu::Device::poll`].
    pub fn end_frame_timing(&self) {
        profiling::scope!("gpu::end_frame_timing");
        let mut ft = self
            .submission
            .frame_timing
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        ft.end_frame();
    }

    /// Adds main-thread pacing time that should be subtracted from the HUD CPU frame value.
    ///
    /// Use this for waits on GPU/display/compositor readiness. The timing accumulator ignores
    /// calls outside an active frame, so callers may record measured waits unconditionally.
    pub(crate) fn record_frame_timing_excluded_wait(&self, wait: Duration) {
        if wait.is_zero() {
            return;
        }
        let mut ft = self
            .submission
            .frame_timing
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        ft.record_excluded_wait(wait);
    }

    /// Mutable reference to the GPU profiler, when one is active.
    ///
    /// Returns [`None`] when the `tracy` feature is off, or when the adapter lacks the required
    /// timestamp-query features (see [`crate::profiling::GpuProfilerHandle::try_new`]).
    pub fn gpu_profiler_mut(&mut self) -> Option<&mut crate::profiling::GpuProfilerHandle> {
        self.finish_deferred_gpu_profiler_frames_if_ready();
        self.ensure_active_gpu_profiler();
        self.refresh_gpu_profiler_tracy_bridge();
        self.submission.gpu_profiler_pool.active_mut()
    }

    /// Shared reference to the GPU profiler, when one is active.
    ///
    /// Use this for helpers that only need to reserve pass-level timestamp queries and already
    /// receive an immutable [`GpuContext`] borrow. Query resolution still happens through the
    /// mutable frame-end paths.
    pub fn gpu_profiler(&self) -> Option<&crate::profiling::GpuProfilerHandle> {
        self.submission.gpu_profiler_pool.active()
    }

    /// Temporarily removes the GPU profiler handle from [`GpuContext`] and returns it.
    ///
    /// Use this when code must hold a borrowed reference into `GpuContext` (e.g. a
    /// `ResolvedView` that borrows `depth_texture`) while also needing to drive the profiler
    /// inside a nested loop. Pair every call with [`Self::restore_gpu_profiler`].
    ///
    /// Returns [`None`] when no profiler is active (feature off or adapter unsupported).
    pub fn take_gpu_profiler(&mut self) -> Option<crate::profiling::GpuProfilerHandle> {
        self.finish_deferred_gpu_profiler_frames_if_ready();
        self.ensure_active_gpu_profiler();
        self.refresh_gpu_profiler_tracy_bridge();
        self.submission.gpu_profiler_pool.take_active()
    }

    /// Restores a profiler handle previously removed by [`Self::take_gpu_profiler`].
    ///
    /// If `profiler` is [`None`], this is a no-op.
    pub fn restore_gpu_profiler(&mut self, profiler: Option<crate::profiling::GpuProfilerHandle>) {
        self.submission.gpu_profiler_pool.restore_active(profiler);
    }

    /// Ends the GPU profiling frame and drains completed query results into Tracy.
    ///
    /// Call once per render tick after all command encoders for the tick have been submitted
    /// (e.g. from the app driver's frame epilogue).
    /// Does nothing when no GPU profiler is active.
    pub fn end_gpu_profiler_frame(&mut self) {
        profiling::scope!("gpu::drain_gpu_profiler");
        self.finish_deferred_gpu_profiler_frames_if_ready();
        self.drain_gpu_profiler_results();
        self.ensure_active_gpu_profiler();
        if self.submission.gpu_profiler_pool.active().is_none() {
            return;
        }
        let had_queries =
            self.submission.gpu_profiler_pool.active().is_some_and(
                crate::profiling::GpuProfilerHandle::has_queries_opened_since_frame_end,
            );
        if !had_queries {
            return;
        }
        let frame_order = self.submission.gpu_profiler_pool.allocate_frame_order();
        let Some(submit_token) = self.last_frame_submit_token() else {
            logger::warn!("GPU profiler frame had queries but no submit token");
            self.end_active_gpu_profiler_frame(frame_order);
            self.drain_gpu_profiler_results();
            return;
        };
        if self.submission.driver_thread.is_submit_done(submit_token) {
            self.end_active_gpu_profiler_frame(frame_order);
            self.drain_gpu_profiler_results();
            return;
        }
        if self
            .submission
            .gpu_profiler_pool
            .defer_active_until_submit(submit_token, frame_order)
        {
            self.ensure_active_gpu_profiler();
        }
    }

    fn last_frame_submit_token(&self) -> Option<SubmitToken> {
        let raw = self
            .submission
            .last_frame_submit_token
            .load(Ordering::Acquire);
        (raw != 0).then(|| SubmitToken::new(raw))
    }

    fn finish_deferred_gpu_profiler_frames_if_ready(&mut self) {
        while let Some(submit_token) = self
            .submission
            .gpu_profiler_pool
            .front_pending_submit_token()
        {
            if !self.submission.driver_thread.is_submit_done(submit_token) {
                break;
            }
            let Some(pending) = self.submission.gpu_profiler_pool.pop_front_pending_submit() else {
                break;
            };
            #[cfg(feature = "tracy")]
            let mut profiler = pending.profiler;
            #[cfg(not(feature = "tracy"))]
            let profiler = pending.profiler;
            profiler.end_frame_if_queries_opened(pending.frame_order);
            self.submission.gpu_profiler_pool.push_ready(profiler);
        }
    }

    /// Refreshes the Tracy GPU bridge after all older query frames that can safely end have ended.
    ///
    /// Late Tracy GUI attach/detach is handled by swapping the underlying `wgpu-profiler`
    /// instance only when no command-buffer submit still owns the current profiler frame.
    fn refresh_gpu_profiler_tracy_bridge(&mut self) {
        let pending_submit_end = self.submission.gpu_profiler_pool.has_pending_submit_end();
        let backend = self.adapter_info.backend;
        let device = Arc::clone(&self.device);
        let queue = Arc::clone(&self.queue);
        if let Some(profiler) = self.submission.gpu_profiler_pool.active_mut() {
            profiler.refresh_tracy_bridge(
                backend,
                device.as_ref(),
                queue.as_ref(),
                pending_submit_end,
            );
        }
    }

    #[cfg(feature = "tracy")]
    fn end_active_gpu_profiler_frame(&mut self, frame_order: u64) {
        if let Some(profiler) = self.submission.gpu_profiler_pool.active_mut() {
            profiler.end_frame_if_queries_opened(frame_order);
        }
    }

    #[cfg(not(feature = "tracy"))]
    fn end_active_gpu_profiler_frame(&self, frame_order: u64) {
        if let Some(profiler) = self.submission.gpu_profiler_pool.active() {
            profiler.end_frame_if_queries_opened(frame_order);
        }
    }

    #[cfg(feature = "tracy")]
    fn drain_gpu_profiler_results(&mut self) {
        let ts_period = self.queue.get_timestamp_period();
        let mut latest_snapshot = None;
        let latest_published = self
            .submission
            .gpu_profiler_pool
            .latest_published_frame_order();
        if let Some(profiler) = self.submission.gpu_profiler_pool.active_mut() {
            drain_profiler_handle_results(
                profiler,
                ts_period,
                latest_published,
                &mut latest_snapshot,
            );
        }
        for profiler in self.submission.gpu_profiler_pool.ready_mut() {
            drain_profiler_handle_results(
                profiler,
                ts_period,
                latest_published,
                &mut latest_snapshot,
            );
        }
        if let Some(snapshot) = latest_snapshot.as_ref() {
            self.submission
                .gpu_profiler_pool
                .mark_published_frame_order(snapshot.stats.frame_order);
        }
        self.publish_latest_gpu_profiler_snapshot(latest_snapshot);
    }

    #[cfg(feature = "tracy")]
    fn ensure_active_gpu_profiler(&mut self) {
        if self.submission.gpu_profiler_pool.activate_ready() {
            return;
        }
        if self.submission.gpu_profiler_pool.active().is_some() {
            return;
        }
        if !self.submission.gpu_profiler_pool.can_allocate_replacement() {
            self.log_gpu_profiler_pool_unavailable_if_needed();
            return;
        }
        let backend = self.adapter_info.backend;
        let device = Arc::clone(&self.device);
        let queue = Arc::clone(&self.queue);
        let profiler = if self.submission.gpu_profiler_pool.has_pending_submit_end() {
            crate::profiling::GpuProfilerHandle::try_new_for_backend_unbridged(
                backend,
                device.as_ref(),
                queue.as_ref(),
            )
        } else {
            crate::profiling::GpuProfilerHandle::try_new_for_backend(
                backend,
                device.as_ref(),
                queue.as_ref(),
            )
        };
        let Some(profiler) = profiler else {
            if let Some(occurrence) = GPU_PROFILER_REPLACEMENT_FAILED_LOG.should_log(1, 256) {
                logger::warn!(
                    "GPU profiler replacement allocation failed; GPU Tracy scopes will be skipped until a handle is available (occurrence={occurrence})"
                );
            }
            return;
        };
        if !self
            .submission
            .gpu_profiler_pool
            .insert_allocated_active(profiler)
            && let Some(occurrence) = GPU_PROFILER_POOL_EXHAUSTED_LOG.should_log(1, 256)
        {
            logger::warn!(
                "GPU profiler replacement handle was dropped because the pool filled concurrently (occurrence={occurrence})"
            );
        }
    }

    #[cfg(feature = "tracy")]
    fn log_gpu_profiler_pool_unavailable_if_needed(&self) {
        let pool = &self.submission.gpu_profiler_pool;
        if !pool.enabled() || pool.active().is_some() {
            return;
        }
        if let Some(occurrence) = GPU_PROFILER_POOL_EXHAUSTED_LOG.should_log(1, 256) {
            logger::warn!(
                "GPU profiler handle pool exhausted; skipping GPU Tracy scopes for this frame (live_handles={} occurrence={occurrence})",
                pool.live_handle_count()
            );
        }
    }

    #[cfg(not(feature = "tracy"))]
    fn ensure_active_gpu_profiler(&self) {}

    #[cfg(not(feature = "tracy"))]
    fn drain_gpu_profiler_results(&self) {
        let ts_period = self.queue.get_timestamp_period();
        let mut latest_snapshot = None;
        let latest_published = self
            .submission
            .gpu_profiler_pool
            .latest_published_frame_order();
        if let Some(profiler) = self.submission.gpu_profiler_pool.active() {
            drain_profiler_handle_results(
                profiler,
                ts_period,
                latest_published,
                &mut latest_snapshot,
            );
        }
        self.publish_latest_gpu_profiler_snapshot(latest_snapshot);
    }

    fn publish_latest_gpu_profiler_snapshot(
        &self,
        latest_snapshot: Option<crate::profiling::GpuProfilerSnapshot>,
    ) {
        if let Some(snapshot) = latest_snapshot
            && let Ok(mut slot) = self.submission.latest_gpu_profiler_snapshot.lock()
        {
            *slot = snapshot;
        }
    }

    /// Returns a shared handle to the latest flattened per-pass GPU timings and query stats.
    ///
    /// The debug HUD polls this once per frame. The underlying vector is replaced atomically by
    /// [`Self::end_gpu_profiler_frame`] on the main thread; readers clone the current contents
    /// under a short lock and render them without blocking the renderer.
    pub fn latest_gpu_profiler_snapshot_handle(
        &self,
    ) -> Arc<Mutex<crate::profiling::GpuProfilerSnapshot>> {
        Arc::clone(&self.submission.latest_gpu_profiler_snapshot)
    }

    /// Most recently completed primary CPU/GPU frame work for the debug HUD.
    ///
    /// Returns [`None`] until the first primary submit has both published its main-thread CPU
    /// duration via [`Self::record_main_thread_cpu_end`] and delivered a completion signal.
    /// Timestamp-backed completions include GPU busy time; callback-only completions keep GPU
    /// busy time unavailable. Once a sample has been observed, it survives across frames so the
    /// overlay never goes blank. Lags the current tick by at least one frame in steady state,
    /// since the GPU readback for frame N typically lands after frame N+1's tick has begun.
    pub fn primary_frame_work_timing_for_hud(
        &self,
    ) -> Option<crate::gpu::frame_cpu_gpu_timing::PrimaryFrameWorkTiming> {
        let ft = self
            .submission
            .frame_timing
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        ft.last_completed_primary_frame_ms
    }

    /// Publishes the active main-thread CPU frame duration synchronously.
    ///
    /// Call from the runtime tick epilogue, after the last [`wgpu::Queue::submit`] dispatch
    /// but before the event-loop yields. The timing accumulator subtracts excluded pacing waits
    /// before publishing the HUD's "CPU" row reading -- see
    /// [`crate::gpu::frame_cpu_gpu_timing::FrameCpuGpuTiming::record_main_thread_cpu_end`].
    pub fn record_main_thread_cpu_end(&self, cpu_end: Instant) {
        profiling::scope!("gpu::record_main_thread_cpu_end");
        let mut ft = self
            .submission
            .frame_timing
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        ft.record_main_thread_cpu_end(cpu_end);
    }

    /// Most recently completed timestamp-backed primary GPU time in **seconds**, for the IPC
    /// [`crate::shared::PerformanceState::render_time`] field consumed by
    /// `FrooxEngine.PerformanceMetrics.RenderTime`.
    ///
    /// Returns [`None`] until every primary submit in at least one frame has completed with
    /// hardware timestamp data; callers that need the host-visible "unavailable" sentinel should
    /// map [`None`] to `-1.0`.
    pub fn last_completed_gpu_render_time_seconds(&self) -> Option<f32> {
        let ft = self
            .submission
            .frame_timing
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        ft.last_completed_gpu_frame_ms
            .map(|ms| (ms / 1000.0) as f32)
    }
}

#[cfg(test)]
mod tests {
    use crate::profiling::{GpuPassEntry, GpuProfilerFrameStats, GpuProfilerSnapshot};

    use super::record_newer_gpu_profiler_snapshot;

    #[test]
    fn snapshot_publish_selection_keeps_newest_unpublished_frame() {
        let mut latest = None;

        record_newer_gpu_profiler_snapshot(snapshot(7), 4, &mut latest);
        record_newer_gpu_profiler_snapshot(snapshot(6), 4, &mut latest);
        record_newer_gpu_profiler_snapshot(snapshot(9), 4, &mut latest);

        assert_eq!(latest.expect("snapshot selected").stats.frame_order, 9);
    }

    #[test]
    fn snapshot_publish_selection_ignores_already_published_frames() {
        let mut latest = None;

        record_newer_gpu_profiler_snapshot(snapshot(3), 4, &mut latest);
        record_newer_gpu_profiler_snapshot(snapshot(4), 4, &mut latest);

        assert!(latest.is_none());
    }

    fn snapshot(frame_order: u64) -> GpuProfilerSnapshot {
        GpuProfilerSnapshot {
            entries: vec![GpuPassEntry {
                name: format!("frame_{frame_order}"),
                ms: frame_order as f32,
                depth: 0,
            }],
            stats: GpuProfilerFrameStats {
                frame_order,
                opened_queries: 1,
                skipped_queries: 0,
                soft_query_budget: 512,
            },
        }
    }
}
