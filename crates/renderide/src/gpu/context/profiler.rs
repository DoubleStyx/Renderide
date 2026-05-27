//! Frame-timing and GPU-profiler facade methods on [`GpuContext`].
//!
//! Couples the wall-clock CPU/GPU intervals consumed by the debug HUD with the wgpu
//! profiler's pass-level timestamp queries; both feed the same `submission` bundle so the
//! main tick reads them without blocking.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::gpu::driver_thread::SubmitToken;
use crate::gpu::submission_state::PendingGpuProfilerEnd;

use super::GpuContext;

impl GpuContext {
    /// Call at the start of each winit frame tick (same instant as [`crate::runtime::RendererRuntime::tick_frame_wall_clock_begin`]).
    pub fn begin_frame_timing(&mut self, frame_start: Instant) {
        profiling::scope!("gpu::begin_frame_timing");
        self.finish_deferred_gpu_profiler_frame_if_ready();
        self.drain_gpu_profiler_results();
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
        self.submission.gpu_profiler.as_mut()
    }

    /// Shared reference to the GPU profiler, when one is active.
    ///
    /// Use this for helpers that only need to reserve pass-level timestamp queries and already
    /// receive an immutable [`GpuContext`] borrow. Query resolution still happens through the
    /// mutable frame-end paths.
    pub fn gpu_profiler(&self) -> Option<&crate::profiling::GpuProfilerHandle> {
        self.submission.gpu_profiler.as_ref()
    }

    /// Temporarily removes the GPU profiler handle from [`GpuContext`] and returns it.
    ///
    /// Use this when code must hold a borrowed reference into `GpuContext` (e.g. a
    /// `ResolvedView` that borrows `depth_texture`) while also needing to drive the profiler
    /// inside a nested loop. Pair every call with [`Self::restore_gpu_profiler`].
    ///
    /// Returns [`None`] when no profiler is active (feature off or adapter unsupported).
    pub fn take_gpu_profiler(&mut self) -> Option<crate::profiling::GpuProfilerHandle> {
        self.submission.gpu_profiler.take()
    }

    /// Restores a profiler handle previously removed by [`Self::take_gpu_profiler`].
    ///
    /// If `profiler` is [`None`], this is a no-op.
    pub fn restore_gpu_profiler(&mut self, profiler: Option<crate::profiling::GpuProfilerHandle>) {
        if self.submission.gpu_profiler.is_none() {
            self.submission.gpu_profiler = profiler;
        }
    }

    /// Ends the GPU profiling frame and drains completed query results into Tracy.
    ///
    /// Call once per render tick after all command encoders for the tick have been submitted
    /// (e.g. from the app driver's frame epilogue).
    /// Does nothing when no GPU profiler is active.
    pub fn end_gpu_profiler_frame(&mut self) {
        profiling::scope!("gpu::drain_gpu_profiler");
        self.finish_deferred_gpu_profiler_frame_if_ready();
        self.drain_gpu_profiler_results();
        if self.submission.gpu_profiler.is_none() {
            return;
        }
        let had_queries =
            self.submission.gpu_profiler.as_ref().is_some_and(
                crate::profiling::GpuProfilerHandle::has_queries_opened_since_frame_end,
            );
        if !had_queries {
            return;
        }
        let Some(submit_token) = self.last_frame_submit_token() else {
            logger::warn!("GPU profiler frame had queries but no submit token");
            self.end_active_gpu_profiler_frame();
            self.drain_gpu_profiler_results();
            return;
        };
        if self.submission.driver_thread.is_submit_done(submit_token) {
            self.end_active_gpu_profiler_frame();
            self.drain_gpu_profiler_results();
            return;
        }
        if let Some(profiler) = self.submission.gpu_profiler.take() {
            self.submission.pending_gpu_profiler_end = Some(PendingGpuProfilerEnd {
                submit_token,
                profiler,
            });
        }
    }

    fn last_frame_submit_token(&self) -> Option<SubmitToken> {
        let raw = self
            .submission
            .last_frame_submit_token
            .load(Ordering::Acquire);
        (raw != 0).then(|| SubmitToken::new(raw))
    }

    fn finish_deferred_gpu_profiler_frame_if_ready(&mut self) {
        let Some(pending) = self.submission.pending_gpu_profiler_end.as_ref() else {
            return;
        };
        if !self
            .submission
            .driver_thread
            .is_submit_done(pending.submit_token)
        {
            return;
        }
        #[cfg(feature = "tracy")]
        let Some(mut pending) = self.submission.pending_gpu_profiler_end.take() else {
            return;
        };
        #[cfg(not(feature = "tracy"))]
        let Some(pending) = self.submission.pending_gpu_profiler_end.take() else {
            return;
        };
        pending.profiler.end_frame_if_queries_opened();
        self.submission.gpu_profiler = Some(pending.profiler);
    }

    /// Refreshes the Tracy GPU bridge after all older query frames that can safely end have ended.
    ///
    /// Late Tracy GUI attach/detach is handled by swapping the underlying `wgpu-profiler`
    /// instance only when no command-buffer submit still owns the current profiler frame.
    fn refresh_gpu_profiler_tracy_bridge(&mut self) {
        let pending_submit_end = self.submission.pending_gpu_profiler_end.is_some();
        let backend = self.adapter_info.backend;
        let device = Arc::clone(&self.device);
        let queue = Arc::clone(&self.queue);
        if let Some(profiler) = self.submission.gpu_profiler.as_mut() {
            profiler.refresh_tracy_bridge(
                backend,
                device.as_ref(),
                queue.as_ref(),
                pending_submit_end,
            );
        }
    }

    #[cfg(feature = "tracy")]
    fn end_active_gpu_profiler_frame(&mut self) {
        if let Some(profiler) = self.submission.gpu_profiler.as_mut() {
            profiler.end_frame_if_queries_opened();
        }
    }

    #[cfg(not(feature = "tracy"))]
    fn end_active_gpu_profiler_frame(&self) {
        if let Some(profiler) = self.submission.gpu_profiler.as_ref() {
            profiler.end_frame_if_queries_opened();
        }
    }

    #[cfg(feature = "tracy")]
    fn drain_gpu_profiler_results(&mut self) {
        let ts_period = self.queue.get_timestamp_period();
        let Some(profiler) = self.submission.gpu_profiler.as_mut() else {
            return;
        };
        let mut latest_snapshot = None;
        while let Some(snapshot) = profiler.process_finished_frame(ts_period) {
            latest_snapshot = Some(snapshot);
        }
        self.publish_latest_gpu_profiler_snapshot(latest_snapshot);
    }

    #[cfg(not(feature = "tracy"))]
    fn drain_gpu_profiler_results(&self) {
        let ts_period = self.queue.get_timestamp_period();
        let Some(profiler) = self.submission.gpu_profiler.as_ref() else {
            return;
        };
        let mut latest_snapshot = None;
        while let Some(snapshot) = profiler.process_finished_frame(ts_period) {
            latest_snapshot = Some(snapshot);
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

    /// Most recently completed CPU and GPU per-frame ms for the debug HUD, paired so both
    /// values describe the **same** frame.
    ///
    /// Returns `(None, None)` until the first submit has both published its main-thread CPU
    /// duration via [`Self::record_main_thread_cpu_end`] *and* delivered a GPU value (real
    /// timestamp readback or callback-latency fallback). Once a pair has been observed, the
    /// values survive across frames so the overlay never goes blank. Lags the current tick by
    /// at least one frame in steady state, since the GPU readback for frame N typically lands
    /// after frame N+1's tick has begun.
    pub fn frame_cpu_gpu_ms_for_hud(&self) -> (Option<f64>, Option<f64>) {
        let ft = self
            .submission
            .frame_timing
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match ft.last_completed_paired_frame_ms {
            Some((cpu, gpu)) => (Some(cpu), Some(gpu)),
            None => (None, None),
        }
    }

    /// Origin of the most recent `gpu_frame_ms` value, so the HUD can label the row honestly.
    ///
    /// Returns [`None`] until the first GPU value has been published. See
    /// [`crate::gpu::frame_cpu_gpu_timing::GpuMsSource`].
    pub fn last_gpu_ms_source(&self) -> Option<crate::gpu::frame_cpu_gpu_timing::GpuMsSource> {
        let ft = self
            .submission
            .frame_timing
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        ft.last_gpu_source
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

    /// Most recently completed whole-frame GPU time in **seconds**, for the IPC
    /// [`crate::shared::PerformanceState::render_time`] field consumed by
    /// `FrooxEngine.PerformanceMetrics.RenderTime`.
    ///
    /// Returns [`None`] until every tracked submit in at least one frame has completed; callers
    /// that need the host-visible "unavailable" sentinel should map [`None`] to `-1.0`.
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
