//! Frame-timing and GPU-profiler facade methods on [`GpuContext`].
//!
//! Couples the wall-clock CPU/GPU intervals consumed by the debug HUD with the wgpu
//! profiler's pass-level timestamp queries; both feed the same `submission` bundle so the
//! main tick reads them without blocking.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::GpuContext;

impl GpuContext {
    /// Call at the start of each winit frame tick (same instant as [`crate::runtime::RendererRuntime::tick_frame_wall_clock_begin`]).
    pub fn begin_frame_timing(&self, frame_start: Instant) {
        profiling::scope!("gpu::begin_frame_timing");
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

    /// Mutable reference to the GPU profiler, when one is active.
    ///
    /// Returns [`None`] when the `tracy` feature is off, or when the adapter lacks the required
    /// timestamp-query features (see [`crate::profiling::GpuProfilerHandle::try_new`]).
    pub fn gpu_profiler_mut(&mut self) -> Option<&mut crate::profiling::GpuProfilerHandle> {
        self.submission.gpu_profiler.as_mut()
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
        if self.submission.gpu_profiler.is_none() {
            return;
        }
        let had_queries =
            self.submission.gpu_profiler.as_ref().is_some_and(
                crate::profiling::GpuProfilerHandle::has_queries_opened_since_frame_end,
            );
        if had_queries {
            // `wgpu_profiler::end_frame` calls `map_async` on the same Query Read Buffer that
            // `resolve_queries` just wrote a copy into. The render graph hands those resolve
            // command buffers to the driver thread for an asynchronous `Queue::submit`, so if
            // the driver has not yet drained the ring by the time we reach this point,
            // `map_async` would put the buffer in pending-mapped state before the submit runs
            // and wgpu validation would reject it with "buffer is still mapped". Flushing the
            // driver guarantees every prior submit has completed before we transition the
            // buffer. Empty redraw ticks skip the flush and the profiler frame close.
            self.submission.driver_thread.flush();
        }
        if let Some(p) = self.submission.gpu_profiler.as_mut() {
            p.end_frame_if_queries_opened();
            let ts_period = self.queue.get_timestamp_period();
            let mut latest_timings = None;
            while let Some(timings) = p.process_finished_frame(ts_period) {
                latest_timings = Some(timings);
            }
            if let Some(timings) = latest_timings
                && let Ok(mut slot) = self.submission.latest_gpu_pass_timings.lock()
            {
                *slot = timings;
            }
        }
    }

    /// Returns a shared handle to the latest flattened per-pass GPU timings.
    ///
    /// The debug HUD polls this once per frame. The underlying vector is replaced atomically by
    /// [`Self::end_gpu_profiler_frame`] on the main thread; readers clone the current contents
    /// under a short lock and render them without blocking the renderer.
    pub fn latest_gpu_pass_timings_handle(
        &self,
    ) -> Arc<Mutex<Vec<crate::profiling::GpuPassEntry>>> {
        Arc::clone(&self.submission.latest_gpu_pass_timings)
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

    /// Publishes the main-thread CPU frame duration synchronously.
    ///
    /// Call from the runtime tick epilogue, after the last [`wgpu::Queue::submit`] dispatch
    /// but before the event-loop yields. The captured duration becomes the HUD's "CPU" row
    /// reading -- see
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

    /// Most recently completed GPU frame ms in **seconds**, for the IPC
    /// [`crate::shared::PerformanceState::render_time`] field consumed by
    /// `FrooxEngine.PerformanceMetrics.RenderTime`.
    ///
    /// Returns [`None`] until the first [`wgpu::Queue::on_submitted_work_done`] callback has run;
    /// callers that need the host-visible "unavailable" sentinel should map [`None`] to `-1.0`,
    /// matching the Renderite.Unity `XRStats.TryGetGPUTimeLastFrame` contract.
    pub fn last_completed_gpu_render_time_seconds(&self) -> Option<f32> {
        let ft = self
            .submission
            .frame_timing
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        ft.last_completed_gpu_frame_ms
            .map(|ms| (ms / 1000.0) as f32)
    }

    /// Process-local GPU memory from wgpu's allocator when the active backend supports
    /// [`wgpu::Device::generate_allocator_report`].
    ///
    /// Returns `(allocated_bytes, reserved_bytes)`, or `(None, None)` when the backend does not report.
    /// The **Stats** debug HUD tab uses these totals every capture; the **GPU memory** tab uses a
    /// throttled full [`wgpu::AllocatorReport`] via [`crate::runtime::RendererRuntime`].
    pub fn gpu_allocator_bytes(&self) -> (Option<u64>, Option<u64>) {
        self.device
            .generate_allocator_report()
            .map_or((None, None), |r| {
                (Some(r.total_allocated_bytes), Some(r.total_reserved_bytes))
            })
    }
}
