//! Submission, frame timing, and GPU profiling state owned by [`super::GpuContext`].

use std::collections::VecDeque;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};

use super::driver_thread::{DriverThread, RING_CAPACITY, SubmitToken};
use super::profiling::frame_bracket::FrameBracket;
use super::profiling::frame_cpu_gpu_timing::FrameCpuGpuTimingHandle;

/// Maximum live GPU profiler handles kept while driver-thread submit completion lags recording.
pub(super) const GPU_PROFILER_POOL_CAPACITY: usize = RING_CAPACITY + 2;

/// GPU profiler frame waiting for the driver thread to submit its command buffers.
pub(super) struct PendingGpuProfilerEnd<T> {
    /// Submit token that must complete before `wgpu-profiler::end_frame` is legal.
    pub(super) submit_token: SubmitToken,
    /// Monotonic profiler-frame order assigned when the frame leaves the active slot.
    pub(super) frame_order: u64,
    /// Profiler handle that owns the query frame waiting for submit completion.
    pub(super) profiler: T,
}

/// Bounded pool of GPU profiler handles used to preserve frame pipelining.
pub(super) struct GpuProfilerPool<T> {
    /// Handle currently used by new query reservations.
    active: Option<T>,
    /// Whether the active handle is temporarily checked out through `take_gpu_profiler`.
    checked_out_active: bool,
    /// Ended handles ready to be reused for new query reservations.
    ready: Vec<T>,
    /// Handles waiting for the driver thread to submit command buffers before frame close.
    pending_submit_end: VecDeque<PendingGpuProfilerEnd<T>>,
    /// Maximum number of live handles this pool may retain.
    #[cfg(any(test, feature = "tracy"))]
    capacity: usize,
    /// Whether profiling was available at startup and replacement handles should be attempted.
    #[cfg(any(test, feature = "tracy"))]
    enabled: bool,
    /// Next profiler-frame order assigned to a non-empty frame.
    next_frame_order: u64,
    /// Newest profiler-frame order already published to diagnostics.
    latest_published_frame_order: u64,
}

impl<T> GpuProfilerPool<T> {
    /// Creates a profiler pool with an optional initial active handle.
    pub(super) fn new(initial: Option<T>, capacity: usize) -> Self {
        #[cfg(any(test, feature = "tracy"))]
        let enabled = initial.is_some();
        #[cfg(not(any(test, feature = "tracy")))]
        let _ = capacity;
        Self {
            active: initial,
            checked_out_active: false,
            ready: Vec::new(),
            pending_submit_end: VecDeque::new(),
            #[cfg(any(test, feature = "tracy"))]
            capacity,
            #[cfg(any(test, feature = "tracy"))]
            enabled,
            next_frame_order: 1,
            latest_published_frame_order: 0,
        }
    }

    /// Returns whether replacement handles may be created for this pool.
    #[cfg(feature = "tracy")]
    #[inline]
    pub(super) const fn enabled(&self) -> bool {
        self.enabled
    }

    /// Returns the active profiler handle, when available.
    #[inline]
    pub(super) fn active(&self) -> Option<&T> {
        self.active.as_ref()
    }

    /// Returns the active profiler handle mutably, when available.
    #[inline]
    pub(super) fn active_mut(&mut self) -> Option<&mut T> {
        self.active.as_mut()
    }

    /// Returns mutable access to handles that are ready for reuse.
    #[cfg(feature = "tracy")]
    #[inline]
    pub(super) fn ready_mut(&mut self) -> &mut [T] {
        self.ready.as_mut_slice()
    }

    /// Temporarily removes the active handle for nested recording code.
    #[inline]
    pub(super) fn take_active(&mut self) -> Option<T> {
        let active = self.active.take();
        if active.is_some() {
            self.checked_out_active = true;
        }
        active
    }

    /// Restores a handle previously returned by [`Self::take_active`].
    pub(super) fn restore_active(&mut self, profiler: Option<T>) {
        let Some(profiler) = profiler else {
            return;
        };
        self.checked_out_active = false;
        if self.active.is_none() {
            self.active = Some(profiler);
        } else {
            self.ready.push(profiler);
        }
    }

    /// Moves a submitted active profiler frame into the pending-submit queue.
    pub(super) fn defer_active_until_submit(
        &mut self,
        submit_token: SubmitToken,
        frame_order: u64,
    ) -> bool {
        let Some(profiler) = self.active.take() else {
            return false;
        };
        self.pending_submit_end.push_back(PendingGpuProfilerEnd {
            submit_token,
            frame_order,
            profiler,
        });
        true
    }

    /// Returns the oldest submit token that still needs a profiler frame end.
    #[inline]
    pub(super) fn front_pending_submit_token(&self) -> Option<SubmitToken> {
        self.pending_submit_end
            .front()
            .map(|pending| pending.submit_token)
    }

    /// Pops the oldest pending-submit profiler frame.
    #[inline]
    pub(super) fn pop_front_pending_submit(&mut self) -> Option<PendingGpuProfilerEnd<T>> {
        self.pending_submit_end.pop_front()
    }

    /// Pushes a handle whose frame has ended back into the ready pool.
    #[inline]
    pub(super) fn push_ready(&mut self, profiler: T) {
        self.ready.push(profiler);
    }

    /// Activates a ready handle when the active slot is empty.
    #[cfg(any(test, feature = "tracy"))]
    pub(super) fn activate_ready(&mut self) -> bool {
        if self.active.is_some() || self.checked_out_active {
            return false;
        }
        let Some(profiler) = self.ready.pop() else {
            return false;
        };
        self.active = Some(profiler);
        true
    }

    /// Returns whether a new replacement handle may be allocated.
    #[cfg(any(test, feature = "tracy"))]
    #[inline]
    pub(super) fn can_allocate_replacement(&self) -> bool {
        self.enabled
            && self.active.is_none()
            && !self.checked_out_active
            && self.live_handle_count() < self.capacity
    }

    /// Inserts a newly allocated replacement handle into the active slot.
    #[cfg(any(test, feature = "tracy"))]
    #[inline]
    pub(super) fn insert_allocated_active(&mut self, profiler: T) -> bool {
        if !self.can_allocate_replacement() {
            return false;
        }
        self.active = Some(profiler);
        true
    }

    /// Returns whether any profiler frame is waiting for driver-thread submit completion.
    #[inline]
    pub(super) fn has_pending_submit_end(&self) -> bool {
        !self.pending_submit_end.is_empty()
    }

    /// Assigns and returns a monotonic profiler-frame order.
    #[inline]
    pub(super) fn allocate_frame_order(&mut self) -> u64 {
        let order = self.next_frame_order;
        self.next_frame_order = self.next_frame_order.saturating_add(1);
        order
    }

    /// Newest profiler-frame order already published to diagnostics.
    #[inline]
    pub(super) const fn latest_published_frame_order(&self) -> u64 {
        self.latest_published_frame_order
    }

    /// Records the newest profiler-frame order published to diagnostics.
    #[cfg(any(test, feature = "tracy"))]
    #[inline]
    pub(super) fn mark_published_frame_order(&mut self, frame_order: u64) {
        self.latest_published_frame_order = self.latest_published_frame_order.max(frame_order);
    }

    /// Returns the number of live handles owned or temporarily checked out by the pool.
    #[cfg(any(test, feature = "tracy"))]
    #[inline]
    pub(super) fn live_handle_count(&self) -> usize {
        self.active.is_some() as usize
            + self.checked_out_active as usize
            + self.ready.len()
            + self.pending_submit_end.len()
    }
}

/// Long-lived state used when handing recorded command buffers to the driver thread.
pub(super) struct GpuSubmissionState {
    /// Declared first so the driver thread shuts down before timing/profiler handles are dropped.
    pub(super) driver_thread: DriverThread,
    /// Debug HUD CPU/GPU frame timing accumulator.
    pub(super) frame_timing: FrameCpuGpuTimingHandle,
    /// Real-GPU-timestamp factory for the debug HUD's `gpu_frame_ms`. Always present; whether it
    /// produces sessions depends on the adapter feature set ([`FrameBracket::enabled`]).
    pub(super) frame_bracket: FrameBracket,
    /// GPU timestamp profiler handles for the Tracy timeline.
    pub(super) gpu_profiler_pool: GpuProfilerPool<crate::profiling::GpuProfilerHandle>,
    /// Last submit token recorded for the current app-driver frame tick. Zero means none.
    pub(super) last_frame_submit_token: AtomicU64,
    /// Flattened per-pass GPU timings and query stats from the most recently drained profiling frame.
    pub(super) latest_gpu_profiler_snapshot: Arc<Mutex<crate::profiling::GpuProfilerSnapshot>>,
}

impl GpuSubmissionState {
    /// Creates a submission state bundle from already-initialized runtime handles.
    pub(super) fn new(
        driver_thread: DriverThread,
        frame_timing: FrameCpuGpuTimingHandle,
        frame_bracket: FrameBracket,
        gpu_profiler: Option<crate::profiling::GpuProfilerHandle>,
        latest_gpu_profiler_snapshot: Arc<Mutex<crate::profiling::GpuProfilerSnapshot>>,
    ) -> Self {
        Self {
            driver_thread,
            frame_timing,
            frame_bracket,
            gpu_profiler_pool: GpuProfilerPool::new(gpu_profiler, GPU_PROFILER_POOL_CAPACITY),
            last_frame_submit_token: AtomicU64::new(0),
            latest_gpu_profiler_snapshot,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{GPU_PROFILER_POOL_CAPACITY, GpuProfilerPool};
    use crate::gpu::driver_thread::SubmitToken;

    #[test]
    fn deferring_active_frame_leaves_pool_ready_for_replacement() {
        let mut pool = GpuProfilerPool::new(Some(10_u32), GPU_PROFILER_POOL_CAPACITY);
        let frame_order = pool.allocate_frame_order();

        assert!(pool.defer_active_until_submit(SubmitToken::new(1), frame_order));

        assert!(pool.active().is_none());
        assert!(pool.can_allocate_replacement());
        assert_eq!(pool.live_handle_count(), 1);
    }

    #[test]
    fn pending_frames_close_in_submit_token_order() {
        let mut pool = GpuProfilerPool::new(Some(10_u32), GPU_PROFILER_POOL_CAPACITY);
        assert!(pool.defer_active_until_submit(SubmitToken::new(1), 1));
        assert!(pool.insert_allocated_active(20));
        assert!(pool.defer_active_until_submit(SubmitToken::new(2), 2));

        assert_eq!(
            pool.front_pending_submit_token().map(SubmitToken::raw),
            Some(1)
        );
        assert_eq!(
            pool.pop_front_pending_submit()
                .expect("front pending frame is ready")
                .frame_order,
            1
        );
        assert_eq!(
            pool.front_pending_submit_token().map(SubmitToken::raw),
            Some(2)
        );
        assert_eq!(
            pool.pop_front_pending_submit()
                .expect("second pending frame is now at the front")
                .frame_order,
            2
        );
    }

    #[test]
    fn pool_exhaustion_prevents_replacement_without_blocking() {
        let mut pool = GpuProfilerPool::new(Some(1_u32), 2);
        assert!(pool.defer_active_until_submit(SubmitToken::new(1), 1));
        assert!(pool.insert_allocated_active(2));
        assert!(pool.defer_active_until_submit(SubmitToken::new(2), 2));

        assert!(!pool.can_allocate_replacement());
        assert!(!pool.insert_allocated_active(3));
        assert_eq!(pool.live_handle_count(), 2);
    }

    #[test]
    fn ready_profiler_reactivates_before_new_allocation() {
        let mut pool = GpuProfilerPool::new(Some(1_u32), 3);
        assert!(pool.defer_active_until_submit(SubmitToken::new(1), 1));
        let pending = pool
            .pop_front_pending_submit()
            .expect("pending frame is submitted");
        pool.push_ready(pending.profiler);

        assert!(pool.activate_ready());

        assert_eq!(pool.active(), Some(&1));
        assert_eq!(pool.live_handle_count(), 1);
    }

    #[test]
    fn published_frame_order_only_moves_forward() {
        let mut pool = GpuProfilerPool::<u32>::new(None, 3);

        pool.mark_published_frame_order(8);
        pool.mark_published_frame_order(4);

        assert_eq!(pool.latest_published_frame_order(), 8);
    }
}
