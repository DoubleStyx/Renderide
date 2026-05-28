//! Shared CPU parallelism admission policy for frame-critical renderer work.
//!
//! The renderer has many small opportunities to use Rayon, but spawning workers for two tiny
//! independent items can cost more than the work itself. This module keeps the frame hot paths on
//! one set of thresholds so view, graph, and draw-preparation stages make consistent decisions.

/// Minimum useful chunks before a Rayon fan-out is allowed.
pub(crate) const MIN_PARALLEL_CHUNKS: usize = 2;

/// Baseline draw count where view-level frame work is usually large enough for Rayon.
const DRAW_HEAVY_PARALLEL_BASE_DRAWS: usize = 512;

/// Additional draw count per worker used to scale the draw-heavy gate on larger machines.
const DRAW_HEAVY_PARALLEL_DRAWS_PER_WORKER: usize = 128;

/// Admission decision for one parallel work site.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ParallelAdmission {
    /// Run the work on the caller thread.
    Serial,
    /// Run the work through Rayon with the supplied minimum chunk size.
    Parallel {
        /// Minimum number of domain items owned by one worker split.
        chunk_size: usize,
    },
}

impl ParallelAdmission {
    /// Returns `true` when the work site should use Rayon.
    pub(crate) const fn is_parallel(self) -> bool {
        matches!(self, Self::Parallel { .. })
    }

    /// Returns the admitted chunk size, or `None` for serial execution.
    pub(crate) const fn chunk_size(self) -> Option<usize> {
        match self {
            Self::Serial => None,
            Self::Parallel { chunk_size } => Some(chunk_size),
        }
    }
}

/// Compact description of one frame-critical CPU work site.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FrameCpuWorkload {
    /// Independent view count represented by the work.
    view_count: usize,
    /// Estimated draw or renderer count represented by the work.
    total_draw_count: usize,
    /// Independent domain item count available for worker splits.
    independent_item_count: usize,
}

impl FrameCpuWorkload {
    /// Creates a workload from explicit view, draw, and independent-item counts.
    pub(crate) const fn new(
        view_count: usize,
        total_draw_count: usize,
        independent_item_count: usize,
    ) -> Self {
        Self {
            view_count,
            total_draw_count,
            independent_item_count,
        }
    }

    /// Creates a workload for independent non-draw items such as render contexts.
    pub(crate) const fn independent_items(item_count: usize) -> Self {
        Self::new(0, 0, item_count)
    }

    /// Creates a workload for view-owned draw work.
    pub(crate) const fn view_draws(view_count: usize, total_draw_count: usize) -> Self {
        Self::new(view_count, total_draw_count, view_count)
    }

    /// Independent domain item count available for worker splits.
    pub(crate) const fn independent_item_count(self) -> usize {
        self.independent_item_count
    }

    /// Independent view count represented by the work.
    pub(crate) const fn view_count(self) -> usize {
        self.view_count
    }

    /// Estimated draw or renderer count represented by the work.
    pub(crate) const fn total_draw_count(self) -> usize {
        self.total_draw_count
    }
}

/// Per-frame CPU parallelism thresholds derived from the active Rayon worker count.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FrameParallelPolicy {
    /// Number of Rayon workers available to frame-critical work.
    worker_count: usize,
}

impl FrameParallelPolicy {
    /// Builds a policy from Rayon worker count.
    pub(crate) const fn new(worker_count: usize) -> Self {
        Self {
            worker_count: if worker_count == 0 { 1 } else { worker_count },
        }
    }

    /// Builds a policy for the currently executing Rayon pool.
    pub(crate) fn for_current_thread_pool() -> Self {
        Self::new(rayon::current_num_threads())
    }

    /// Draw count required before view-level frame work is considered heavy.
    pub(crate) const fn draw_heavy_threshold(self) -> usize {
        let scaled = self
            .worker_count
            .saturating_mul(DRAW_HEAVY_PARALLEL_DRAWS_PER_WORKER);
        if scaled > DRAW_HEAVY_PARALLEL_BASE_DRAWS {
            scaled
        } else {
            DRAW_HEAVY_PARALLEL_BASE_DRAWS
        }
    }

    /// Minimum independent non-draw items required before worker fan-out is useful.
    pub(crate) const fn independent_item_threshold(self) -> usize {
        if self.worker_count <= 1 {
            usize::MAX
        } else if self.worker_count >= 8 {
            4
        } else if self.worker_count >= 4 {
            3
        } else {
            MIN_PARALLEL_CHUNKS
        }
    }

    /// Returns `true` when the estimated draw work crosses the draw-heavy gate.
    pub(crate) const fn is_draw_heavy(self, total_draw_count: usize) -> bool {
        self.worker_count > 1 && total_draw_count >= self.draw_heavy_threshold()
    }

    /// Decides whether independent item work should fan out through Rayon.
    pub(crate) const fn admit_independent_items(
        self,
        workload: FrameCpuWorkload,
        chunk_size: usize,
    ) -> ParallelAdmission {
        let chunk_size = if chunk_size == 0 { 1 } else { chunk_size };
        let enough_chunks =
            workload.independent_item_count() >= chunk_size.saturating_mul(MIN_PARALLEL_CHUNKS);
        let enough_items = workload.independent_item_count() >= self.independent_item_threshold();
        if self.worker_count > 1 && enough_chunks && enough_items {
            ParallelAdmission::Parallel { chunk_size }
        } else {
            ParallelAdmission::Serial
        }
    }

    /// Decides whether view-level draw work should fan out through Rayon.
    pub(crate) const fn admit_draw_heavy_views(
        self,
        workload: FrameCpuWorkload,
        chunk_size: usize,
    ) -> ParallelAdmission {
        let chunk_size = if chunk_size == 0 { 1 } else { chunk_size };
        let enough_views = workload.view_count() >= chunk_size.saturating_mul(MIN_PARALLEL_CHUNKS);
        if enough_views && self.is_draw_heavy(workload.total_draw_count()) {
            ParallelAdmission::Parallel { chunk_size }
        } else {
            ParallelAdmission::Serial
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{FrameCpuWorkload, FrameParallelPolicy, ParallelAdmission};

    #[test]
    fn draw_heavy_threshold_scales_with_worker_count() {
        assert_eq!(FrameParallelPolicy::new(1).draw_heavy_threshold(), 512);
        assert_eq!(FrameParallelPolicy::new(4).draw_heavy_threshold(), 512);
        assert_eq!(FrameParallelPolicy::new(8).draw_heavy_threshold(), 1024);
    }

    #[test]
    fn independent_items_require_multiple_chunks_and_worker_scaled_item_count() {
        let policy = FrameParallelPolicy::new(8);
        assert_eq!(
            policy.admit_independent_items(FrameCpuWorkload::independent_items(2), 1),
            ParallelAdmission::Serial
        );
        assert!(
            policy
                .admit_independent_items(FrameCpuWorkload::independent_items(4), 1)
                .is_parallel()
        );
    }

    #[test]
    fn draw_heavy_views_require_multiple_views_and_enough_draws() {
        let policy = FrameParallelPolicy::new(4);
        assert_eq!(
            policy.admit_draw_heavy_views(FrameCpuWorkload::view_draws(1, 4096), 1),
            ParallelAdmission::Serial
        );
        assert_eq!(
            policy.admit_draw_heavy_views(FrameCpuWorkload::view_draws(2, 511), 1),
            ParallelAdmission::Serial
        );
        assert!(
            policy
                .admit_draw_heavy_views(FrameCpuWorkload::view_draws(2, 512), 1)
                .is_parallel()
        );
    }
}
