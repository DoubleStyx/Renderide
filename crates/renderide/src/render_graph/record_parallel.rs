//! Scaffolding for per-view parallel command encoding under
//! [`crate::config::RecordParallelism::PerViewParallel`].
//!
//! ## Landed scaffolding
//!
//! - `record(&self, …)` on every pass trait ([`crate::render_graph::pass`]).
//! - [`crate::render_graph::FrameSystemsShared`] / [`crate::render_graph::FrameRenderParamsView`]
//!   split, so per-view surface state is value-typed.
//! - [`crate::render_graph::FrameUploadBatch`] drains on the main thread post-submit, so deferred
//!   `Queue::write_buffer` calls do not need shared mutable queue state.
//! - Transient textures / buffers pre-resolved once before the per-view loop
//!   ([`crate::render_graph::compiled::CompiledRenderGraph::pre_resolve_transients_for_views`]).
//! - Per-view scratch slabs (per-draw uniforms + byte slab) keyed by
//!   [`crate::backend::OcclusionViewId`] in [`crate::backend::FrameResourceManager`], so two
//!   workers cannot alias the same scratch.
//! - Per-view per-draw resources and per-view frame state pre-warmed before recording
//!   (`pre_warm_per_view_resources_for_views` in
//!   [`crate::render_graph::compiled::CompiledRenderGraph`]).
//! - [`crate::render_graph::compiled::CompiledRenderGraph::execute_pass_node`] /
//!   [`crate::render_graph::compiled::CompiledRenderGraph::encode_per_view_to_cmd`] take `&self`.
//!
//! ## Remaining blockers for full `rayon::scope` fan-out
//!
//! - Interior mutability on the [`crate::backend::OcclusionSystem`] (Hi-Z temporal capture) and
//!   on the [`crate::backend::MaterialSystem`] (`MaterialPipelineCache` LRU and
//!   `EmbeddedMaterialBindResources` `RefCell`-based caches): per-view passes today take `&mut`
//!   on these via [`crate::render_graph::FrameSystemsShared`].
//! - Gating around the singleton `GpuProfiler` take/restore pattern in
//!   [`crate::gpu::GpuContext`].
//!
//! Until those land the parallel branch logs once via [`warn_parallel_falls_back_once`] and
//! falls back to serial. The serial path still benefits from the scaffolding above (per-view
//! scratch, pre-warmed resources) so the work is not wasted.

use std::sync::atomic::{AtomicBool, Ordering};

/// One-time latch so the fallback `info!` only fires on the first frame after opt-in.
static PARALLEL_FALLBACK_LOGGED: AtomicBool = AtomicBool::new(false);

/// Logs a single `info!` the first time per-view parallel recording is requested but the fallback
/// is taken. Subsequent calls are no-ops.
///
/// `view_count` is recorded so the log reflects the VR / secondary-camera context that motivated
/// the opt-in.
pub fn warn_parallel_falls_back_once(view_count: usize) {
    if !PARALLEL_FALLBACK_LOGGED.swap(true, Ordering::Relaxed) {
        logger::info!(
            "record_parallelism = PerViewParallel requested for {view_count} views; \
             scaffolding is complete — full rayon::scope fan-out is gated on interior \
             mutability for OcclusionSystem / MaterialSystem and the GpuProfiler singleton, \
             recording serially this frame."
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exercises the one-time log latch so repeated calls do not re-log.
    #[test]
    fn warn_parallel_falls_back_once_latches_after_first_call() {
        PARALLEL_FALLBACK_LOGGED.store(false, Ordering::Relaxed);
        warn_parallel_falls_back_once(2);
        assert!(PARALLEL_FALLBACK_LOGGED.load(Ordering::Relaxed));
        warn_parallel_falls_back_once(4);
        assert!(PARALLEL_FALLBACK_LOGGED.load(Ordering::Relaxed));
    }
}
