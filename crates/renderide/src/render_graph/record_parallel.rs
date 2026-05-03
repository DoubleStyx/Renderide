//! Per-view parallel command encoding under [`crate::config::RecordParallelism::PerViewParallel`].
//!
//! The executor now prepares immutable per-view work items on the main thread, then fans them out
//! with `rayon::scope` so each worker records one view's command encoder independently. The
//! resulting command buffers are reassembled in input order before the existing single
//! [`wgpu::Queue::submit`] call, preserving deterministic submit order for swapchain, VR, HUD, and
//! secondary render-texture workloads.
//!
//! The landed implementation relies on the following concurrency-safe pieces:
//!
//! - `record(&self, ...)` on every pass trait, plus `Send + Sync` pass trait bounds.
//! - [`crate::render_graph::FrameUploadBatch`] for deferred `Queue::write_buffer` calls drained on
//!   the main thread before submit.
//! - Pre-resolved transient textures and buffers cloned per view before imported resources are
//!   overlaid.
//! - Pre-synchronized shared frame resources (`FrameGpuResources`) per unique view layout before
//!   any worker starts recording.
//! - Per-view `OcclusionSystem` slots, per-view per-draw slabs, and per-view scratch storage so
//!   workers only contend on their own view-local mutexes.
//! - Mutex-wrapped pipeline and embedded-material caches for lazy cache hits and rare misses.
//! - Hoisted GPU-profiler ownership: workers borrow one shared profiler handle for timestamp
//!   queries, and query resolution is encoded once on the main thread after all workers finish.

#[cfg(test)]
mod tests {
    use crate::backend::FrameResourceManager;
    use crate::materials::EmbeddedMaterialBindResources;
    use crate::materials::MaterialPipelineCache;
    use crate::occlusion::OcclusionSystem;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn per_view_parallel_primitives_are_send_sync() {
        assert_send_sync::<EmbeddedMaterialBindResources>();
        assert_send_sync::<FrameResourceManager>();
        assert_send_sync::<MaterialPipelineCache>();
        assert_send_sync::<OcclusionSystem>();
    }
}
