//! Per-view parallel command encoding for multi-view graph batches.
//!
//! The executor prepares immutable per-view work items on the main thread. Automatic command
//! recording only fans those items out when multi-view draw work crosses the command-recording
//! threshold; explicit diagnostics overrides can still force across-view recording or scheduler
//! in-view recording. Finished command buffers are reassembled in input order before the single
//! [`wgpu::Queue::submit`] call, preserving deterministic submit order for swapchain, VR, HUD, and
//! secondary render-texture workloads.
//!
//! The implementation relies on the following concurrency-safe pieces:
//!
//! - `record(&self, ...)` on every pass trait, plus `Send + Sync` pass trait bounds.
//! - [`crate::render_graph::FrameUploadBatch`] plus scoped upload sinks for deferred buffer
//!   writes drained on the main thread before submit.
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
    use crate::materials::EmbeddedMaterialBindResources;
    use crate::materials::MaterialPipelineCache;
    use crate::occlusion::OcclusionGraphHook;

    fn assert_send_sync<T: Send + Sync + ?Sized>() {}

    #[test]
    fn per_view_parallel_primitives_are_send_sync() {
        assert_send_sync::<EmbeddedMaterialBindResources>();
        assert_send_sync::<MaterialPipelineCache>();
        assert_send_sync::<dyn OcclusionGraphHook>();
    }
}
