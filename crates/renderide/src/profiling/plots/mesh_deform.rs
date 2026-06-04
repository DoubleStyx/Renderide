//! Tracy plots for mesh-deform workload and cache pressure.
//!
//! Plot names emitted here are an external contract with the Tracy GUI and dashboards; do not
//! rename them.

use super::tracy_plot::tracy_plot;

/// Mesh-deform workload and cache pressure counters for one frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct MeshDeformProfileSample {
    /// Deform work items collected for this frame.
    pub work_items: u64,
    /// Compute passes opened while recording mesh deformation.
    pub compute_passes: u64,
    /// Bind groups created while recording mesh deformation.
    pub bind_groups_created: u64,
    /// Bind groups reused from mesh-deform caches.
    pub bind_group_cache_reuses: u64,
    /// Encoder copy operations recorded by mesh deformation.
    pub copy_ops: u64,
    /// Sparse blendshape compute dispatches recorded.
    pub blend_dispatches: u64,
    /// Skinning compute dispatches recorded.
    pub skin_dispatches: u64,
    /// Work items skipped because their deform inputs were stable.
    pub stable_skips: u64,
    /// Scratch-buffer grow operations triggered by this frame.
    pub scratch_buffer_grows: u64,
    /// Work items skipped because the skin cache could not allocate safely.
    pub skipped_allocations: u64,
    /// Skin-cache entries reused.
    pub cache_reuses: u64,
    /// Skin-cache entries allocated.
    pub cache_allocations: u64,
    /// Skin-cache arena growth operations.
    pub cache_grows: u64,
    /// Prior-frame skin-cache entries evicted.
    pub cache_evictions: u64,
    /// Allocation attempts where all evictable entries were current-frame entries.
    pub cache_current_frame_eviction_refusals: u64,
}

/// Records mesh-deform workload and cache pressure counters for the current frame.
pub fn plot_mesh_deform(sample: MeshDeformProfileSample) {
    tracy_plot!("mesh_deform::work_items", sample.work_items as f64);
    tracy_plot!("mesh_deform::compute_passes", sample.compute_passes as f64);
    tracy_plot!(
        "mesh_deform::bind_groups_created",
        sample.bind_groups_created as f64
    );
    tracy_plot!(
        "mesh_deform::bind_group_cache_reuses",
        sample.bind_group_cache_reuses as f64
    );
    tracy_plot!("mesh_deform::copy_ops", sample.copy_ops as f64);
    tracy_plot!(
        "mesh_deform::blend_dispatches",
        sample.blend_dispatches as f64
    );
    tracy_plot!(
        "mesh_deform::skin_dispatches",
        sample.skin_dispatches as f64
    );
    tracy_plot!("mesh_deform::stable_skips", sample.stable_skips as f64);
    tracy_plot!(
        "mesh_deform::scratch_buffer_grows",
        sample.scratch_buffer_grows as f64
    );
    tracy_plot!(
        "mesh_deform::skipped_allocations",
        sample.skipped_allocations as f64
    );
    tracy_plot!("mesh_deform::cache_reuses", sample.cache_reuses as f64);
    tracy_plot!(
        "mesh_deform::cache_allocations",
        sample.cache_allocations as f64
    );
    tracy_plot!("mesh_deform::cache_grows", sample.cache_grows as f64);
    tracy_plot!(
        "mesh_deform::cache_evictions",
        sample.cache_evictions as f64
    );
    tracy_plot!(
        "mesh_deform::cache_current_frame_eviction_refusals",
        sample.cache_current_frame_eviction_refusals as f64
    );
}
