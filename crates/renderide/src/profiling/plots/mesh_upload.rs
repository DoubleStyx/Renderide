//! Tracy plots for mesh upload staging and derived stream work.

use super::tracy_plot::tracy_plot;

/// Mesh upload staging counters emitted as Tracy plots.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct MeshUploadBatchProfileSample {
    /// Number of queued buffer writes drained.
    pub(crate) writes: usize,
    /// Total payload bytes drained.
    pub(crate) bytes: usize,
    /// Writes served by staging-buffer copy commands.
    pub(crate) staged_writes: usize,
    /// Writes replayed through queue writes.
    pub(crate) fallback_writes: usize,
    /// Required staging bytes for aligned writes.
    pub(crate) staging_bytes: u64,
    /// Number of copy commands recorded.
    pub(crate) copy_ops: usize,
    /// Writes replayed because the queue gate was busy.
    pub(crate) queue_gate_fallbacks: usize,
    /// Adjacent writes merged before staging or queue fallback replay.
    pub(crate) coalesced_writes: usize,
}

/// Records one mesh upload batch flush.
pub(crate) fn plot_mesh_upload_batch(sample: &MeshUploadBatchProfileSample) {
    tracy_plot!("mesh_upload::writes", sample.writes as f64);
    tracy_plot!("mesh_upload::bytes", sample.bytes as f64);
    tracy_plot!("mesh_upload::staged_writes", sample.staged_writes as f64);
    tracy_plot!(
        "mesh_upload::fallback_writes",
        sample.fallback_writes as f64
    );
    tracy_plot!("mesh_upload::staging_bytes", sample.staging_bytes as f64);
    tracy_plot!("mesh_upload::copy_ops", sample.copy_ops as f64);
    tracy_plot!(
        "mesh_upload::queue_gate_fallbacks",
        sample.queue_gate_fallbacks as f64
    );
    tracy_plot!(
        "mesh_upload::coalesced_writes",
        sample.coalesced_writes as f64
    );
}

/// Records derived stream demand and dirty masks as raw bit patterns and popcounts.
pub(crate) fn plot_mesh_derived_stream_masks(demand_bits: u16, dirty_bits: u16) {
    tracy_plot!("mesh_upload::derived_demand_mask", demand_bits as f64);
    tracy_plot!("mesh_upload::derived_dirty_mask", dirty_bits as f64);
    tracy_plot!(
        "mesh_upload::derived_demand_streams",
        demand_bits.count_ones() as f64
    );
    tracy_plot!(
        "mesh_upload::derived_dirty_streams",
        dirty_bits.count_ones() as f64
    );
}
