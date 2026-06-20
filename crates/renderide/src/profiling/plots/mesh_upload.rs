//! Tracy plots for mesh upload staging and derived stream work.

use super::tracy_plot::tracy_plot;
use crate::upload_stats::UploadTrafficStats;

/// Mesh upload staging counters emitted as Tracy plots.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct MeshUploadBatchProfileSample {
    /// Queue-write and staging-copy traffic.
    pub(crate) traffic: UploadTrafficStats,
    /// Writes replayed because the queue gate was busy.
    pub(crate) queue_gate_fallbacks: usize,
    /// Adjacent writes merged before staging or queue fallback replay.
    pub(crate) coalesced_writes: usize,
}

/// Records one mesh upload batch flush.
pub(crate) fn plot_mesh_upload_batch(sample: &MeshUploadBatchProfileSample) {
    tracy_plot!("mesh_upload::writes", sample.traffic.writes as f64);
    tracy_plot!("mesh_upload::bytes", sample.traffic.bytes as f64);
    tracy_plot!(
        "mesh_upload::staged_writes",
        sample.traffic.staged_writes as f64
    );
    tracy_plot!(
        "mesh_upload::fallback_writes",
        sample.traffic.fallback_writes as f64
    );
    tracy_plot!(
        "mesh_upload::staging_bytes",
        sample.traffic.staging_bytes as f64
    );
    tracy_plot!("mesh_upload::copy_ops", sample.traffic.copy_ops as f64);
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
