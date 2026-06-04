//! Tracy plots for mesh upload staging and derived stream work.

use crate::backend::asset_transfers::MeshUploadBatchStats;

use super::tracy_plot::tracy_plot;

/// Records one mesh upload batch flush.
pub(crate) fn plot_mesh_upload_batch(stats: &MeshUploadBatchStats) {
    tracy_plot!("mesh_upload::writes", stats.writes as f64);
    tracy_plot!("mesh_upload::bytes", stats.bytes as f64);
    tracy_plot!("mesh_upload::staged_writes", stats.staged_writes as f64);
    tracy_plot!("mesh_upload::fallback_writes", stats.fallback_writes as f64);
    tracy_plot!("mesh_upload::staging_bytes", stats.staging_bytes as f64);
    tracy_plot!("mesh_upload::copy_ops", stats.copy_ops as f64);
    tracy_plot!(
        "mesh_upload::queue_gate_fallbacks",
        stats.queue_gate_fallbacks as f64
    );
    tracy_plot!(
        "mesh_upload::coalesced_writes",
        stats.coalesced_writes as f64
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
