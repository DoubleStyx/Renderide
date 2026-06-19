//! Tracy plots for per-frame upload traffic and world-mesh batch compression.
//!
//! Plot names emitted here are an external contract with the Tracy GUI and dashboards; do not
//! rename them.

use super::tracy_plot::tracy_plot;
use crate::upload_stats::{UploadArenaStats, UploadTrafficStats};

/// Persistent upload arena pressure and fallback counters for one frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameUploadArenaProfileSample {
    /// Queue-write and staging-copy traffic.
    pub traffic: UploadTrafficStats,
    /// Persistent upload arena acquire and pressure counters.
    pub arena: UploadArenaStats,
}

/// Records, per call to `crate::passes::world_mesh_forward::encode::draw_subset`,
/// how many instance batches and how many input draws were submitted in that subpass.
///
/// One sample lands on the Tracy timeline per opaque or intersection subpass record, so the
/// plot trace shows fragmentation visually: when batches ~= draws, the merge isn't compressing;
/// when batches << draws, instancing is collapsing same-mesh runs as intended. Pair with
/// [`crate::world_mesh::WorldMeshDrawStats::gpu_instances_emitted`] in the HUD for a
/// per-frame integral. Expands to nothing when the `tracy` feature is off.
pub fn plot_world_mesh_subpass(batches: usize, draws: usize) {
    tracy_plot!("world_mesh::subpass_batches", batches as f64);
    tracy_plot!("world_mesh::subpass_draws", draws as f64);
}

/// Records deferred queue-write traffic for one frame.
pub fn plot_frame_upload_batch(writes: usize, bytes: usize) {
    tracy_plot!("frame_upload::writes", writes as f64);
    tracy_plot!("frame_upload::bytes", bytes as f64);
}

/// Records persistent upload arena pressure and fallback counters for one frame.
pub fn plot_frame_upload_arena(sample: &FrameUploadArenaProfileSample) {
    tracy_plot!(
        "frame_upload::fallback_writes",
        sample.traffic.fallback_writes as f64
    );
    tracy_plot!(
        "frame_upload::persistent_staging_bytes",
        sample.arena.acquire.persistent_staging_bytes as f64
    );
    tracy_plot!(
        "frame_upload::persistent_slot_reuses",
        sample.arena.acquire.persistent_slot_reuses as f64
    );
    tracy_plot!(
        "frame_upload::persistent_slot_grows",
        sample.arena.acquire.persistent_slot_grows as f64
    );
    tracy_plot!(
        "frame_upload::temporary_staging_bytes",
        sample.arena.acquire.temporary_staging_bytes as f64
    );
    tracy_plot!(
        "frame_upload::temporary_staging_fallbacks",
        sample.arena.acquire.temporary_staging_fallbacks as f64
    );
    tracy_plot!(
        "frame_upload::oversized_queue_fallback_writes",
        sample.arena.acquire.oversized_queue_fallback_writes as f64
    );
    tracy_plot!(
        "frame_upload::arena_capacity_bytes",
        sample.arena.pressure.capacity_bytes as f64
    );
    tracy_plot!(
        "frame_upload::arena_free_slots",
        sample.arena.pressure.free_slots as f64
    );
    tracy_plot!(
        "frame_upload::arena_in_flight_slots",
        sample.arena.pressure.in_flight_slots as f64
    );
    tracy_plot!(
        "frame_upload::arena_remapping_slots",
        sample.arena.pressure.remapping_slots as f64
    );
}
