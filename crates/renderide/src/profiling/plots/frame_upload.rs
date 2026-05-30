//! Tracy plots for per-frame upload traffic and world-mesh batch compression.
//!
//! Plot names emitted here are an external contract with the Tracy GUI and dashboards; do not
//! rename them.

use super::tracy_plot::tracy_plot;

/// Persistent upload arena pressure and fallback counters for one frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameUploadArenaProfileSample {
    /// Writes replayed through [`wgpu::Queue::write_buffer`] because staging was not usable.
    pub fallback_writes: usize,
    /// Bytes staged through persistent arena slots.
    pub persistent_staging_bytes: u64,
    /// Persistent arena slot reuse count.
    pub persistent_slot_reuses: usize,
    /// Persistent arena slot allocation or growth count.
    pub persistent_slot_grows: usize,
    /// Bytes staged through temporary fallback buffers.
    pub temporary_staging_bytes: u64,
    /// Temporary fallback count caused by unavailable persistent slots.
    pub temporary_staging_fallbacks: usize,
    /// Staged writes replayed through queue writes because no staging buffer fit.
    pub oversized_queue_fallback_writes: usize,
    /// Total bytes allocated across persistent arena slots.
    pub arena_capacity_bytes: u64,
    /// Persistent arena slots mapped and free.
    pub arena_free_slots: usize,
    /// Persistent arena slots referenced by submitted GPU work.
    pub arena_in_flight_slots: usize,
    /// Persistent arena slots waiting for remap completion.
    pub arena_remapping_slots: usize,
}

/// Records, per call to `crate::passes::world_mesh_forward::encode::draw_subset`,
/// how many instance batches and how many input draws were submitted in that subpass.
///
/// One sample lands on the Tracy timeline per opaque or intersection subpass record, so the
/// plot trace shows fragmentation visually: when batches ~= draws, the merge isn't compressing;
/// when batches << draws, instancing is collapsing same-mesh runs as intended. Pair with
/// [`crate::world_mesh::WorldMeshDrawStats::gpu_instances_emitted`] in the HUD for a
/// per-frame integral. Expands to nothing when the `tracy` feature is off.
#[inline]
pub fn plot_world_mesh_subpass(batches: usize, draws: usize) {
    tracy_plot!("world_mesh::subpass_batches", batches as f64);
    tracy_plot!("world_mesh::subpass_draws", draws as f64);
}

/// Records deferred queue-write traffic for one frame.
#[inline]
pub fn plot_frame_upload_batch(writes: usize, bytes: usize) {
    tracy_plot!("frame_upload::writes", writes as f64);
    tracy_plot!("frame_upload::bytes", bytes as f64);
}

/// Records persistent upload arena pressure and fallback counters for one frame.
#[inline]
pub fn plot_frame_upload_arena(sample: &FrameUploadArenaProfileSample) {
    tracy_plot!(
        "frame_upload::fallback_writes",
        sample.fallback_writes as f64
    );
    tracy_plot!(
        "frame_upload::persistent_staging_bytes",
        sample.persistent_staging_bytes as f64
    );
    tracy_plot!(
        "frame_upload::persistent_slot_reuses",
        sample.persistent_slot_reuses as f64
    );
    tracy_plot!(
        "frame_upload::persistent_slot_grows",
        sample.persistent_slot_grows as f64
    );
    tracy_plot!(
        "frame_upload::temporary_staging_bytes",
        sample.temporary_staging_bytes as f64
    );
    tracy_plot!(
        "frame_upload::temporary_staging_fallbacks",
        sample.temporary_staging_fallbacks as f64
    );
    tracy_plot!(
        "frame_upload::oversized_queue_fallback_writes",
        sample.oversized_queue_fallback_writes as f64
    );
    tracy_plot!(
        "frame_upload::arena_capacity_bytes",
        sample.arena_capacity_bytes as f64
    );
    tracy_plot!(
        "frame_upload::arena_free_slots",
        sample.arena_free_slots as f64
    );
    tracy_plot!(
        "frame_upload::arena_in_flight_slots",
        sample.arena_in_flight_slots as f64
    );
    tracy_plot!(
        "frame_upload::arena_remapping_slots",
        sample.arena_remapping_slots as f64
    );
}
