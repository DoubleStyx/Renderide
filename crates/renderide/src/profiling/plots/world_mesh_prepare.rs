//! Tracy plots for CPU-side world-mesh forward preparation.

use super::tracy_plot::tracy_plot;

/// Records the size of one prepared world-mesh forward view.
pub fn plot_world_mesh_prepare(draws: usize, material_packets: usize, primary_groups: usize) {
    tracy_plot!("world_mesh_prepare::draws", draws as f64);
    tracy_plot!(
        "world_mesh_prepare::material_packets",
        material_packets as f64
    );
    tracy_plot!("world_mesh_prepare::primary_groups", primary_groups as f64);
}

/// Retained forward instance-plan cache counters.
///
/// This cache is the keystone of the GPU-driven path: the GPU cull structural cache keys on
/// `Arc::ptr_eq` of the plan this cache hands out, so a miss here cascades into re-uploading the
/// whole cull candidate set. Its counters existed but were only ever read by tests.
#[derive(Clone, Copy, Debug, Default)]
pub struct WorldMeshInstancePlanCacheProfileSample {
    pub entries: usize,
    pub hits: u64,
    pub misses: u64,
    pub skipped_small: u64,
    pub skipped_thrash: u64,
    pub hit_rate_per_mille: u16,
    pub insertions: u64,
    pub evictions: u64,
}

/// Emits retained instance-plan cache state. Counters are cumulative, so read the slope.
pub fn plot_world_mesh_instance_plan_cache(sample: WorldMeshInstancePlanCacheProfileSample) {
    tracy_plot!(
        "instance_plan_cache::hit_rate_per_mille",
        f64::from(sample.hit_rate_per_mille)
    );
    tracy_plot!("instance_plan_cache::entries", sample.entries as f64);
    tracy_plot!("instance_plan_cache::hits", sample.hits as f64);
    tracy_plot!("instance_plan_cache::misses", sample.misses as f64);
    tracy_plot!(
        "instance_plan_cache::skipped_small",
        sample.skipped_small as f64
    );
    // Nonzero slope here means the thrash guard switched the cache off, which is self-reinforcing:
    // a bypassed lookup can never become a hit.
    tracy_plot!(
        "instance_plan_cache::skipped_thrash",
        sample.skipped_thrash as f64
    );
    tracy_plot!("instance_plan_cache::insertions", sample.insertions as f64);
    tracy_plot!("instance_plan_cache::evictions", sample.evictions as f64);
}
