//! Profiling adapters for retained forward-preparation caches.

use super::WorldMeshForwardInstancePlanCache;

/// Records the current retained instance-plan cache counters.
pub(super) fn plot_instance_plan_cache(cache: &WorldMeshForwardInstancePlanCache) {
    let stats = cache.stats();
    crate::profiling::plot_world_mesh_instance_plan_cache(
        crate::profiling::WorldMeshInstancePlanCacheProfileSample {
            entries: stats.entries,
            hits: stats.hits,
            misses: stats.misses,
            skipped_small: stats.skipped_small,
            skipped_thrash: stats.skipped_thrash,
            hit_rate_per_mille: stats.hit_rate_per_mille,
            insertions: stats.insertions,
            evictions: stats.evictions,
        },
    );
}
