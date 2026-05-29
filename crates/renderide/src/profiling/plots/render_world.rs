//! Tracy plots for retained render-world cache maintenance.
//!
//! Plot names emitted here are an external contract with the Tracy GUI and dashboards; do not
//! rename them.

use crate::world_mesh::RenderWorldMaintenanceStats;

use super::tracy_plot::tracy_plot;

/// Records retained render-world dirty, rebuild, and spatial maintenance counters.
#[inline]
pub fn plot_render_world_maintenance(stats: RenderWorldMaintenanceStats) {
    tracy_plot!(
        "render_world::topology_dirty",
        stats.topology_dirty_count as f64
    );
    tracy_plot!(
        "render_world::material_dirty",
        stats.material_dirty_count as f64
    );
    tracy_plot!(
        "render_world::transform_only_dirty",
        stats.transform_only_dirty_count as f64
    );
    tracy_plot!(
        "render_world::mesh_asset_dirty",
        stats.mesh_asset_dirty_renderer_count as f64
    );
    tracy_plot!(
        "render_world::full_rebuild",
        stats.full_world_rebuild_count as f64
    );
    tracy_plot!(
        "render_world::spatial_rebuild",
        stats.spatial_rebuild_count as f64
    );
    tracy_plot!(
        "render_world::spatial_refit",
        stats.spatial_refit_count as f64
    );
}
