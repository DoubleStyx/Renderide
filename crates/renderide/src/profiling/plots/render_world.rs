//! Tracy plots for retained render-world cache maintenance.
//!
//! Plot names emitted here are an external contract with the Tracy GUI and dashboards; do not
//! rename them.

use crate::world_mesh::RenderWorldMaintenanceStats;

use super::tracy_plot::tracy_plot;

/// Records retained render-world dirty, rebuild, and spatial maintenance counters.
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
        "render_world::transform_root_dirty",
        stats.transform_root_dirty_count as f64
    );
    tracy_plot!(
        "render_world::transform_root_scanned_nodes",
        stats.transform_root_scanned_node_count as f64
    );
    tracy_plot!(
        "render_world::transform_root_expanded_renderers",
        stats.transform_root_expanded_renderer_count as f64
    );
    tracy_plot!(
        "render_world::transform_root_full_space",
        stats.transform_root_full_space_count as f64
    );
    tracy_plot!(
        "render_world::mesh_asset_dirty",
        stats.mesh_asset_dirty_renderer_count as f64
    );
    tracy_plot!(
        "render_world::dirty_renderers",
        stats.dirty_renderer_count as f64
    );
    tracy_plot!(
        "render_world::bounds_dirty_renderers",
        stats.bounds_dirty_renderer_count as f64
    );
    tracy_plot!(
        "render_world::bounds_refreshed_renderers",
        stats.bounds_refreshed_renderer_count as f64
    );
    tracy_plot!(
        "render_world::refreshed_renderers",
        stats.refreshed_renderer_count as f64
    );
    tracy_plot!(
        "render_world::refreshed_templates",
        stats.refreshed_template_count as f64
    );
    tracy_plot!(
        "render_world::mesh_asset_invalidations",
        stats.mesh_asset_invalidation_count as f64
    );
    tracy_plot!(
        "render_world::full_rebuild",
        stats.full_world_rebuild_count as f64
    );
    tracy_plot!(
        "render_world::particle_snapshot_rebuild",
        stats.particle_snapshot_rebuild_count as f64
    );
    tracy_plot!(
        "render_world::snapshot_tasks",
        stats.snapshot_rebuild_task_count as f64
    );
    tracy_plot!(
        "render_world::snapshot_retained_draws",
        stats.snapshot_retained_draw_count as f64
    );
    tracy_plot!(
        "render_world::snapshot_reused_spaces",
        stats.snapshot_reused_space_count as f64
    );
    tracy_plot!(
        "render_world::spatial_rebuild",
        stats.spatial_rebuild_count as f64
    );
    tracy_plot!(
        "render_world::spatial_refit",
        stats.spatial_refit_count as f64
    );
    tracy_plot!(
        "render_world::retained_templates",
        stats.retained_template_count as f64
    );
    tracy_plot!(
        "render_world::context_invariant",
        stats.context_invariant_count as f64
    );
    tracy_plot!(
        "render_world::steady_state_skip",
        stats.steady_state_skip_count as f64
    );
}
