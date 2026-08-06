//! Tracy plots for retained render-world cache maintenance.
//!
//! Plot names emitted here are an external contract with the Tracy GUI and dashboards; do not
//! rename them.

use super::tracy_plot::tracy_plot;

/// Retained render-world maintenance counters emitted as Tracy plots.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RenderWorldMaintenanceProfileSample {
    /// Renderer records dirtied by topology or renderer-state changes this frame.
    pub topology_dirty_count: usize,
    /// Renderer records dirtied by material override changes this frame.
    pub material_dirty_count: usize,
    /// Renderer records dirtied only by transform or bounds changes this frame.
    pub transform_only_dirty_count: usize,
    /// Unique transform-root node ids consumed while expanding deferred scene changes.
    pub transform_root_dirty_count: usize,
    /// Retained node-index entries scanned while expanding transform-root dirties.
    pub transform_root_scanned_node_count: usize,
    /// Renderer records found by transform-root dirty expansion.
    pub transform_root_expanded_renderer_count: usize,
    /// Transform-root dirties that covered an entire retained render space.
    pub transform_root_full_space_count: usize,
    /// Renderer records dirtied by mesh-asset mutations this frame.
    pub mesh_asset_dirty_renderer_count: usize,
    /// Skinned renderer records whose deformation eligibility may have changed.
    pub deformation_dirty_renderer_count: usize,
    /// Renderer records whose retained templates were requested dirty this frame.
    pub dirty_renderer_count: usize,
    /// Renderer records whose retained bounds were requested dirty this frame.
    pub bounds_dirty_renderer_count: usize,
    /// Renderer records whose retained bounds were refreshed this frame.
    pub bounds_refreshed_renderer_count: usize,
    /// Renderer records actually refreshed this frame.
    pub refreshed_renderer_count: usize,
    /// Draw templates regenerated while refreshing dirty renderer records.
    pub refreshed_template_count: usize,
    /// Mesh asset ids consumed from the mesh-pool mutation log this frame.
    pub mesh_asset_invalidation_count: usize,
    /// Mesh mutations whose draw-preparation-relevant state changed.
    pub mesh_asset_draw_prep_change_count: usize,
    /// Mesh mutations suppressed because their draw-preparation-relevant state was unchanged.
    pub mesh_asset_draw_prep_noop_count: usize,
    /// Full render-world rebuild requests processed this frame.
    pub full_world_rebuild_count: usize,
    /// Prepared snapshots rebuilt only because generated particle meshes changed.
    pub particle_snapshot_rebuild_count: usize,
    /// Generated particle renderer ranges patched directly in the prepared snapshot.
    pub particle_renderer_patch_count: usize,
    /// Fresh generated particle draws copied by direct range patches.
    pub particle_patch_draw_count: usize,
    /// Direct particle updates whose prepared metadata needed structural reconstruction.
    pub particle_patch_structural_rebuild_count: usize,
    /// Static/skinned renderer ranges patched directly in the prepared snapshot.
    pub mesh_renderer_patch_count: usize,
    /// Static/skinned renderer patch candidates suppressed as semantic no-ops.
    pub mesh_renderer_patch_noop_count: usize,
    /// Fresh static/skinned draw rows considered by direct range patching.
    pub mesh_patch_draw_count: usize,
    /// Static/skinned range patches that rebuilt prepared metadata.
    pub mesh_patch_structural_rebuild_count: usize,
    /// Prepared-snapshot copy tasks built while rebuilding retained templates.
    pub snapshot_rebuild_task_count: usize,
    /// Retained draw templates considered while rebuilding prepared snapshots.
    pub snapshot_retained_draw_count: usize,
    /// Render spaces reused from the previous prepared snapshot during a partial rebuild.
    pub snapshot_reused_space_count: usize,
    /// Prepared spatial indexes rebuilt because run membership changed.
    pub spatial_rebuild_count: usize,
    /// Prepared spatial indexes refit because dynamic bounds changed.
    pub spatial_refit_count: usize,
    /// Retained draw templates currently cached after maintenance.
    pub retained_template_count: usize,
    /// Render-world caches serving contexts with no draw-prep overrides.
    pub context_invariant_count: usize,
    /// Lightweight prepared overlays serving contexts with exact overrides.
    pub context_overlay_count: usize,
    /// Base prepared snapshots synchronized into context overlays.
    pub context_overlay_sync_count: usize,
    /// Exact override renderer ranges patched into context overlays.
    pub context_override_patch_count: usize,
    /// Frames where this render world proved its retained snapshot did not need rebuilding.
    pub steady_state_skip_count: usize,
}

/// Records retained render-world dirty, rebuild, and spatial maintenance counters.
pub fn plot_render_world_maintenance(sample: RenderWorldMaintenanceProfileSample) {
    tracy_plot!(
        "render_world::topology_dirty",
        sample.topology_dirty_count as f64
    );
    tracy_plot!(
        "render_world::material_dirty",
        sample.material_dirty_count as f64
    );
    tracy_plot!(
        "render_world::transform_only_dirty",
        sample.transform_only_dirty_count as f64
    );
    tracy_plot!(
        "render_world::transform_root_dirty",
        sample.transform_root_dirty_count as f64
    );
    tracy_plot!(
        "render_world::transform_root_scanned_nodes",
        sample.transform_root_scanned_node_count as f64
    );
    tracy_plot!(
        "render_world::transform_root_expanded_renderers",
        sample.transform_root_expanded_renderer_count as f64
    );
    tracy_plot!(
        "render_world::transform_root_full_space",
        sample.transform_root_full_space_count as f64
    );
    tracy_plot!(
        "render_world::mesh_asset_dirty",
        sample.mesh_asset_dirty_renderer_count as f64
    );
    tracy_plot!(
        "render_world::deformation_dirty",
        sample.deformation_dirty_renderer_count as f64
    );
    tracy_plot!(
        "render_world::dirty_renderers",
        sample.dirty_renderer_count as f64
    );
    tracy_plot!(
        "render_world::bounds_dirty_renderers",
        sample.bounds_dirty_renderer_count as f64
    );
    tracy_plot!(
        "render_world::bounds_refreshed_renderers",
        sample.bounds_refreshed_renderer_count as f64
    );
    tracy_plot!(
        "render_world::refreshed_renderers",
        sample.refreshed_renderer_count as f64
    );
    tracy_plot!(
        "render_world::refreshed_templates",
        sample.refreshed_template_count as f64
    );
    tracy_plot!(
        "render_world::mesh_asset_invalidations",
        sample.mesh_asset_invalidation_count as f64
    );
    tracy_plot!(
        "render_world::mesh_draw_prep_changes",
        sample.mesh_asset_draw_prep_change_count as f64
    );
    tracy_plot!(
        "render_world::mesh_draw_prep_noops",
        sample.mesh_asset_draw_prep_noop_count as f64
    );
    tracy_plot!(
        "render_world::full_rebuild",
        sample.full_world_rebuild_count as f64
    );
    tracy_plot!(
        "render_world::particle_snapshot_rebuild",
        sample.particle_snapshot_rebuild_count as f64
    );
    tracy_plot!(
        "render_world::particle_renderer_patches",
        sample.particle_renderer_patch_count as f64
    );
    tracy_plot!(
        "render_world::particle_patch_draws",
        sample.particle_patch_draw_count as f64
    );
    tracy_plot!(
        "render_world::particle_patch_structural_rebuild",
        sample.particle_patch_structural_rebuild_count as f64
    );
    tracy_plot!(
        "render_world::mesh_renderer_patches",
        sample.mesh_renderer_patch_count as f64
    );
    tracy_plot!(
        "render_world::mesh_renderer_patch_noops",
        sample.mesh_renderer_patch_noop_count as f64
    );
    tracy_plot!(
        "render_world::mesh_patch_draws",
        sample.mesh_patch_draw_count as f64
    );
    tracy_plot!(
        "render_world::mesh_patch_structural_rebuild",
        sample.mesh_patch_structural_rebuild_count as f64
    );
    tracy_plot!(
        "render_world::snapshot_tasks",
        sample.snapshot_rebuild_task_count as f64
    );
    tracy_plot!(
        "render_world::snapshot_retained_draws",
        sample.snapshot_retained_draw_count as f64
    );
    tracy_plot!(
        "render_world::snapshot_reused_spaces",
        sample.snapshot_reused_space_count as f64
    );
    tracy_plot!(
        "render_world::spatial_rebuild",
        sample.spatial_rebuild_count as f64
    );
    tracy_plot!(
        "render_world::spatial_refit",
        sample.spatial_refit_count as f64
    );
    tracy_plot!(
        "render_world::retained_templates",
        sample.retained_template_count as f64
    );
    tracy_plot!(
        "render_world::context_invariant",
        sample.context_invariant_count as f64
    );
    tracy_plot!(
        "render_world::context_overlay",
        sample.context_overlay_count as f64
    );
    tracy_plot!(
        "render_world::context_overlay_sync",
        sample.context_overlay_sync_count as f64
    );
    tracy_plot!(
        "render_world::context_override_patches",
        sample.context_override_patch_count as f64
    );
    tracy_plot!(
        "render_world::steady_state_skip",
        sample.steady_state_skip_count as f64
    );
}
