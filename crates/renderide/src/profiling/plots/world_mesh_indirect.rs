//! Tracy plots for forward indirect-draw coverage and fallback reasons.
//!
//! Plot names emitted here are an external contract with the Tracy GUI and dashboards; do not
//! rename them.

use super::tracy_plot::tracy_plot;

/// Geometry-arena population coverage for one graph submission.
#[derive(Clone, Copy, Debug, Default)]
pub struct WorldMeshGeometryArenaProfileSample {
    /// Prepared draw rows inspected by the population pass.
    pub input_draws: usize,
    /// Unique mesh ids selected after excluding deformed draw rows.
    pub unique_meshes: usize,
    /// Draw rows excluded because they use skin/deform output.
    pub deformed_draws: usize,
    /// Unique renderer-generated particle meshes excluded because their geometry is dynamic.
    pub generated_meshes: usize,
    /// Other unique dynamic meshes excluded from the static arena.
    pub dynamic_meshes: usize,
    /// Unique mesh ids missing from the resident mesh pool.
    pub missing_meshes: usize,
    /// Unique resident meshes without a decomposed position stream.
    pub missing_position_meshes: usize,
    /// Eligible meshes already allocated before this population pass.
    pub already_resident_meshes: usize,
    /// Eligible meshes available in the arena after `ensure_mesh`.
    pub ready_meshes: usize,
    /// Eligible meshes that could not be allocated in the arena.
    pub allocation_failures: usize,
    /// Live arena backing-buffer capacity, including growth headroom.
    pub allocated_bytes: u64,
    /// Bytes occupied by resident geometry ranges, excluding unused buffer headroom.
    pub resident_allocation_bytes: u64,
    /// Whether a pending removal/sparse-stream reclamation request was evaluated this frame.
    pub compaction_evaluated: bool,
    /// Steady-state backing capacity eliminated by a GPU-only packed rebuild this frame.
    pub compaction_reclaimed_bytes: u64,
    /// Committed geometry bytes copied GPU-to-GPU by the packed rebuild.
    pub compaction_copy_bytes: u64,
}

/// Records geometry-arena population coverage for one graph submission.
pub fn plot_world_mesh_geometry_arena(sample: WorldMeshGeometryArenaProfileSample) {
    tracy_plot!(
        "world_mesh::geometry_arena_input_draws",
        sample.input_draws as f64
    );
    tracy_plot!(
        "world_mesh::geometry_arena_unique_meshes",
        sample.unique_meshes as f64
    );
    tracy_plot!(
        "world_mesh::geometry_arena_deformed_draws",
        sample.deformed_draws as f64
    );
    tracy_plot!(
        "world_mesh::geometry_arena_generated_meshes",
        sample.generated_meshes as f64
    );
    tracy_plot!(
        "world_mesh::geometry_arena_dynamic_meshes",
        sample.dynamic_meshes as f64
    );
    tracy_plot!(
        "world_mesh::geometry_arena_missing_meshes",
        sample.missing_meshes as f64
    );
    tracy_plot!(
        "world_mesh::geometry_arena_missing_position_meshes",
        sample.missing_position_meshes as f64
    );
    tracy_plot!(
        "world_mesh::geometry_arena_already_resident_meshes",
        sample.already_resident_meshes as f64
    );
    tracy_plot!(
        "world_mesh::geometry_arena_ready_meshes",
        sample.ready_meshes as f64
    );
    tracy_plot!(
        "world_mesh::geometry_arena_allocation_failures",
        sample.allocation_failures as f64
    );
    tracy_plot!(
        "world_mesh::geometry_arena_allocated_bytes",
        sample.allocated_bytes as f64
    );
    tracy_plot!(
        "world_mesh::geometry_arena_resident_allocation_bytes",
        sample.resident_allocation_bytes as f64
    );
    tracy_plot!(
        "world_mesh::geometry_arena_unused_capacity_bytes",
        sample
            .allocated_bytes
            .saturating_sub(sample.resident_allocation_bytes) as f64
    );
    let utilization_percent = if sample.allocated_bytes == 0 {
        0.0
    } else {
        sample.resident_allocation_bytes as f64 * 100.0 / sample.allocated_bytes as f64
    };
    tracy_plot!(
        "world_mesh::geometry_arena_utilization_percent",
        utilization_percent
    );
    tracy_plot!(
        "world_mesh::geometry_arena_compaction_evaluated",
        f64::from(u8::from(sample.compaction_evaluated))
    );
    tracy_plot!(
        "world_mesh::geometry_arena_compaction_reclaimed_bytes",
        sample.compaction_reclaimed_bytes as f64
    );
    tracy_plot!(
        "world_mesh::geometry_arena_compaction_copy_bytes",
        sample.compaction_copy_bytes as f64
    );
}

/// Per-frame shared-static source-buffer release activity.
#[derive(Clone, Copy, Debug, Default)]
pub struct WorldMeshStaticSourceReleaseProfileSample {
    /// Meshes whose duplicate dedicated source buffers were actually released this frame.
    pub released_meshes: usize,
    /// Exact dedicated source-buffer bytes released this frame.
    pub released_bytes: u64,
    /// Dedicated source-buffer bytes still duplicating committed shared-arena contents.
    pub remaining_duplicate_source_bytes: u64,
}

/// Records shared-static source release progress and remaining duplicate VRAM.
pub fn plot_world_mesh_static_source_release(sample: WorldMeshStaticSourceReleaseProfileSample) {
    tracy_plot!(
        "world_mesh::shared_static_source_released_meshes",
        sample.released_meshes as f64
    );
    tracy_plot!(
        "world_mesh::shared_static_source_released_bytes",
        sample.released_bytes as f64
    );
    tracy_plot!(
        "world_mesh::shared_static_duplicate_source_bytes",
        sample.remaining_duplicate_source_bytes as f64
    );
}

/// Per-frame retained GPU-cull structural-cache activity.
#[derive(Clone, Copy, Debug, Default)]
pub struct WorldMeshGpuCullCacheProfileSample {
    /// Whether candidate/run planning reused the retained instance-plan structure.
    pub structural_hit: bool,
    /// Whether the already materialized GPU candidate/run byte arrays were reused.
    pub input_hit: bool,
    /// Candidate/run bytes uploaded this frame (zero on the steady-state retained path).
    pub static_upload_bytes: usize,
    /// Bytes avoided by retaining the candidate/run buffers.
    pub avoided_static_upload_bytes: usize,
    /// Small camera/history matrix payload that remains frame-varying.
    pub matrix_upload_bytes: usize,
}

/// Records retained GPU-cull cache effectiveness for one view dispatch.
pub fn plot_world_mesh_gpu_cull_cache(sample: WorldMeshGpuCullCacheProfileSample) {
    tracy_plot!(
        "world_mesh::gpu_cull_structural_cache_hit",
        f64::from(u8::from(sample.structural_hit))
    );
    tracy_plot!(
        "world_mesh::gpu_cull_input_cache_hit",
        f64::from(u8::from(sample.input_hit))
    );
    tracy_plot!(
        "world_mesh::gpu_cull_static_upload_bytes",
        sample.static_upload_bytes as f64
    );
    tracy_plot!(
        "world_mesh::gpu_cull_avoided_static_upload_bytes",
        sample.avoided_static_upload_bytes as f64
    );
    tracy_plot!(
        "world_mesh::gpu_cull_matrix_upload_bytes",
        sample.matrix_upload_bytes as f64
    );
}

/// Per-frame retained CPU draw-plan cache result.
#[derive(Clone, Copy, Debug, Default)]
pub struct WorldMeshDrawPlanCacheProfileSample {
    /// Whether collection and sorting were skipped by either cache mode.
    pub hit: bool,
    /// Whether the hit reused a rigid GPU-static plan across camera/Hi-Z changes.
    pub gpu_static_hit: bool,
    /// Number of view plans reused on the hit.
    pub reused_views: usize,
    /// Number of world/shadow/overlay draw rows reused on the hit.
    pub reused_draws: usize,
}

/// Records whether retained view plans removed CPU collection/sort work this frame.
pub fn plot_world_mesh_draw_plan_cache(sample: WorldMeshDrawPlanCacheProfileSample) {
    tracy_plot!(
        "world_mesh::draw_plan_cache_hit",
        f64::from(u8::from(sample.hit))
    );
    tracy_plot!(
        "world_mesh::draw_plan_cache_gpu_static_hit",
        f64::from(u8::from(sample.gpu_static_hit))
    );
    tracy_plot!(
        "world_mesh::draw_plan_cache_reused_views",
        sample.reused_views as f64
    );
    tracy_plot!(
        "world_mesh::draw_plan_cache_reused_draws",
        sample.reused_draws as f64
    );
}

/// Coverage counters for one world-mesh forward subpass.
///
/// A command is one [`crate::gpu::indirect_buffer::IndexedIndirectCommand`] row, while a run is
/// one `multi_draw_indexed_indirect` API call. Keeping both makes command-buffer coverage distinct
/// from actual CPU recording compression.
#[cfg(feature = "tracy")]
#[derive(Clone, Copy, Debug, Default)]
pub struct WorldMeshForwardIndirectProfileSample {
    /// Draw groups presented to this subpass.
    pub input_groups: usize,
    /// Indirect command rows emitted.
    pub indirect_commands: usize,
    /// `multi_draw_indexed_indirect` calls emitted.
    pub indirect_runs: usize,
    /// Indirect runs containing only one command.
    pub singleton_runs: usize,
    /// Largest command count submitted by one indirect run.
    pub max_commands_per_run: usize,
    /// Groups retained on the ordinary per-mesh recording path.
    pub fallback_groups: usize,
    /// Groups skipped because they do not submit visible forward work.
    pub skipped_groups: usize,
    /// Groups with an invalid node/index range.
    pub invalid_draw_groups: usize,
    /// Groups whose mesh was missing from the resident mesh pool.
    pub mesh_unavailable_groups: usize,
    /// Groups that were batchable but had no indirect path to take: arena-resident, static, single
    /// pipeline, yet still recorded per mesh. This is the size of the prize for widening indirect
    /// coverage, so it counts only groups that passed every eligibility check.
    pub path_unavailable_groups: usize,
    /// Skinned, world-space-deformed, or blendshape-deformed groups.
    pub deformed_groups: usize,
    /// Per-draw UI scissor groups.
    pub scissored_groups: usize,
    /// Groups whose material packet was not ready for submission.
    pub pipeline_unavailable_groups: usize,
    /// Groups whose material expands to more than one raster pipeline pass.
    pub multi_pipeline_groups: usize,
    /// Renderer-generated particle groups, whose mesh buffers are dynamic.
    pub generated_mesh_groups: usize,
    /// Other dynamic mesh groups excluded from the static geometry arena.
    pub dynamic_mesh_groups: usize,
    /// Static groups whose mesh had no geometry-arena allocation.
    pub arena_nonresident_groups: usize,
    /// Arena-resident groups missing one or more material-required vertex streams.
    pub missing_stream_groups: usize,
}

/// Records forward indirect coverage for one subpass.
#[cfg(feature = "tracy")]
pub fn plot_world_mesh_forward_indirect(sample: WorldMeshForwardIndirectProfileSample) {
    tracy_plot!(
        "world_mesh::forward_indirect_input_groups",
        sample.input_groups as f64
    );
    tracy_plot!(
        "world_mesh::forward_indirect_commands",
        sample.indirect_commands as f64
    );
    tracy_plot!(
        "world_mesh::forward_indirect_runs",
        sample.indirect_runs as f64
    );
    tracy_plot!(
        "world_mesh::forward_indirect_singleton_runs",
        sample.singleton_runs as f64
    );
    tracy_plot!(
        "world_mesh::forward_indirect_max_commands_per_run",
        sample.max_commands_per_run as f64
    );
    tracy_plot!(
        "world_mesh::forward_recording_calls",
        sample.indirect_runs.saturating_add(sample.fallback_groups) as f64
    );
    tracy_plot!(
        "world_mesh::forward_fallback_groups",
        sample.fallback_groups as f64
    );
    tracy_plot!(
        "world_mesh::forward_skipped_groups",
        sample.skipped_groups as f64
    );
    tracy_plot!(
        "world_mesh::forward_invalid_draw_groups",
        sample.invalid_draw_groups as f64
    );
    tracy_plot!(
        "world_mesh::forward_mesh_unavailable_groups",
        sample.mesh_unavailable_groups as f64
    );
    tracy_plot!(
        "world_mesh::forward_fallback_path_unavailable",
        sample.path_unavailable_groups as f64
    );
    tracy_plot!(
        "world_mesh::forward_fallback_deformed",
        sample.deformed_groups as f64
    );
    tracy_plot!(
        "world_mesh::forward_fallback_scissored",
        sample.scissored_groups as f64
    );
    tracy_plot!(
        "world_mesh::forward_pipeline_unavailable",
        sample.pipeline_unavailable_groups as f64
    );
    tracy_plot!(
        "world_mesh::forward_fallback_multi_pipeline",
        sample.multi_pipeline_groups as f64
    );
    tracy_plot!(
        "world_mesh::forward_fallback_generated_mesh",
        sample.generated_mesh_groups as f64
    );
    tracy_plot!(
        "world_mesh::forward_fallback_dynamic_mesh",
        sample.dynamic_mesh_groups as f64
    );
    tracy_plot!(
        "world_mesh::forward_fallback_arena_nonresident",
        sample.arena_nonresident_groups as f64
    );
    tracy_plot!(
        "world_mesh::forward_fallback_missing_streams",
        sample.missing_stream_groups as f64
    );
    let commands_per_run = if sample.indirect_runs == 0 {
        0.0
    } else {
        sample.indirect_commands as f64 / sample.indirect_runs as f64
    };
    tracy_plot!(
        "world_mesh::forward_indirect_commands_per_run",
        commands_per_run
    );
    let coverage_percent = if sample.input_groups == 0 {
        0.0
    } else {
        sample.indirect_commands as f64 * 100.0 / sample.input_groups as f64
    };
    tracy_plot!(
        "world_mesh::forward_indirect_coverage_percent",
        coverage_percent
    );
}
