//! Compile-time validated **render graph**: pass ordering, resource flow checks, and a single
//! command-encode path per frame (v1).
//!
//! ## Responsibilities
//!
//! - **[`GraphBuilder`]** — register [`RenderPass`] nodes, optional [`GraphBuilder::add_pass_if`],
//!   edges, then [`GraphBuilder::build`] for a topological order and producer/consumer validation.
//! - **[`CompiledRenderGraph`]** — immutable schedule; [`CompiledRenderGraph::execute`] acquires
//!   the swapchain at most once when any pass writes [`ResourceSlot::Backbuffer`], records all
//!   passes into one encoder, submits, and presents.
//!
//! ## Phase 2 (not implemented here)
//!
//! - Nested subgraphs / phase labels.
//! - Real GPU resource handles and automatic barriers per slot.
//! - Multiple encoders, parallel recording, and async compute queue routing.
//! - Graph reuse across frames with invalidation keys (resolution, MSAA, toggles).

mod builder;
mod camera;
mod compiled;
mod context;
mod error;
mod frame_params;
mod frustum;
mod hi_z_cpu;
mod hi_z_occlusion;
mod ids;
mod output_depth_mode;
mod pass;
mod resources;
mod reverse_z_depth;
mod skinning_palette;
mod world_mesh_cull;
mod world_mesh_draw_prep;

pub mod passes;

pub use world_mesh_draw_prep::{
    collect_and_sort_world_mesh_draws, resolved_material_slots, sort_world_mesh_draws,
    world_mesh_draw_stats_from_sorted, MaterialDrawBatchKey, WorldMeshDrawCollection,
    WorldMeshDrawItem, WorldMeshDrawStats,
};

pub use builder::GraphBuilder;
pub use camera::{
    apply_view_handedness_fix, clamp_desktop_fov_degrees, effective_head_output_clip_planes,
    reverse_z_orthographic, reverse_z_perspective, reverse_z_perspective_openxr_fov,
    view_matrix_from_render_transform,
};
pub use camera::{DESKTOP_FOV_DEGREES_MAX, DESKTOP_FOV_DEGREES_MIN};
pub use compiled::{CompileStats, CompiledRenderGraph, ExternalFrameTargets};
pub use context::RenderPassContext;
pub use error::{GraphBuildError, GraphExecuteError, RenderPassError};
pub use frame_params::{FrameRenderParams, HostCameraFrame};
pub use frustum::{
    mesh_bounds_degenerate_for_cull, mesh_bounds_max_half_extent, world_aabb_from_local_bounds,
    world_aabb_from_skinned_bone_origins, world_aabb_visible_in_homogeneous_clip, Frustum, Plane,
    HOMOGENEOUS_CLIP_EPS,
};
pub use hi_z_cpu::{
    hi_z_pyramid_dimensions, hi_z_snapshot_from_linear_linear, mip_dimensions,
    mip_levels_for_extent, unpack_linear_rows_to_mips, HiZCpuSnapshot, HiZCullData,
    HiZStereoCpuSnapshot,
};
pub use hi_z_occlusion::{
    hi_z_view_proj_matrices, mesh_fully_occluded_in_hiz, stereo_hiz_keeps_draw,
};
pub use ids::PassId;
pub use output_depth_mode::OutputDepthMode;
pub use pass::RenderPass;
pub use resources::{PassResources, ResourceSlot};
pub use reverse_z_depth::{MAIN_FORWARD_DEPTH_CLEAR, MAIN_FORWARD_DEPTH_COMPARE};
pub use skinning_palette::build_skinning_palette;
pub use world_mesh_cull::{
    build_world_mesh_cull_proj_params, capture_hi_z_temporal, HiZTemporalState, WorldMeshCullInput,
    WorldMeshCullProjParams,
};

/// Builds the default graph: mesh deform compute, clustered lights, world forward, then Hi-Z readback.
pub fn build_default_main_graph() -> Result<CompiledRenderGraph, GraphBuildError> {
    let mut builder = GraphBuilder::new();
    let deform = builder.add_pass(Box::new(passes::MeshDeformPass::new()));
    let clustered = builder.add_pass(Box::new(passes::ClusteredLightPass::new()));
    let forward = builder.add_pass(Box::new(passes::WorldMeshForwardPass::new()));
    let hiz = builder.add_pass(Box::new(passes::HiZBuildPass::new()));
    builder.add_edge(deform, clustered);
    builder.add_edge(clustered, forward);
    builder.add_edge(forward, hiz);
    builder.build()
}

#[cfg(test)]
mod default_graph_tests {
    use super::*;

    #[test]
    fn default_main_needs_surface_and_four_passes() {
        let g = build_default_main_graph().expect("default graph");
        assert!(g.needs_surface_acquire());
        assert_eq!(g.pass_count(), 4);
        assert_eq!(g.compile_stats.topo_levels, 4);
    }
}
