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
mod ids;
mod pass;
mod resources;
mod reverse_z_depth;
mod world_mesh_draw_prep;

pub mod passes;

pub use world_mesh_draw_prep::{
    collect_and_sort_world_mesh_draws, resolved_material_slots, sort_world_mesh_draws,
    world_mesh_draw_stats_from_sorted, MaterialDrawBatchKey, WorldMeshDrawItem, WorldMeshDrawStats,
};

pub use builder::GraphBuilder;
pub use camera::{
    apply_view_handedness_fix, clamp_desktop_fov_degrees, reverse_z_orthographic,
    reverse_z_perspective, reverse_z_perspective_openxr_fov, view_matrix_from_render_transform,
};
pub use camera::{DESKTOP_FOV_DEGREES_MAX, DESKTOP_FOV_DEGREES_MIN};
pub use compiled::{CompileStats, CompiledRenderGraph, ExternalFrameTargets};
pub use context::RenderPassContext;
pub use error::{GraphBuildError, GraphExecuteError, RenderPassError};
pub use frame_params::{FrameRenderParams, HostCameraFrame};
pub use ids::PassId;
pub use pass::RenderPass;
pub use resources::{PassResources, ResourceSlot};
pub use reverse_z_depth::{MAIN_FORWARD_DEPTH_CLEAR, MAIN_FORWARD_DEPTH_COMPARE};

/// Builds the default graph: mesh deform compute, then world forward (clear + depth + mesh draw).
pub fn build_default_main_graph() -> Result<CompiledRenderGraph, GraphBuildError> {
    let mut builder = GraphBuilder::new();
    let deform = builder.add_pass(Box::new(passes::MeshDeformPass::new()));
    let forward = builder.add_pass(Box::new(passes::WorldMeshForwardPass::new()));
    builder.add_edge(deform, forward);
    builder.build()
}

#[cfg(test)]
mod default_graph_tests {
    use super::*;

    #[test]
    fn default_main_needs_surface_and_two_passes() {
        let g = build_default_main_graph().expect("default graph");
        assert!(g.needs_surface_acquire());
        assert_eq!(g.pass_count(), 2);
        assert_eq!(g.compile_stats.topo_levels, 2);
    }
}
