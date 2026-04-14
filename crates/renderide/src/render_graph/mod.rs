//! Compile-time validated **render graph**: pass ordering, resource flow checks, and a single
//! command-encode path per frame (v1).
//!
//! **Hi-Z-related code:** CPU helpers for mip layout, depth readback unpacking, and screen-space
//! occlusion tests live in [`hi_z_cpu`] and [`hi_z_occlusion`]. GPU pyramid build, staging, and
//! pipelines are under [`crate::render_graph::occlusion`].
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
//!
//! ## Frame pipeline (v1 ordering)
//!
//! Runtime and passes combine to the following **logical** phases each frame (some CPU-side,
//! some GPU passes in [`passes`]):
//!
//! 1. **LightPrep** — [`crate::backend::FrameResourceManager::prepare_lights_from_scene`] packs
//!    clustered lights (see [`cluster_frame`]); at most one full pack per winit tick (coalesced across graph entry points).
//! 2. **Camera / cluster params** — [`frame_params::FrameRenderParams`] + [`cluster_frame`] from
//!    host camera and [`HostCameraFrame`].
//! 3. **Cull** — frustum and Hi-Z occlusion in [`world_mesh_cull`] (inputs to forward pass).
//! 4. **Sort** — [`world_mesh_draw_prep`] builds draw order and batch keys.
//! 5. **DrawPrep** — per-draw uniforms and material resolution inside [`passes::WorldMeshForwardPass`].
//! 6. **RenderPasses** — [`CompiledRenderGraph`] runs deform → clustered lights → clear → forward
//!    (see [`default_graph_tests`] / builder).
//! 7. **HiZ** — [`passes::HiZBuildPass`] after depth is written; CPU readback feeds next frame’s cull
//!    ([`crate::render_graph::occlusion`]).
//! 8. **FrameEnd** — submit, optional debug HUD composite, present, Hi-Z frame bookkeeping.

mod builder;
mod camera;
mod cluster_frame;
mod compiled;
mod context;
mod error;
mod frame_params;
mod frustum;
mod hi_z_cpu;
mod hi_z_occlusion;
mod ids;
pub mod occlusion;
mod output_depth_mode;
mod pass;
mod resources;
mod reverse_z_depth;
mod secondary_camera;
mod skinning_palette;
mod world_mesh_cull;
mod world_mesh_cull_eval;
mod world_mesh_draw_prep;
mod world_mesh_draw_stats;

pub mod passes;

pub use world_mesh_draw_prep::{
    build_instance_batches, collect_and_sort_world_mesh_draws,
    collect_and_sort_world_mesh_draws_with_parallelism, draw_filter_from_camera_entry,
    resolved_material_slots, sort_world_mesh_draws, CameraTransformDrawFilter, InstanceBatch,
    MaterialDrawBatchKey, WorldMeshDrawCollectParallelism, WorldMeshDrawCollection,
    WorldMeshDrawItem,
};
pub use world_mesh_draw_stats::{world_mesh_draw_stats_from_sorted, WorldMeshDrawStats};

pub use builder::GraphBuilder;
pub use camera::{
    apply_view_handedness_fix, clamp_desktop_fov_degrees, effective_head_output_clip_planes,
    reverse_z_orthographic, reverse_z_perspective, reverse_z_perspective_openxr_fov,
    view_matrix_for_world_mesh_render_space, view_matrix_from_render_transform,
};
pub use camera::{DESKTOP_FOV_DEGREES_MAX, DESKTOP_FOV_DEGREES_MIN};
pub use cluster_frame::{cluster_frame_params, cluster_frame_params_stereo, ClusterFrameParams};
pub use compiled::{
    CompileStats, CompiledRenderGraph, ExternalFrameTargets, ExternalOffscreenTargets, FrameView,
    FrameViewTarget,
};
pub use context::RenderPassContext;
pub use error::{GraphBuildError, GraphExecuteError, RenderPassError};
pub use frame_params::{FrameRenderParams, HostCameraFrame, OcclusionViewId};
pub use frustum::{
    mesh_bounds_degenerate_for_cull, mesh_bounds_max_half_extent, world_aabb_from_local_bounds,
    world_aabb_from_skinned_bone_origins, world_aabb_visible_in_homogeneous_clip, Frustum, Plane,
    HOMOGENEOUS_CLIP_EPS,
};
pub use hi_z_cpu::{
    hi_z_pyramid_dimensions, hi_z_snapshot_from_linear_linear, mip_dimensions,
    mip_levels_for_extent, unpack_linear_rows_to_mips, HiZCpuSnapshot, HiZCullData,
    HiZStereoCpuSnapshot, HI_Z_PYRAMID_MAX_LONG_EDGE,
};
pub use hi_z_occlusion::{
    hi_z_view_proj_matrices, mesh_fully_occluded_in_hiz, stereo_hiz_keeps_draw,
};
pub use ids::PassId;
pub use output_depth_mode::OutputDepthMode;
pub use pass::{PassPhase, RenderPass};
pub use resources::{PassResources, ResourceSlot};
pub use reverse_z_depth::{MAIN_FORWARD_DEPTH_CLEAR, MAIN_FORWARD_DEPTH_COMPARE};
pub use secondary_camera::{camera_state_enabled, host_camera_frame_for_render_texture};
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
