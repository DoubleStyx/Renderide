//! World-mesh visibility planning: frustum + Hi-Z culling, draw collection, sorting, instance grouping.
//!
//! Pure-CPU subsystem that consumes scene state and Hi-Z snapshots and produces a sorted draw list
//! for the render-graph world-mesh forward pass. Owns no GPU resources.

pub(crate) mod cluster_frame;
pub(crate) mod cull;
pub(crate) mod cull_eval;
pub(crate) mod draw_prep;
pub(crate) mod draw_stats;
pub mod forward_state;
pub(crate) mod frustum;

pub use cluster_frame::{ClusterFrameParams, cluster_frame_params, cluster_frame_params_stereo};
pub use cull::{
    HiZTemporalState, WorldMeshCullInput, WorldMeshCullProjParams,
    build_world_mesh_cull_proj_params, capture_hi_z_temporal,
};
pub use draw_prep::{
    CameraTransformDrawFilter, DrawCollectionContext, DrawGroup, FrameMaterialBatchCache,
    FramePreparedRenderables, InstancePlan, MaterialDrawBatchKey, WorldMeshDrawCollectParallelism,
    WorldMeshDrawCollection, WorldMeshDrawItem, build_instance_plan,
    collect_and_sort_world_mesh_draws, collect_and_sort_world_mesh_draws_with_parallelism,
    draw_filter_from_camera_entry, resolved_material_slots, sort_world_mesh_draws,
};
pub use draw_stats::{
    WorldMeshDrawStateRow, WorldMeshDrawStats, world_mesh_draw_state_rows_from_sorted,
    world_mesh_draw_stats_from_sorted,
};
pub use forward_state::{
    PrefetchedWorldMeshDrawsSlot, PrefetchedWorldMeshViewDraws, PreparedWorldMeshForwardFrame,
    WorldMeshForwardPipelineState, WorldMeshForwardPlanSlot, WorldMeshHelperNeeds,
};
pub use frustum::{
    Frustum, HOMOGENEOUS_CLIP_EPS, Plane, mesh_bounds_degenerate_for_cull,
    world_aabb_from_local_bounds, world_aabb_visible_in_homogeneous_clip,
};
