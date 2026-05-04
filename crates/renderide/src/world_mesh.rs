//! World-mesh visibility planning: frustum + Hi-Z culling, draw collection, sorting, instance grouping.
//!
//! Pure-CPU subsystem that consumes scene state and Hi-Z snapshots and produces a sorted draw list
//! for the render-graph world-mesh forward pass. Owns no GPU resources.

pub(crate) mod cluster;
pub(crate) mod culling;
pub(crate) mod diagnostics;
pub(crate) mod draw_prep;
pub(crate) mod instances;
pub(crate) mod materials;
pub mod prefetch;

pub use cluster::{ClusterFrameParams, cluster_frame_params, cluster_frame_params_stereo};
pub use culling::{
    Frustum, HOMOGENEOUS_CLIP_EPS, HiZTemporalState, Plane, WorldMeshCullInput,
    WorldMeshCullProjParams, build_world_mesh_cull_proj_params, capture_hi_z_temporal,
    mesh_bounds_degenerate_for_cull, world_aabb_from_local_bounds,
    world_aabb_visible_in_homogeneous_clip,
};
pub use diagnostics::{
    WorldMeshDrawStateRow, WorldMeshDrawStats, state_rows_from_sorted, stats_from_sorted,
};
pub use draw_prep::{
    CameraTransformDrawFilter, DrawCollectionContext, FramePreparedRenderables, RenderWorld,
    WorldMeshDrawCollectParallelism, WorldMeshDrawCollection, WorldMeshDrawItem,
    collect_and_sort_draws, collect_and_sort_draws_with_parallelism, draw_filter_from_camera_entry,
    resolved_material_slots, sort_draws,
};
pub use instances::{DrawGroup, InstancePlan, build_plan};
pub use materials::{FrameMaterialBatchCache, MaterialDrawBatchKey, compute_batch_key_hash};
pub use prefetch::{PrefetchedWorldMeshViewDraws, WorldMeshHelperNeeds};
