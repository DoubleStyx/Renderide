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
pub(crate) mod phase_classification;
pub mod prefetch;
#[cfg(test)]
pub(crate) mod test_fixtures;

pub use culling::{
    HiZTemporalState, WorldMeshCullInput, WorldMeshCullProjParams,
    build_world_mesh_cull_proj_params, capture_hi_z_temporal,
};
pub use diagnostics::{
    WorldMeshDrawStateRow, WorldMeshDrawStats, state_rows_from_sorted, stats_from_sorted,
    stats_from_sorted_with_plan,
};
#[cfg(test)]
pub(crate) use draw_prep::WorldMeshDrawCollection;
pub use draw_prep::{
    CameraTransformDrawFilter, DrawCollectionFrameCaches, DrawCollectionInputs,
    DrawCollectionMaterialInputs, DrawCollectionSceneAssets, DrawCollectionViewInputs,
    FramePreparedRenderables, RenderWorld, RenderWorldMaintenanceStats, WorldMeshCommandCacheStats,
    WorldMeshDrawArrangeParallelism, WorldMeshDrawCollectParallelism, WorldMeshDrawItem,
    draw_filter_from_camera_entry,
};
pub(crate) use draw_prep::{
    QueuedWorldMeshDraws, WorldMeshCommandCache, fingerprint_world_mesh_draws,
    queue_draws_with_parallelism, queue_prepared_draws_for_views_with_parallelism,
};
pub use instances::{DrawGroup, InstancePlan, MeshPassKind, WorldMeshPhase, build_plan_for_shader};
pub(crate) use instances::{depth_prepass_group_eligible, depth_prepass_item_eligible};
#[cfg(test)]
pub(crate) use materials::compute_batch_key_hash;
pub use materials::{FrameMaterialBatchCache, MaterialDrawBatchKey, TransparentMaterialClass};
pub use prefetch::{PrefetchedWorldMeshViewDraws, WorldMeshDrawPlan, WorldMeshHelperNeeds};
