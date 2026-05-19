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
#[cfg(test)]
pub(crate) mod test_fixtures;

pub use culling::{
    HiZTemporalState, WorldMeshCullInput, WorldMeshCullProjParams,
    build_world_mesh_cull_proj_params, capture_hi_z_temporal,
};
pub use diagnostics::{
    WorldMeshDrawStateRow, WorldMeshDrawStats, state_rows_from_sorted, stats_from_sorted,
};
#[cfg(test)]
pub(crate) use draw_prep::WorldMeshDrawCollection;
pub use draw_prep::{
    CameraTransformDrawFilter, DrawCollectionContext, FramePreparedRenderables, RenderWorld,
    WorldMeshDrawCollectParallelism, WorldMeshDrawItem, WorldMeshDrawKind,
    collect_and_sort_draws_with_parallelism, draw_filter_from_camera_entry,
};
pub use instances::{DrawGroup, InstancePlan, build_plan};
pub(crate) use instances::{
    ShadowCasterRoute, depth_prepass_group_eligible, shadow_caster_batch_route_for_item,
};
#[cfg(test)]
pub(crate) use materials::compute_batch_key_hash;
pub use materials::{FrameMaterialBatchCache, MaterialDrawBatchKey, TransparentMaterialClass};
pub use prefetch::{PrefetchedWorldMeshViewDraws, WorldMeshDrawPlan, WorldMeshHelperNeeds};
