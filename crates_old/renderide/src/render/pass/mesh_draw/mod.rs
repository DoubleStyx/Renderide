//! Mesh draw collection and recording for mesh and overlay passes.
//!
//! Collects draws from batches, partitions by overlay/skinned, and records into render passes.
//! Uses glam for SIMD-optimized matrix operations.

mod collect;
mod material_bind_record;
mod pbr_bind;
mod pipeline;
mod record_non_skinned;
mod record_skinned;
mod types;

#[cfg(test)]
mod tests;

pub(super) use collect::collect_mesh_draws;
#[allow(unused_imports)]
pub(super) use pipeline::{
    mesh_pipeline_variant_for_mrt, overlay_pipeline_variant_for_orthographic,
};
pub(super) use record_non_skinned::record_non_skinned_draws;
pub(super) use record_skinned::record_skinned_draws;
pub(crate) use types::{BatchedDraw, SkinnedBatchedDraw};
pub(super) use types::{
    CollectMeshDrawsContext, MeshDrawParams, PbrSceneParams, RtShadowBindParams,
};
