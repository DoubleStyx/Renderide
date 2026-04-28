//! Scene walk that pairs material slots with submesh ranges and applies optional CPU culling.
//!
//! [`collect_and_sort_world_mesh_draws`] walks each render space in 128-renderable parallel chunks
//! ([`rayon`]), merges in [`SceneCoordinator::render_space_ids`] order, assigns
//! [`WorldMeshDrawItem::collect_order`], then sorts.
//!
//! Material-derived batch key fields are computed once per `(material_asset_id, property_block_id)`
//! per call via [`FrameMaterialBatchCache`] before the parallel phase begins. This eliminates
//! repeated dictionary and router lookups for the common case where hundreds of draws share a
//! few dozen materials.

use hashbrown::HashMap;

use glam::{Mat4, Vec3};
use rayon::prelude::*;

use crate::assets::material::{MaterialDictionary, MaterialPropertyLookupIds};
use crate::materials::{MaterialPipelinePropertyIds, MaterialRouter, RasterFrontFace};
use crate::pipelines::ShaderPermutation;
use crate::resources::MeshPool;
use crate::scene::{RenderSpaceId, SceneCoordinator};
use crate::shared::RenderingContext;

use super::material_batch_cache::{FrameMaterialBatchCache, MaterialResolveCtx};
use super::prepared::FramePreparedRenderables;
use super::sort::{batch_key_for_slot_cached, sort_world_mesh_draws, sort_world_mesh_draws_serial};
use super::types::{WorldMeshDrawCollection, WorldMeshDrawItem};

#[path = "collect_candidate.rs"]
mod candidate;
#[path = "collect_filter.rs"]
mod filter;
#[path = "collect_prepared.rs"]
mod prepared_collect;
#[path = "collect_scene_walk.rs"]
mod scene_walk;

use filter::build_per_space_filter_masks;
use prepared_collect::{collect_prepared_chunk, PREPARED_CHUNK_SIZE};
use scene_walk::{build_chunk_specs, collect_chunk, estimate_active_renderable_count};

#[cfg(test)]
use super::prepared::FramePreparedDraw;
#[cfg(test)]
use prepared_collect::prepared_draws_share_renderer;
#[cfg(test)]
use scene_walk::transform_chain_has_degenerate_scale;

/// Resolves the draw's world matrix when the selected vertex stream is still local-space.
#[inline]
fn world_matrix_for_local_vertex_stream(
    ctx: &DrawCollectionContext<'_>,
    space_id: RenderSpaceId,
    node_id: i32,
) -> Option<Mat4> {
    if node_id < 0 {
        return None;
    }
    ctx.scene.world_matrix_for_render_context(
        space_id,
        node_id as usize,
        ctx.render_context,
        ctx.head_output_transform,
    )
}

/// Resolves the raster front face for the model matrix used by the forward vertex shader.
#[inline]
fn front_face_for_world_matrix(world_matrix: Option<Mat4>) -> RasterFrontFace {
    world_matrix
        .map(RasterFrontFace::from_model_matrix)
        .unwrap_or_default()
}

/// Read-only scene, material, and cull state shared across all spaces during draw collection.
pub struct DrawCollectionContext<'a> {
    /// Scene graph for mesh renderables.
    pub scene: &'a SceneCoordinator,
    /// Resident meshes (submeshes, deform buffers).
    pub mesh_pool: &'a MeshPool,
    /// Material property dictionary for batch keys.
    pub material_dict: &'a MaterialDictionary<'a>,
    /// Shader stem / pipeline routing.
    pub material_router: &'a MaterialRouter,
    /// Interned material property ids that affect pipeline state.
    pub pipeline_property_ids: &'a MaterialPipelinePropertyIds,
    /// Default vs multiview permutation for embedded materials.
    pub shader_perm: ShaderPermutation,
    /// Mono vs stereo / overlay render context.
    pub render_context: RenderingContext,
    /// Head / rig transform for world matrix resolution.
    pub head_output_transform: Mat4,
    /// Camera world position for back-to-front distance sorting of transparent draws.
    ///
    /// Populate from `HostCameraFrame::explicit_camera_world_position.unwrap_or_else(|| head_output_transform.col(3).truncate())`.
    pub view_origin_world: Vec3,
    /// Optional CPU frustum + Hi-Z cull inputs.
    pub culling: Option<&'a super::super::world_mesh_cull::WorldMeshCullInput<'a>>,
    /// Optional per-camera node filter.
    pub transform_filter: Option<&'a super::types::CameraTransformDrawFilter>,
    /// Optional pre-built material batch cache shared across multiple views in the same frame.
    ///
    /// When `Some`, collection reuses the shared cache instead of rebuilding one per call. Callers
    /// that render multiple views in one frame (secondary render-texture cameras + main
    /// swapchain) should build the cache once via [`FrameMaterialBatchCache::build_for_frame`] and
    /// hand the same borrow to every per-view context. When `None`, a fresh cache is built
    /// internally for this call (backwards-compatible single-view path).
    pub material_cache: Option<&'a FrameMaterialBatchCache>,
    /// Optional pre-expanded dense draw list shared across multiple views in the same frame.
    ///
    /// When `Some`, collection iterates the flat list instead of walking every active render
    /// space and looking up mesh pool entries per view. The prepared list must have been built
    /// for the **same** [`Self::render_context`] used here; otherwise material-override
    /// resolution may disagree. Single-view callers can leave this `None` and fall back to the
    /// scene-walk path.
    pub prepared: Option<&'a FramePreparedRenderables>,
}

/// How [`collect_and_sort_world_mesh_draws_with_parallelism`] parallelizes per-chunk collection and sorting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorldMeshDrawCollectParallelism {
    /// Per-chunk collection and draw sort both use rayon.
    Full,
    /// Serial per-chunk merge and serial sort; use when an outer `par_iter` already fans out (e.g. multiple secondary RTs).
    SerialInnerForNestedBatch,
}

/// Collects draws from active spaces, then sorts for batching (material / pipeline boundaries).
///
/// When `culling` is [`Some`], instances outside the frustum (and optional Hi-Z) are dropped (see
/// [`mesh_draw_passes_cpu_cull`](super::super::world_mesh_cull_eval::mesh_draw_passes_cpu_cull)).
///
/// Collection runs over 128-renderer chunks in parallel via [`rayon`] by default; results are
/// merged in the same order as [`SceneCoordinator::render_space_ids`], then
/// [`WorldMeshDrawItem::collect_order`] is assigned for transparent sort stability.
pub fn collect_and_sort_world_mesh_draws(
    ctx: &DrawCollectionContext<'_>,
) -> WorldMeshDrawCollection {
    collect_and_sort_world_mesh_draws_with_parallelism(ctx, WorldMeshDrawCollectParallelism::Full)
}

/// Like [`collect_and_sort_world_mesh_draws`], with control over inner rayon use (see [`WorldMeshDrawCollectParallelism`]).
pub fn collect_and_sort_world_mesh_draws_with_parallelism(
    ctx: &DrawCollectionContext<'_>,
    parallelism: WorldMeshDrawCollectParallelism,
) -> WorldMeshDrawCollection {
    profiling::scope!("mesh::collect_and_sort");
    let space_ids: Vec<RenderSpaceId> = ctx.scene.render_space_ids().collect();
    let cap_hint = estimate_active_renderable_count(&space_ids, ctx);

    let owned_cache;
    let cache: &FrameMaterialBatchCache = match ctx.material_cache {
        Some(shared) => shared,
        None => {
            let mut local = FrameMaterialBatchCache::new();
            local.refresh_for_frame(
                ctx.scene,
                ctx.material_dict,
                ctx.material_router,
                ctx.pipeline_property_ids,
                ctx.shader_perm,
            );
            owned_cache = local;
            &owned_cache
        }
    };
    let filter_masks = build_per_space_filter_masks(&space_ids, ctx);

    let per_chunk = collect_world_mesh_chunks(ctx, parallelism, cache, &filter_masks, &space_ids);

    let mut out = Vec::with_capacity(cap_hint);
    let mut cull_stats = (0usize, 0usize, 0usize);
    for (items, cs) in per_chunk {
        cull_stats.0 += cs.0;
        cull_stats.1 += cs.1;
        cull_stats.2 += cs.2;
        out.extend(items);
    }

    for (i, item) in out.iter_mut().enumerate() {
        item.collect_order = i;
    }

    {
        profiling::scope!("mesh::sort");
        match parallelism {
            WorldMeshDrawCollectParallelism::Full => sort_world_mesh_draws(&mut out),
            WorldMeshDrawCollectParallelism::SerialInnerForNestedBatch => {
                sort_world_mesh_draws_serial(&mut out);
            }
        }
    }
    WorldMeshDrawCollection {
        items: out,
        draws_pre_cull: cull_stats.0,
        draws_culled: cull_stats.1,
        draws_hi_z_culled: cull_stats.2,
    }
}

/// Dispatches chunk collection to the prepared-draw path or the scene-walk fallback.
///
/// `Full` parallelism maps chunks via rayon; `SerialInnerForNestedBatch` keeps iteration serial
/// so nested multi-view batches don't hammer rayon with contention.
fn collect_world_mesh_chunks(
    ctx: &DrawCollectionContext<'_>,
    parallelism: WorldMeshDrawCollectParallelism,
    cache: &FrameMaterialBatchCache,
    filter_masks: &HashMap<RenderSpaceId, Vec<bool>>,
    space_ids: &[RenderSpaceId],
) -> Vec<(Vec<WorldMeshDrawItem>, (usize, usize, usize))> {
    if let Some(prepared) = ctx.prepared {
        debug_assert_eq!(
            prepared.render_context(),
            ctx.render_context,
            "prepared renderables were built for a different render context than the per-view draw collection — material overrides would disagree"
        );
        profiling::scope!("mesh::collect_prepared");
        match parallelism {
            WorldMeshDrawCollectParallelism::Full => prepared
                .draws
                .par_chunks(PREPARED_CHUNK_SIZE)
                .map(|chunk| collect_prepared_chunk(chunk, ctx, cache, filter_masks))
                .collect(),
            WorldMeshDrawCollectParallelism::SerialInnerForNestedBatch => prepared
                .draws
                .chunks(PREPARED_CHUNK_SIZE)
                .map(|chunk| collect_prepared_chunk(chunk, ctx, cache, filter_masks))
                .collect(),
        }
    } else {
        let chunks = build_chunk_specs(space_ids, ctx);
        profiling::scope!("mesh::collect");
        match parallelism {
            WorldMeshDrawCollectParallelism::Full => chunks
                .par_iter()
                .map(|spec| collect_chunk(spec, ctx, cache, filter_masks))
                .collect(),
            WorldMeshDrawCollectParallelism::SerialInnerForNestedBatch => chunks
                .iter()
                .map(|spec| collect_chunk(spec, ctx, cache, filter_masks))
                .collect(),
        }
    }
}

#[cfg(test)]
#[path = "collect_tests.rs"]
mod tests;
