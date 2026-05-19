//! Scene walk that pairs material slots with submesh ranges and applies optional CPU culling.
//!
//! [`collect_and_sort_draws`] walks each render space in 128-renderable parallel chunks
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

use crate::gpu_pools::MeshPool;
use crate::materials::ShaderPermutation;
use crate::materials::host_data::MaterialDictionary;
use crate::materials::{MaterialPipelinePropertyIds, MaterialRouter};
use crate::reflection_probes::specular::ReflectionProbeFrameSelection;
use crate::scene::{RenderSpaceId, SceneCoordinator};
use crate::shared::RenderingContext;
use crate::world_mesh::culling::WorldMeshCullInput;
use crate::world_mesh::materials::FrameMaterialBatchCache;

use super::arrange::arrange_draws_by_phase_bins;
use super::filter::CameraTransformDrawFilter;
use super::item::{WorldMeshDrawCollection, WorldMeshDrawItem};
use super::prepared_renderables::FramePreparedRenderables;

mod candidate;
mod filter_masks;
pub(super) mod prepared;
mod scene_walk;
mod world_matrix;

use filter_masks::build_per_space_filter_masks;
use prepared::collect_prepared_chunk;
use scene_walk::{build_chunk_specs, collect_chunk, estimate_active_renderable_count};

#[cfg(test)]
use super::prepared_renderables::FramePreparedDraw;
#[cfg(test)]
use prepared::prepared_draws_share_renderer;
#[cfg(test)]
use scene_walk::transform_chain_has_degenerate_scale;

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
    /// Populate from `HostCameraFrame::view_origin_world()`.
    pub view_origin_world: Vec3,
    /// Optional CPU frustum + Hi-Z cull inputs.
    pub culling: Option<&'a WorldMeshCullInput<'a>>,
    /// Optional per-camera node filter.
    pub transform_filter: Option<&'a CameraTransformDrawFilter>,
    /// Optional render-space scope for offscreen cameras/tasks.
    pub render_space_filter: Option<RenderSpaceId>,
    /// Optional pre-built material batch cache shared across multiple views in the same frame.
    ///
    /// When `Some`, collection reuses the shared cache instead of rebuilding one per call. Callers
    /// that render multiple views in one frame (secondary render-texture cameras + main
    /// swapchain) should build the cache once via [`FrameMaterialBatchCache::build_for_frame`] and
    /// hand the same borrow to every per-view context. When `None`, a fresh cache is built
    /// internally for this call (backwards-compatible single-view path).
    pub material_cache: Option<&'a FrameMaterialBatchCache>,
    /// Optional frame reflection-probe selector used to choose the set of specular IBL probes to use per draw.
    pub reflection_probes: Option<&'a ReflectionProbeFrameSelection>,
    /// Optional pre-expanded dense draw list shared across multiple views in the same frame.
    ///
    /// When `Some`, collection iterates the flat list instead of walking every active render
    /// space and looking up mesh pool entries per view. The prepared list must have been built
    /// for the **same** [`Self::render_context`] used here; otherwise material-override
    /// resolution may disagree. Single-view callers can leave this `None` and fall back to the
    /// scene-walk path.
    pub prepared: Option<&'a FramePreparedRenderables>,
}

/// How [`collect_and_sort_draws_with_parallelism`] parallelizes per-chunk collection and transparent sorting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorldMeshDrawCollectParallelism {
    /// Per-chunk collection and transparent draw sorting both use rayon.
    Full,
    /// Serial per-chunk merge and transparent sorting; use when an outer `par_iter` already fans out (e.g. multiple secondary RTs).
    SerialInnerForNestedBatch,
}

/// Collects draws from active spaces, then arranges them for batching with control over inner rayon use.
pub fn collect_and_sort_draws_with_parallelism(
    ctx: &DrawCollectionContext<'_>,
    parallelism: WorldMeshDrawCollectParallelism,
) -> WorldMeshDrawCollection {
    profiling::scope!("mesh::collect_and_sort");
    let owned_space_ids;
    let space_ids: &[RenderSpaceId] = {
        profiling::scope!("mesh::collect_and_sort::resolve_space_ids");
        if let Some(prepared) = ctx.prepared {
            if let Some(space_id) = ctx.render_space_filter {
                owned_space_ids = prepared
                    .active_space_ids()
                    .iter()
                    .copied()
                    .filter(|id| *id == space_id)
                    .collect::<Vec<_>>();
                &owned_space_ids
            } else {
                prepared.active_space_ids()
            }
        } else {
            owned_space_ids = match ctx.render_space_filter {
                Some(space_id) => ctx
                    .scene
                    .space(space_id)
                    .filter(|space| space.is_active())
                    .map_or_else(Vec::new, |_| vec![space_id]),
                None => ctx.scene.render_space_ids().collect::<Vec<_>>(),
            };
            &owned_space_ids
        }
    };
    let cap_hint = {
        profiling::scope!("mesh::collect_and_sort::estimate_capacity");
        estimate_active_renderable_count(space_ids, ctx)
    };

    let owned_cache;
    let cache: &FrameMaterialBatchCache = {
        profiling::scope!("mesh::collect_and_sort::resolve_material_cache");
        if let Some(shared) = ctx.material_cache {
            shared
        } else {
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
    let filter_masks = {
        profiling::scope!("mesh::collect_and_sort::build_filter_masks");
        build_per_space_filter_masks(space_ids, ctx)
    };

    let per_chunk = {
        profiling::scope!("mesh::collect_and_sort::collect_chunks");
        collect_world_mesh_chunks(ctx, parallelism, cache, &filter_masks, space_ids)
    };

    let mut out = Vec::with_capacity(cap_hint);
    let mut cull_stats = (0usize, 0usize, 0usize);
    {
        profiling::scope!("mesh::collect_and_sort::merge_chunks");
        for (items, cs) in per_chunk {
            cull_stats.0 += cs.0;
            cull_stats.1 += cs.1;
            cull_stats.2 += cs.2;
            out.extend(items);
        }
    }

    {
        profiling::scope!("mesh::collect_and_sort::assign_collect_order");
        for (i, item) in out.iter_mut().enumerate() {
            item.collect_order = i;
        }
    }

    let arrangement = {
        profiling::scope!("mesh::arrange");
        arrange_draws_by_phase_bins(
            &mut out,
            parallelism == WorldMeshDrawCollectParallelism::Full,
        )
    };
    WorldMeshDrawCollection {
        items: out,
        draws_pre_cull: cull_stats.0,
        draws_culled: cull_stats.1,
        draws_hi_z_culled: cull_stats.2,
        arrangement,
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
            "prepared renderables were built for a different render context than the per-view draw collection -- material overrides would disagree"
        );
        profiling::scope!("mesh::collect_prepared");
        // Cached run-aligned chunking ensures every renderer's slots stay inside one chunk so the
        // per-renderer CPU cull and material-batch lookup happens at most once per renderer per
        // view without allocating a chunk list per view.
        let run_chunks = prepared.run_chunks();
        let draws = prepared.draws();
        if parallelism == WorldMeshDrawCollectParallelism::Full && run_chunks.len() >= 2 {
            profiling::scope!("mesh::collect_prepared::parallel_chunks");
            run_chunks
                .par_iter()
                .map(|&chunk| {
                    profiling::scope!("mesh::collect_prepared::chunk_worker");
                    let runs = prepared.runs_for_chunk(chunk);
                    collect_prepared_chunk(draws, runs, ctx, cache, filter_masks)
                })
                .collect()
        } else {
            profiling::scope!("mesh::collect_prepared::serial_chunks");
            run_chunks
                .iter()
                .map(|&chunk| {
                    let runs = prepared.runs_for_chunk(chunk);
                    collect_prepared_chunk(draws, runs, ctx, cache, filter_masks)
                })
                .collect()
        }
    } else {
        let chunks = {
            profiling::scope!("mesh::collect::build_chunk_specs");
            build_chunk_specs(space_ids, ctx)
        };
        profiling::scope!("mesh::collect");
        if parallelism == WorldMeshDrawCollectParallelism::Full && chunks.len() >= 2 {
            profiling::scope!("mesh::collect::parallel_chunks");
            chunks
                .par_iter()
                .map(|spec| {
                    profiling::scope!("mesh::collect::chunk_worker");
                    collect_chunk(spec, ctx, cache, filter_masks)
                })
                .collect()
        } else {
            profiling::scope!("mesh::collect::serial_chunks");
            chunks
                .iter()
                .map(|spec| collect_chunk(spec, ctx, cache, filter_masks))
                .collect()
        }
    }
}

#[cfg(test)]
mod tests;
