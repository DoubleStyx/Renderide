//! Scene walk that pairs material slots with submeshes and applies optional CPU culling.
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
use crate::materials::{MaterialPipelinePropertyIds, MaterialRouter};
use crate::pipelines::ShaderPermutation;
use crate::resources::MeshPool;
use crate::scene::{
    MeshMaterialSlot, RenderSpaceId, SceneCoordinator, SkinnedMeshRenderer, StaticMeshRenderer,
};
use crate::shared::RenderingContext;

use super::material_batch_cache::FrameMaterialBatchCache;
use super::prepared::{FramePreparedDraw, FramePreparedRenderables};
use super::sort::{batch_key_for_slot_cached, sort_world_mesh_draws, sort_world_mesh_draws_serial};
use super::types::{WorldMeshDrawCollection, WorldMeshDrawItem};

use super::super::world_mesh_cull_eval::{
    mesh_draw_passes_cpu_cull, CpuCullFailure, MeshCullTarget,
};

/// Renders per chunk (static or skinned slice of one render space).
const WORLD_MESH_COLLECT_CHUNK_SIZE: usize = 128;

/// Rayon chunk width when iterating a pre-expanded [`FramePreparedRenderables`] list.
///
/// Matches [`WORLD_MESH_COLLECT_CHUNK_SIZE`] so per-view CPU cost stays bounded by the same
/// per-task overhead as the scene-walk path.
const PREPARED_CHUNK_SIZE: usize = 128;

/// Submesh index range for one material slot pairing during draw collection.
pub(crate) struct SubmeshSlotIndices {
    /// Slot index in [`StaticMeshRenderer`] material slots.
    pub slot_index: usize,
    /// First index in the mesh index buffer for this submesh.
    pub first_index: u32,
    /// Index count for this submesh draw.
    pub index_count: u32,
}

/// Layer and skin deform flags that affect CPU cull and [`WorldMeshDrawItem`] fields.
pub(crate) struct OverlayDeformCullFlags {
    /// Overlay layer uses alternate cull behavior.
    pub is_overlay: bool,
    /// Skinned mesh with world-space deform from the skin cache.
    pub world_space_deformed: bool,
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
    /// Populate from `HostCameraFrame::secondary_camera_world_position.unwrap_or_else(|| head_output_transform.col(3).truncate())`.
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

/// One static or skinned mesh renderer with its resolved [`crate::assets::mesh::GpuMesh`] and submesh index ranges.
struct StaticMeshDrawSource<'a> {
    space_id: RenderSpaceId,
    renderer: &'a StaticMeshRenderer,
    renderable_index: usize,
    skinned: bool,
    skinned_renderer: Option<&'a SkinnedMeshRenderer>,
    mesh: &'a crate::assets::mesh::GpuMesh,
    submeshes: &'a [(u32, u32)],
}

/// Mutable expansion state while expanding one chunk into draw items.
struct DrawCollectionAccumulator<'a> {
    out: &'a mut Vec<WorldMeshDrawItem>,
    cull_stats: &'a mut (usize, usize, usize),
    /// Precomputed filter result per node index. When `Some`, used in place of
    /// [`super::types::CameraTransformDrawFilter::passes_scene_node`] to avoid per-draw ancestor walks.
    filter_pass_mask: Option<&'a [bool]>,
}

/// How [`collect_and_sort_world_mesh_draws_with_parallelism`] parallelizes per-chunk collection and sorting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorldMeshDrawCollectParallelism {
    /// Per-chunk collection and draw sort both use rayon.
    Full,
    /// Serial per-chunk merge and serial sort; use when an outer `par_iter` already fans out (e.g. multiple secondary RTs).
    SerialInnerForNestedBatch,
}

/// Whether a chunk covers the static or skinned renderer list of a render space.
#[derive(Clone, Copy)]
enum ChunkKind {
    Static,
    Skinned,
}

/// One 128-renderable slice of a render space's static or skinned renderer array.
struct WorldMeshChunkSpec {
    space_id: RenderSpaceId,
    kind: ChunkKind,
    range: std::ops::Range<usize>,
}

/// Expands one static mesh renderer into draw items (material slots × submeshes).
///
/// `collect_order` is filled with a placeholder; [`collect_and_sort_world_mesh_draws`] assigns the
/// final stable index after per-chunk results are merged.
fn push_draws_for_renderer(
    ctx: &DrawCollectionContext<'_>,
    acc: &mut DrawCollectionAccumulator<'_>,
    draw: StaticMeshDrawSource<'_>,
    cache: &FrameMaterialBatchCache,
) {
    if let Some(f) = ctx.transform_filter {
        let passes = match acc.filter_pass_mask {
            Some(mask) => {
                let nid = draw.renderer.node_id;
                nid >= 0 && (nid as usize) < mask.len() && mask[nid as usize]
            }
            None => f.passes_scene_node(ctx.scene, draw.space_id, draw.renderer.node_id),
        };
        if !passes {
            return;
        }
    }

    // Resolve material slots inline to avoid the Cow::Owned(vec![..]) allocation for the
    // primary-material fallback path.
    let fallback_slot;
    let slots: &[MeshMaterialSlot] = if !draw.renderer.material_slots.is_empty() {
        &draw.renderer.material_slots
    } else if let Some(mat_id) = draw.renderer.primary_material_asset_id {
        fallback_slot = MeshMaterialSlot {
            material_asset_id: mat_id,
            property_block_id: draw.renderer.primary_property_block_id,
        };
        std::slice::from_ref(&fallback_slot)
    } else {
        return;
    };

    if slots.is_empty() {
        return;
    }
    let n_sub = draw.submeshes.len();
    let n_slot = slots.len();
    if n_sub != n_slot {
        logger::trace!(
            "mesh_asset_id={}: material slot count {} != submesh count {} (using first {} pairings only)",
            draw.renderer.mesh_asset_id,
            n_slot,
            n_sub,
            n_sub.min(n_slot),
        );
    }
    let n = n_sub.min(n_slot);
    if n == 0 {
        return;
    }

    let is_overlay = draw.renderer.layer == crate::shared::LayerType::Overlay;
    let world_space_deformed = draw.skinned
        && draw.mesh.supports_world_space_skin_deform(
            draw.skinned_renderer
                .map(|skinned| skinned.bone_transform_indices.as_slice()),
        );

    for (slot_index, (slot, &(first_index, index_count))) in slots[..n]
        .iter()
        .zip(draw.submeshes[..n].iter())
        .enumerate()
    {
        push_one_slot_draw(
            ctx,
            acc,
            &draw,
            slot,
            SubmeshSlotIndices {
                slot_index,
                first_index,
                index_count,
            },
            OverlayDeformCullFlags {
                is_overlay,
                world_space_deformed,
            },
            cache,
        );
    }
}

/// One material slot × submesh pairing: optional CPU cull, batch key, and [`WorldMeshDrawItem`] push.
fn push_one_slot_draw(
    ctx: &DrawCollectionContext<'_>,
    acc: &mut DrawCollectionAccumulator<'_>,
    draw: &StaticMeshDrawSource<'_>,
    slot: &MeshMaterialSlot,
    indices: SubmeshSlotIndices,
    flags: OverlayDeformCullFlags,
    cache: &FrameMaterialBatchCache,
) {
    let SubmeshSlotIndices {
        slot_index,
        first_index,
        index_count,
    } = indices;
    let OverlayDeformCullFlags {
        is_overlay,
        world_space_deformed,
    } = flags;
    let material_asset_id = ctx
        .scene
        .overridden_material_asset_id(
            draw.space_id,
            ctx.render_context,
            draw.skinned,
            draw.renderable_index,
            slot_index,
        )
        .unwrap_or(slot.material_asset_id);
    if index_count == 0 {
        return;
    }
    if material_asset_id < 0 {
        return;
    }
    let rigid_world_matrix = if draw.skinned {
        None
    } else if let Some(c) = ctx.culling {
        acc.cull_stats.0 += 1;
        let target = MeshCullTarget {
            scene: ctx.scene,
            space_id: draw.space_id,
            mesh: draw.mesh,
            skinned: draw.skinned,
            skinned_renderer: draw.skinned_renderer,
            node_id: draw.renderer.node_id,
        };
        match mesh_draw_passes_cpu_cull(&target, is_overlay, c, ctx.render_context) {
            Err(CpuCullFailure::Frustum) => {
                acc.cull_stats.1 += 1;
                return;
            }
            Err(CpuCullFailure::HiZ) => {
                acc.cull_stats.2 += 1;
                return;
            }
            Ok(m) => m,
        }
    } else {
        ctx.scene.world_matrix_for_render_context(
            draw.space_id,
            draw.renderer.node_id as usize,
            ctx.render_context,
            ctx.head_output_transform,
        )
    };
    let lookup_ids = MaterialPropertyLookupIds {
        material_asset_id,
        mesh_property_block_slot0: slot.property_block_id,
    };
    let batch_key = batch_key_for_slot_cached(
        material_asset_id,
        slot.property_block_id,
        draw.skinned,
        cache,
        ctx.material_dict,
        ctx.material_router,
        ctx.pipeline_property_ids,
        ctx.shader_perm,
    );
    let camera_distance_sq = if batch_key.alpha_blended {
        match rigid_world_matrix {
            Some(m) => (m.col(3).truncate() - ctx.view_origin_world).length_squared(),
            None => 0.0,
        }
    } else {
        0.0
    };
    acc.out.push(WorldMeshDrawItem {
        space_id: draw.space_id,
        node_id: draw.renderer.node_id,
        mesh_asset_id: draw.renderer.mesh_asset_id,
        slot_index,
        first_index,
        index_count,
        is_overlay,
        sorting_order: draw.renderer.sorting_order,
        skinned: draw.skinned,
        world_space_deformed,
        collect_order: 0,
        camera_distance_sq,
        lookup_ids,
        batch_key,
        rigid_world_matrix,
    });
}

/// Builds the chunk list: one entry per 128-renderer slice of static or skinned renderers per space.
fn build_chunk_specs(
    space_ids: &[RenderSpaceId],
    ctx: &DrawCollectionContext<'_>,
) -> Vec<WorldMeshChunkSpec> {
    profiling::scope!("mesh::build_chunk_specs");
    let mut chunks = Vec::new();
    for &space_id in space_ids {
        let Some(space) = ctx.scene.space(space_id) else {
            continue;
        };
        if !space.is_active {
            continue;
        }
        let n_static = space.static_mesh_renderers.len();
        let mut start = 0;
        while start < n_static {
            let end = n_static.min(start + WORLD_MESH_COLLECT_CHUNK_SIZE);
            chunks.push(WorldMeshChunkSpec {
                space_id,
                kind: ChunkKind::Static,
                range: start..end,
            });
            start = end;
        }
        let n_skinned = space.skinned_mesh_renderers.len();
        start = 0;
        while start < n_skinned {
            let end = n_skinned.min(start + WORLD_MESH_COLLECT_CHUNK_SIZE);
            chunks.push(WorldMeshChunkSpec {
                space_id,
                kind: ChunkKind::Skinned,
                range: start..end,
            });
            start = end;
        }
    }
    chunks
}

/// Collects draw items for one chunk (one 128-renderer slice of static or skinned renderers).
fn collect_chunk(
    spec: &WorldMeshChunkSpec,
    ctx: &DrawCollectionContext<'_>,
    cache: &FrameMaterialBatchCache,
    filter_masks: &HashMap<RenderSpaceId, Vec<bool>>,
) -> (Vec<WorldMeshDrawItem>, (usize, usize, usize)) {
    let mut out = Vec::new();
    let mut cull_stats = (0usize, 0usize, 0usize);

    let Some(space) = ctx.scene.space(spec.space_id) else {
        return (out, cull_stats);
    };
    if !space.is_active {
        return (out, cull_stats);
    }

    let filter_pass_mask = filter_masks.get(&spec.space_id).map(|m| m.as_slice());
    let mut acc = DrawCollectionAccumulator {
        out: &mut out,
        cull_stats: &mut cull_stats,
        filter_pass_mask,
    };

    match spec.kind {
        ChunkKind::Static => {
            for renderable_index in spec.range.clone() {
                let r = &space.static_mesh_renderers[renderable_index];
                if r.mesh_asset_id < 0 || r.node_id < 0 {
                    continue;
                }
                let Some(mesh) = ctx.mesh_pool.get_mesh(r.mesh_asset_id) else {
                    continue;
                };
                if mesh.submeshes.is_empty() {
                    continue;
                }
                push_draws_for_renderer(
                    ctx,
                    &mut acc,
                    StaticMeshDrawSource {
                        space_id: spec.space_id,
                        renderer: r,
                        renderable_index,
                        skinned: false,
                        skinned_renderer: None,
                        mesh,
                        submeshes: &mesh.submeshes,
                    },
                    cache,
                );
            }
        }
        ChunkKind::Skinned => {
            for renderable_index in spec.range.clone() {
                let skinned = &space.skinned_mesh_renderers[renderable_index];
                let r = &skinned.base;
                if r.mesh_asset_id < 0 || r.node_id < 0 {
                    continue;
                }
                let Some(mesh) = ctx.mesh_pool.get_mesh(r.mesh_asset_id) else {
                    continue;
                };
                if mesh.submeshes.is_empty() {
                    continue;
                }
                push_draws_for_renderer(
                    ctx,
                    &mut acc,
                    StaticMeshDrawSource {
                        space_id: spec.space_id,
                        renderer: r,
                        renderable_index,
                        skinned: true,
                        skinned_renderer: Some(skinned),
                        mesh,
                        submeshes: &mesh.submeshes,
                    },
                    cache,
                );
            }
        }
    }
    (out, cull_stats)
}

/// Collects draw items for one chunk of a pre-expanded [`FramePreparedRenderables`] list.
///
/// Unlike [`collect_chunk`], there is no scene walk: the prepared draws already captured every
/// valid `(renderer × slot × submesh)` tuple plus its frame-global resolution (material override,
/// submesh index range, overlay flag, skin deform flag). This per-view pass only applies filters
/// and per-view CPU culling, then builds [`WorldMeshDrawItem`]s.
fn collect_prepared_chunk(
    draws: &[FramePreparedDraw],
    ctx: &DrawCollectionContext<'_>,
    cache: &FrameMaterialBatchCache,
    filter_masks: &HashMap<RenderSpaceId, Vec<bool>>,
) -> (Vec<WorldMeshDrawItem>, (usize, usize, usize)) {
    let mut out: Vec<WorldMeshDrawItem> = Vec::with_capacity(draws.len());
    let mut cull_stats = (0usize, 0usize, 0usize);

    for d in draws {
        if let Some(filter) = ctx.transform_filter {
            let passes = match filter_masks.get(&d.space_id) {
                Some(mask) => {
                    d.node_id >= 0 && (d.node_id as usize) < mask.len() && mask[d.node_id as usize]
                }
                None => filter.passes_scene_node(ctx.scene, d.space_id, d.node_id),
            };
            if !passes {
                continue;
            }
        }

        let Some(mesh) = ctx.mesh_pool.get_mesh(d.mesh_asset_id) else {
            continue;
        };

        // Skinned renderers need a borrow into the scene for skinning palette bounds; the index
        // may go stale if the renderer list shrinks mid-frame (rare), so treat `None` as "skip".
        let skinned_renderer: Option<&SkinnedMeshRenderer> = if d.skinned {
            match ctx.scene.space(d.space_id) {
                Some(space) => match space.skinned_mesh_renderers.get(d.renderable_index) {
                    Some(sk) => Some(sk),
                    None => continue,
                },
                None => continue,
            }
        } else {
            None
        };

        let rigid_world_matrix = if let Some(c) = ctx.culling {
            cull_stats.0 += 1;
            let target = MeshCullTarget {
                scene: ctx.scene,
                space_id: d.space_id,
                mesh,
                skinned: d.skinned,
                skinned_renderer,
                node_id: d.node_id,
            };
            match mesh_draw_passes_cpu_cull(&target, d.is_overlay, c, ctx.render_context) {
                Err(CpuCullFailure::Frustum) => {
                    cull_stats.1 += 1;
                    continue;
                }
                Err(CpuCullFailure::HiZ) => {
                    cull_stats.2 += 1;
                    continue;
                }
                Ok(m) => m,
            }
        } else if d.skinned {
            None
        } else {
            ctx.scene.world_matrix_for_render_context(
                d.space_id,
                d.node_id as usize,
                ctx.render_context,
                ctx.head_output_transform,
            )
        };

        let batch_key = batch_key_for_slot_cached(
            d.material_asset_id,
            d.property_block_id,
            d.skinned,
            cache,
            ctx.material_dict,
            ctx.material_router,
            ctx.pipeline_property_ids,
            ctx.shader_perm,
        );
        let camera_distance_sq = if batch_key.alpha_blended {
            match rigid_world_matrix {
                Some(m) => (m.col(3).truncate() - ctx.view_origin_world).length_squared(),
                None => 0.0,
            }
        } else {
            0.0
        };
        out.push(WorldMeshDrawItem {
            space_id: d.space_id,
            node_id: d.node_id,
            mesh_asset_id: d.mesh_asset_id,
            slot_index: d.slot_index,
            first_index: d.first_index,
            index_count: d.index_count,
            is_overlay: d.is_overlay,
            sorting_order: d.sorting_order,
            skinned: d.skinned,
            world_space_deformed: d.world_space_deformed,
            collect_order: 0,
            camera_distance_sq,
            lookup_ids: d.lookup_ids,
            batch_key,
            rigid_world_matrix,
        });
    }

    (out, cull_stats)
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

    let mut cap_hint = 0usize;
    for space_id in &space_ids {
        let Some(space) = ctx.scene.space(*space_id) else {
            continue;
        };
        if space.is_active {
            cap_hint = cap_hint
                .saturating_add(space.static_mesh_renderers.len())
                .saturating_add(space.skinned_mesh_renderers.len());
        }
    }

    // Build per-material cache and per-space filter masks before the parallel phase. When the
    // caller already shared a frame-scope cache (multi-view paths), reuse it instead of rebuilding
    // — that is the whole point of `material_cache` on [`DrawCollectionContext`].
    let owned_cache;
    let cache: &FrameMaterialBatchCache = match ctx.material_cache {
        Some(shared) => shared,
        None => {
            owned_cache = FrameMaterialBatchCache::build_for_frame(
                ctx.scene,
                ctx.material_dict,
                ctx.material_router,
                ctx.pipeline_property_ids,
                ctx.shader_perm,
            );
            &owned_cache
        }
    };
    let filter_masks: HashMap<RenderSpaceId, Vec<bool>> = if ctx.transform_filter.is_some() {
        space_ids
            .iter()
            .copied()
            .filter_map(|sid| {
                let mask = ctx.transform_filter?.build_pass_mask(ctx.scene, sid)?;
                Some((sid, mask))
            })
            .collect()
    } else {
        HashMap::new()
    };

    // Prefer the pre-expanded dense draw list when the caller shared one (multi-view paths build
    // it once per frame). The scene-walk path below stays as the single-view fallback.
    let per_chunk: Vec<(Vec<WorldMeshDrawItem>, (usize, usize, usize))> = if let Some(prepared) =
        ctx.prepared
    {
        debug_assert_eq!(
            prepared.render_context(),
            ctx.render_context,
            "prepared renderables were built for a different render context than the per-view draw collection — material overrides would disagree"
        );
        profiling::scope!("mesh::collect_prepared");
        let prepared_chunks: Vec<&[FramePreparedDraw]> =
            prepared.draws.chunks(PREPARED_CHUNK_SIZE).collect();
        match parallelism {
            WorldMeshDrawCollectParallelism::Full => prepared_chunks
                .par_iter()
                .map(|chunk| collect_prepared_chunk(chunk, ctx, cache, &filter_masks))
                .collect(),
            WorldMeshDrawCollectParallelism::SerialInnerForNestedBatch => prepared_chunks
                .iter()
                .map(|chunk| collect_prepared_chunk(chunk, ctx, cache, &filter_masks))
                .collect(),
        }
    } else {
        let chunks = build_chunk_specs(&space_ids, ctx);
        profiling::scope!("mesh::collect");
        match parallelism {
            WorldMeshDrawCollectParallelism::Full => chunks
                .par_iter()
                .map(|spec| collect_chunk(spec, ctx, cache, &filter_masks))
                .collect(),
            WorldMeshDrawCollectParallelism::SerialInnerForNestedBatch => chunks
                .iter()
                .map(|spec| collect_chunk(spec, ctx, cache, &filter_masks))
                .collect(),
        }
    };

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
