//! Scene walk that pairs material slots with submeshes and applies optional CPU culling.
//!
//! [`collect_and_sort_world_mesh_draws`] walks each render space in parallel ([`rayon`]), merges in
//! [`SceneCoordinator::render_space_ids`] order, assigns [`WorldMeshDrawItem::collect_order`], then sorts.

use hashbrown::HashSet;

use glam::Mat4;
use rayon::prelude::*;

use crate::assets::material::{MaterialDictionary, MaterialPropertyLookupIds};
use crate::assets::mesh::GpuMesh;
use crate::materials::{MaterialPipelinePropertyIds, MaterialRouter};
use crate::pipelines::ShaderPermutation;
use crate::resources::MeshPool;
use crate::scene::{RenderSpaceId, SceneCoordinator, SkinnedMeshRenderer, StaticMeshRenderer};
use crate::shared::RenderingContext;

use super::sort::{batch_key_for_slot, sort_world_mesh_draws, sort_world_mesh_draws_serial};
use super::types::{
    resolved_material_slots, CameraTransformDrawFilter, WorldMeshDrawCollection, WorldMeshDrawItem,
};

use super::super::world_mesh_cull_eval::{
    mesh_draw_passes_cpu_cull, CpuCullFailure, MeshCullTarget,
};

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
    /// Optional CPU frustum + Hi-Z cull inputs.
    pub culling: Option<&'a super::super::world_mesh_cull::WorldMeshCullInput<'a>>,
    /// Optional per-camera node filter.
    pub transform_filter: Option<&'a CameraTransformDrawFilter>,
}

/// One static or skinned mesh renderer with its resolved [`GpuMesh`] and submesh index ranges.
struct StaticMeshDrawSource<'a> {
    space_id: RenderSpaceId,
    renderer: &'a StaticMeshRenderer,
    renderable_index: usize,
    skinned: bool,
    skinned_renderer: Option<&'a SkinnedMeshRenderer>,
    mesh: &'a GpuMesh,
    submeshes: &'a [(u32, u32)],
}

/// Mutable expansion state while expanding one space into draw items.
struct DrawCollectionAccumulator<'a> {
    out: &'a mut Vec<WorldMeshDrawItem>,
    mismatch_warned: &'a mut HashSet<i32>,
    cull_stats: &'a mut (usize, usize, usize),
    /// Precomputed filter result per node index. When `Some`, used in place of
    /// [`CameraTransformDrawFilter::passes_scene_node`] to avoid per-draw ancestor walks.
    filter_pass_mask: Option<&'a [bool]>,
}

/// How [`collect_and_sort_world_mesh_draws_with_parallelism`] parallelizes per-space collection and sorting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorldMeshDrawCollectParallelism {
    /// Per-space collection and draw sort both use rayon.
    Full,
    /// Serial per-space merge and serial sort; use when an outer `par_iter` already fans out (e.g. multiple secondary RTs).
    SerialInnerForNestedBatch,
}

/// Expands one static mesh renderer into draw items (material slots × submeshes).
///
/// `collect_order` is filled with a placeholder; [`collect_and_sort_world_mesh_draws`] assigns the
/// final stable index after per-space results are merged.
fn push_draws_for_renderer(
    ctx: &DrawCollectionContext<'_>,
    acc: &mut DrawCollectionAccumulator<'_>,
    draw: StaticMeshDrawSource<'_>,
) {
    if let Some(f) = ctx.transform_filter {
        let passes = match acc.filter_pass_mask {
            Some(mask) => {
                let nid = draw.renderer.node_id;
                nid >= 0
                    && (nid as usize) < mask.len()
                    && mask[nid as usize]
            }
            None => f.passes_scene_node(ctx.scene, draw.space_id, draw.renderer.node_id),
        };
        if !passes {
            return;
        }
    }
    let slots = resolved_material_slots(draw.renderer);
    if slots.is_empty() {
        return;
    }
    let n_sub = draw.submeshes.len();
    let n_slot = slots.len();
    if n_sub != n_slot && acc.mismatch_warned.insert(draw.renderer.mesh_asset_id) {
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

    for slot_index in 0..n {
        let slot = &slots[slot_index];
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
        let (first_index, index_count) = draw.submeshes[slot_index];
        if index_count == 0 {
            continue;
        }
        if material_asset_id < 0 {
            continue;
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
                    continue;
                }
                Err(CpuCullFailure::HiZ) => {
                    acc.cull_stats.2 += 1;
                    continue;
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
        let batch_key = batch_key_for_slot(
            material_asset_id,
            slot.property_block_id,
            draw.skinned,
            ctx.material_dict,
            ctx.material_router,
            ctx.pipeline_property_ids,
            ctx.shader_perm,
        );
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
            camera_distance_sq: 0.0,
            lookup_ids,
            batch_key,
            rigid_world_matrix,
        });
    }
}

/// Collects draws for one render space (static then skinned renderers).
fn collect_draws_for_one_space(
    space_id: RenderSpaceId,
    ctx: &DrawCollectionContext<'_>,
) -> (Vec<WorldMeshDrawItem>, (usize, usize, usize)) {
    let mut out = Vec::new();
    let mut cull_stats = (0usize, 0usize, 0usize);
    let mut mismatch_warned = HashSet::new();

    let Some(space) = ctx.scene.space(space_id) else {
        return (out, cull_stats);
    };
    if !space.is_active {
        return (out, cull_stats);
    }

    // Precompute per-node filter pass mask so `push_draws_for_renderer` skips the O(depth)
    // ancestor walk on every draw (hot in dashboard / secondary-RT paths).
    let filter_pass_mask: Option<Vec<bool>> = ctx
        .transform_filter
        .and_then(|f| f.build_pass_mask(ctx.scene, space_id));

    let mut acc = DrawCollectionAccumulator {
        out: &mut out,
        mismatch_warned: &mut mismatch_warned,
        cull_stats: &mut cull_stats,
        filter_pass_mask: filter_pass_mask.as_deref(),
    };

    for (renderable_index, r) in space.static_mesh_renderers.iter().enumerate() {
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
                space_id,
                renderer: r,
                renderable_index,
                skinned: false,
                skinned_renderer: None,
                mesh,
                submeshes: &mesh.submeshes,
            },
        );
    }
    for (renderable_index, skinned) in space.skinned_mesh_renderers.iter().enumerate() {
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
                space_id,
                renderer: r,
                renderable_index,
                skinned: true,
                skinned_renderer: Some(skinned),
                mesh,
                submeshes: &mesh.submeshes,
            },
        );
    }

    (out, cull_stats)
}

/// Collects draws from active spaces, then sorts for batching (material / pipeline boundaries).
///
/// When `culling` is [`Some`], instances outside the frustum (and optional Hi-Z) are dropped (see
/// [`mesh_draw_passes_cpu_cull`](super::super::world_mesh_cull_eval::mesh_draw_passes_cpu_cull)).
///
/// Per-space collection runs in parallel via [`rayon`] by default; results are merged in the same order as
/// [`SceneCoordinator::render_space_ids`], then [`WorldMeshDrawItem::collect_order`] is assigned for transparent sort stability.
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

    let per_space: Vec<(Vec<WorldMeshDrawItem>, (usize, usize, usize))> = match parallelism {
        WorldMeshDrawCollectParallelism::Full => space_ids
            .par_iter()
            .copied()
            .map(|space_id| collect_draws_for_one_space(space_id, ctx))
            .collect(),
        WorldMeshDrawCollectParallelism::SerialInnerForNestedBatch => space_ids
            .iter()
            .copied()
            .map(|space_id| collect_draws_for_one_space(space_id, ctx))
            .collect(),
    };

    let mut out = Vec::with_capacity(cap_hint.saturating_mul(8));
    let mut cull_stats = (0usize, 0usize, 0usize);
    for (items, cs) in per_space {
        cull_stats.0 += cs.0;
        cull_stats.1 += cs.1;
        cull_stats.2 += cs.2;
        out.extend(items);
    }

    for (i, item) in out.iter_mut().enumerate() {
        item.collect_order = i;
    }

    match parallelism {
        WorldMeshDrawCollectParallelism::Full => sort_world_mesh_draws(&mut out),
        WorldMeshDrawCollectParallelism::SerialInnerForNestedBatch => {
            sort_world_mesh_draws_serial(&mut out);
        }
    }
    WorldMeshDrawCollection {
        items: out,
        draws_pre_cull: cull_stats.0,
        draws_culled: cull_stats.1,
        draws_hi_z_culled: cull_stats.2,
    }
}
