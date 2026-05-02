//! Scene-walk draw collection fallback for world-mesh renderables.

use hashbrown::HashMap;

use crate::scene::{MeshMaterialSlot, RenderSpaceId, SkinnedMeshRenderer, StaticMeshRenderer};

use crate::world_mesh::culling::{CpuCullFailure, MeshCullTarget, mesh_draw_passes_cpu_cull};
use crate::world_mesh::materials::FrameMaterialBatchCache;

use super::candidate::{DrawCandidate, evaluate_draw_candidate};
use super::{
    DrawCollectionContext, front_face_for_world_matrix, world_matrix_for_local_vertex_stream,
};

use super::super::item::{
    WorldMeshDrawItem, resolved_material_slot_count, stacked_material_submesh_range,
};

/// Renders per chunk (static or skinned slice of one render space).
pub(super) const WORLD_MESH_COLLECT_CHUNK_SIZE: usize = 128;

/// Submesh index range for one material slot pairing during draw collection.
pub(crate) struct SubmeshSlotIndices {
    /// Slot index in [`StaticMeshRenderer`] material slots.
    pub slot_index: usize,
    /// First index in the mesh index buffer for this submesh.
    pub first_index: u32,
    /// Index count for this submesh draw.
    pub index_count: u32,
    /// Whether this submesh uses point-list topology.
    pub point_topology: bool,
}

/// Layer and skin deform flags that affect CPU cull and [`WorldMeshDrawItem`] fields.
pub(crate) struct OverlayDeformCullFlags {
    /// Overlay layer uses alternate cull behavior.
    pub is_overlay: bool,
    /// Skinned mesh with world-space deform from the skin cache.
    pub world_space_deformed: bool,
    /// Mesh has active blendshape weights and uses cache-backed positions.
    pub blendshape_deformed: bool,
}

/// One static or skinned mesh renderer with its resolved [`crate::assets::mesh::GpuMesh`] and submesh index ranges.
struct StaticMeshDrawSource<'a> {
    /// Render space containing the renderer.
    space_id: RenderSpaceId,
    /// Base static renderer fields.
    renderer: &'a StaticMeshRenderer,
    /// Renderer index inside its static or skinned list.
    renderable_index: usize,
    /// Renderer-local identity that survives dense table reindexing.
    instance_id: crate::scene::MeshRendererInstanceId,
    /// Whether this source comes from the skinned renderer list.
    skinned: bool,
    /// Skinned renderer data when [`Self::skinned`] is true.
    skinned_renderer: Option<&'a SkinnedMeshRenderer>,
    /// Resident mesh data.
    mesh: &'a crate::assets::mesh::GpuMesh,
    /// Submesh index ranges.
    submeshes: &'a [(u32, u32, bool)],
}

/// Mutable expansion state while expanding one chunk into draw items.
struct DrawCollectionAccumulator<'a> {
    /// Draw output buffer for the current chunk.
    out: &'a mut Vec<WorldMeshDrawItem>,
    /// Pre-cull, frustum-cull, and Hi-Z-cull counters.
    cull_stats: &'a mut (usize, usize, usize),
    /// Precomputed filter result per node index. When `Some`, used in place of
    /// [`super::super::filter::CameraTransformDrawFilter::passes_scene_node`] to avoid per-draw ancestor walks.
    filter_pass_mask: Option<&'a [bool]>,
}

/// Whether a chunk covers the static or skinned renderer list of a render space.
#[derive(Clone, Copy)]
enum ChunkKind {
    /// Static mesh renderer slice.
    Static,
    /// Skinned mesh renderer slice.
    Skinned,
}

/// One 128-renderable slice of a render space's static or skinned renderer array.
pub(super) struct WorldMeshChunkSpec {
    /// Render space containing the slice.
    space_id: RenderSpaceId,
    /// Static vs skinned list selection.
    kind: ChunkKind,
    /// Renderer index range inside the selected list.
    range: std::ops::Range<usize>,
}

/// Returns `true` when a renderer node's effective transform chain collapses object scale.
#[inline]
pub(super) fn transform_chain_has_degenerate_scale(
    ctx: &DrawCollectionContext<'_>,
    space_id: RenderSpaceId,
    node_id: i32,
) -> bool {
    node_id >= 0
        && ctx.scene.transform_has_degenerate_scale_for_context(
            space_id,
            node_id as usize,
            ctx.render_context,
        )
}

/// Expands one static mesh renderer into draw items (material slots mapped to submesh ranges).
///
/// `collect_order` is filled with a placeholder; [`super::collect_and_sort_draws`]
/// assigns the final stable index after per-chunk results are merged.
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
    if transform_chain_has_degenerate_scale(ctx, draw.space_id, draw.renderer.node_id) {
        return;
    }

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
    if n_slot > n_sub {
        logger::trace!(
            "mesh_asset_id={}: material slot count {} exceeds submesh count {}; stacking extra slots onto the last submesh",
            draw.renderer.mesh_asset_id,
            n_slot,
            n_sub,
        );
    } else if n_slot < n_sub {
        logger::trace!(
            "mesh_asset_id={}: material slot count {} is below submesh count {}; only material-backed submeshes draw",
            draw.renderer.mesh_asset_id,
            n_slot,
            n_sub,
        );
    }
    if n_sub == 0 {
        return;
    }

    let is_overlay = draw.renderer.layer == crate::shared::LayerType::Overlay;
    let world_space_deformed = draw.skinned
        && draw.mesh.supports_world_space_skin_deform(
            draw.skinned_renderer
                .map(|skinned| skinned.bone_transform_indices.as_slice()),
        );
    let blendshape_deformed = draw
        .mesh
        .supports_active_blendshape_deform(&draw.renderer.blend_shape_weights);

    for (slot_index, slot) in slots.iter().enumerate() {
        let Some((first_index, index_count, point_topology)) =
            stacked_material_submesh_range(slot_index, draw.submeshes)
        else {
            continue;
        };
        push_one_slot_draw(
            ctx,
            acc,
            &draw,
            slot,
            SubmeshSlotIndices {
                slot_index,
                first_index,
                index_count,
                point_topology,
            },
            OverlayDeformCullFlags {
                is_overlay,
                world_space_deformed,
                blendshape_deformed,
            },
            cache,
        );
    }
}

/// One material slot mapped to a submesh range: optional CPU cull, batch key, and [`WorldMeshDrawItem`] push.
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
        point_topology,
    } = indices;
    let OverlayDeformCullFlags {
        is_overlay,
        world_space_deformed,
        blendshape_deformed,
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
    if index_count == 0 || material_asset_id < 0 {
        return;
    }
    let mut rigid_world_matrix = None;
    if !draw.skinned
        && let Some(c) = ctx.culling
    {
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
            Ok(m) => {
                rigid_world_matrix = m;
            }
        }
    }
    if !world_space_deformed && rigid_world_matrix.is_none() {
        rigid_world_matrix =
            world_matrix_for_local_vertex_stream(ctx, draw.space_id, draw.renderer.node_id);
    }
    let front_face = front_face_for_world_matrix(rigid_world_matrix);
    let alpha_distance_sq = rigid_world_matrix.map_or(0.0, |m| {
        (m.col(3).truncate() - ctx.view_origin_world).length_squared()
    });
    let candidate = DrawCandidate {
        space_id: draw.space_id,
        node_id: draw.renderer.node_id,
        renderable_index: draw.renderable_index,
        instance_id: draw.instance_id,
        mesh_asset_id: draw.renderer.mesh_asset_id,
        slot_index,
        first_index,
        index_count,
        point_topology,
        is_overlay,
        sorting_order: draw.renderer.sorting_order,
        skinned: draw.skinned,
        world_space_deformed,
        blendshape_deformed,
        material_asset_id,
        property_block_id: slot.property_block_id,
    };
    if let Some(item) = evaluate_draw_candidate(
        ctx,
        cache,
        candidate,
        front_face,
        rigid_world_matrix,
        alpha_distance_sq,
    ) {
        acc.out.push(item);
    }
}

/// Builds the chunk list: one entry per 128-renderer slice of static or skinned renderers per space.
pub(super) fn build_chunk_specs(
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
pub(super) fn collect_chunk(
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

    let filter_pass_mask = filter_masks.get(&spec.space_id).map(Vec::as_slice);
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
                let Some(mesh) = ctx.mesh_pool.get(r.mesh_asset_id) else {
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
                        instance_id: r.instance_id,
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
                let Some(mesh) = ctx.mesh_pool.get(r.mesh_asset_id) else {
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
                        instance_id: r.instance_id,
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

/// Upper bound on expanded draw slots across active render spaces (capacity hint for the output vec).
pub(super) fn estimate_active_renderable_count(
    space_ids: &[RenderSpaceId],
    ctx: &DrawCollectionContext<'_>,
) -> usize {
    let mut cap_hint = 0usize;
    for space_id in space_ids {
        let Some(space) = ctx.scene.space(*space_id) else {
            continue;
        };
        if !space.is_active {
            continue;
        }
        for renderer in &space.static_mesh_renderers {
            if renderer.mesh_asset_id < 0 || renderer.node_id < 0 {
                continue;
            }
            if ctx
                .mesh_pool
                .get(renderer.mesh_asset_id)
                .is_some_and(|mesh| !mesh.submeshes.is_empty())
            {
                cap_hint = cap_hint.saturating_add(resolved_material_slot_count(renderer));
            }
        }
        for skinned in &space.skinned_mesh_renderers {
            let renderer = &skinned.base;
            if renderer.mesh_asset_id < 0 || renderer.node_id < 0 {
                continue;
            }
            if ctx
                .mesh_pool
                .get(renderer.mesh_asset_id)
                .is_some_and(|mesh| !mesh.submeshes.is_empty())
            {
                cap_hint = cap_hint.saturating_add(resolved_material_slot_count(renderer));
            }
        }
    }
    cap_hint
}
