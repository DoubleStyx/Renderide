//! Scene-walk draw collection fallback for world-mesh renderables.

mod cull_cache;
mod per_renderer;
mod per_slot;

use crate::scene::{RenderSpaceId, SkinnedMeshRenderer, StaticMeshRenderer};

use super::super::item::{MaterialStackOrder, WorldMeshDrawItem, resolved_material_slot_count};
use super::{CollectState, DrawCollectionInputs};

use cull_cache::CachedCull;
use per_renderer::push_draws_for_renderer;

/// Renders per chunk (static or skinned slice of one render space).
///
/// 64 keeps medium render-space slices split across workers under the aggressive fan-out policy.
pub(super) const WORLD_MESH_COLLECT_CHUNK_SIZE: usize = 32;

/// Submesh index range for one material slot pairing during draw collection.
struct SubmeshSlotIndices {
    /// Slot index in [`StaticMeshRenderer`] material slots.
    pub slot_index: usize,
    /// Material-stack ordering marker when this slot reuses the final submesh.
    pub material_stack_order: Option<MaterialStackOrder>,
    /// First index in the mesh index buffer for this submesh.
    pub first_index: u32,
    /// Index count for this submesh draw.
    pub index_count: u32,
}

/// Layer and skin deform flags that affect CPU cull and [`WorldMeshDrawItem`] fields, paired with
/// the cached per-renderer CPU cull outcome shared across the renderer's material slots.
struct OverlayDeformCullFlags<'a> {
    /// Overlay layer uses alternate cull behavior.
    pub is_overlay: bool,
    /// Skinned mesh with world-space deform from the skin cache.
    pub world_space_deformed: bool,
    /// Mesh has active blendshape weights and uses cache-backed positions.
    pub blendshape_deformed: bool,
    /// Active blendshape tangent deltas should run when a material reads tangents.
    pub tangent_blendshape_deform_active: bool,
    /// Cached cull outcome for this renderer; `None` when the cull was skipped (skinned or no
    /// culling pass active).
    pub cached_cull: Option<&'a CachedCull>,
}

/// One static or skinned mesh renderer with its resolved [`crate::assets::mesh::GpuMesh`] and submesh index ranges.
struct StaticMeshDrawSource<'a> {
    /// Render space containing the renderer.
    pub space_id: RenderSpaceId,
    /// Base static renderer fields.
    pub renderer: &'a StaticMeshRenderer,
    /// Renderer index inside its static or skinned list.
    pub renderable_index: usize,
    /// Renderer-local identity that survives dense table reindexing.
    pub instance_id: crate::scene::MeshRendererInstanceId,
    /// Whether this source comes from the skinned renderer list.
    pub skinned: bool,
    /// Skinned renderer data when [`Self::skinned`] is true.
    pub skinned_renderer: Option<&'a SkinnedMeshRenderer>,
    /// Resident mesh data.
    pub mesh: &'a crate::assets::mesh::GpuMesh,
    /// Submesh index ranges.
    pub submeshes: &'a [(u32, u32)],
}

/// Mutable expansion state while expanding one chunk into draw items.
struct DrawCollectionAccumulator<'a> {
    /// Draw output buffer for the current chunk.
    pub out: &'a mut Vec<WorldMeshDrawItem>,
    /// Pre-cull, frustum-cull, and Hi-Z-cull counters.
    pub cull_stats: &'a mut (usize, usize, usize),
    /// Precomputed filter result per node index. When `Some`, used in place of
    /// [`super::super::filter::CameraTransformDrawFilter::passes_scene_node`] to avoid per-draw ancestor walks.
    pub filter_pass_mask: Option<&'a [bool]>,
}

/// Whether a chunk covers the static or skinned renderer list of a render space.
#[derive(Clone, Copy)]
enum ChunkKind {
    /// Static mesh renderer slice.
    Static,
    /// Skinned mesh renderer slice.
    Skinned,
}

/// One renderer slice of a render space's static or skinned renderer array.
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
    ctx: &DrawCollectionInputs<'_>,
    space_id: RenderSpaceId,
    node_id: i32,
) -> bool {
    node_id >= 0
        && ctx
            .scene_assets
            .scene
            .transform_has_degenerate_scale_for_context(
                space_id,
                node_id as usize,
                ctx.view.render_context,
            )
}

/// Builds the chunk list: one entry per 128-renderer slice of static or skinned renderers per space.
pub(super) fn build_chunk_specs(
    space_ids: &[RenderSpaceId],
    ctx: &DrawCollectionInputs<'_>,
) -> Vec<WorldMeshChunkSpec> {
    profiling::scope!("mesh::build_chunk_specs");
    let mut chunks = Vec::new();
    for &space_id in space_ids {
        let Some(space) = ctx.scene_assets.scene.space(space_id) else {
            continue;
        };
        if !space.is_active() {
            continue;
        }
        let n_static = space.static_mesh_renderers().len();
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
        let n_skinned = space.skinned_mesh_renderers().len();
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
    ctx: &DrawCollectionInputs<'_>,
    state: CollectState<'_>,
) -> (Vec<WorldMeshDrawItem>, (usize, usize, usize)) {
    let mut out = Vec::new();
    let mut cull_stats = (0usize, 0usize, 0usize);

    let Some(space) = ctx.scene_assets.scene.space(spec.space_id) else {
        return (out, cull_stats);
    };
    if !space.is_active() {
        return (out, cull_stats);
    }

    let filter_pass_mask = state.filter_masks.get(&spec.space_id).map(Vec::as_slice);
    let mut acc = DrawCollectionAccumulator {
        out: &mut out,
        cull_stats: &mut cull_stats,
        filter_pass_mask,
    };

    match spec.kind {
        ChunkKind::Static => {
            for renderable_index in spec.range.clone() {
                let r = &space.static_mesh_renderers()[renderable_index];
                let Some(source) = static_draw_source(ctx, spec.space_id, renderable_index, r)
                else {
                    continue;
                };
                if !state
                    .lod_visibility
                    .renderer_visible_by_instance(source.space_id, source.instance_id)
                {
                    continue;
                }
                push_draws_for_renderer(ctx, &mut acc, source, state.cache);
            }
        }
        ChunkKind::Skinned => {
            for renderable_index in spec.range.clone() {
                let skinned = &space.skinned_mesh_renderers()[renderable_index];
                let Some(source) =
                    skinned_draw_source(ctx, spec.space_id, renderable_index, skinned)
                else {
                    continue;
                };
                if !state
                    .lod_visibility
                    .renderer_visible_by_instance(source.space_id, source.instance_id)
                {
                    continue;
                }
                push_draws_for_renderer(ctx, &mut acc, source, state.cache);
            }
        }
    }
    (out, cull_stats)
}

/// Builds a [`StaticMeshDrawSource`] from a static renderer entry, or returns `None` if the
/// renderer is filtered out by mesh availability or trivial validity checks.
fn static_draw_source<'a>(
    ctx: &DrawCollectionInputs<'a>,
    space_id: RenderSpaceId,
    renderable_index: usize,
    r: &'a StaticMeshRenderer,
) -> Option<StaticMeshDrawSource<'a>> {
    if !r.emits_visible_color_draws() || r.mesh_asset_id < 0 || r.node_id < 0 {
        return None;
    }
    let mesh = ctx.scene_assets.mesh_pool.get(r.mesh_asset_id)?;
    if mesh.submeshes.is_empty() {
        return None;
    }
    Some(StaticMeshDrawSource {
        space_id,
        renderer: r,
        renderable_index,
        instance_id: r.instance_id,
        skinned: false,
        skinned_renderer: None,
        mesh,
        submeshes: &mesh.submeshes,
    })
}

/// Builds a [`StaticMeshDrawSource`] from a skinned renderer entry, or returns `None` when filtered out.
fn skinned_draw_source<'a>(
    ctx: &DrawCollectionInputs<'a>,
    space_id: RenderSpaceId,
    renderable_index: usize,
    sk: &'a SkinnedMeshRenderer,
) -> Option<StaticMeshDrawSource<'a>> {
    let r = &sk.base;
    if !r.emits_visible_color_draws() || r.mesh_asset_id < 0 || r.node_id < 0 {
        return None;
    }
    let mesh = ctx.scene_assets.mesh_pool.get(r.mesh_asset_id)?;
    if mesh.submeshes.is_empty() {
        return None;
    }
    Some(StaticMeshDrawSource {
        space_id,
        renderer: r,
        renderable_index,
        instance_id: r.instance_id,
        skinned: true,
        skinned_renderer: Some(sk),
        mesh,
        submeshes: &mesh.submeshes,
    })
}

/// Upper bound on expanded draw slots across active render spaces (capacity hint for the output vec).
pub(super) fn estimate_active_renderable_count(
    space_ids: &[RenderSpaceId],
    ctx: &DrawCollectionInputs<'_>,
) -> usize {
    let mut cap_hint = 0usize;
    for space_id in space_ids {
        let Some(space) = ctx.scene_assets.scene.space(*space_id) else {
            continue;
        };
        if !space.is_active() {
            continue;
        }
        for renderer in space.static_mesh_renderers() {
            if !renderer.emits_visible_color_draws()
                || renderer.mesh_asset_id < 0
                || renderer.node_id < 0
            {
                continue;
            }
            if ctx
                .scene_assets
                .mesh_pool
                .get(renderer.mesh_asset_id)
                .is_some_and(|mesh| !mesh.submeshes.is_empty())
            {
                cap_hint = cap_hint.saturating_add(resolved_material_slot_count(renderer));
            }
        }
        for skinned in space.skinned_mesh_renderers() {
            let renderer = &skinned.base;
            if !renderer.emits_visible_color_draws()
                || renderer.mesh_asset_id < 0
                || renderer.node_id < 0
            {
                continue;
            }
            if ctx
                .scene_assets
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
