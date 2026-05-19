//! Frame-time expansion of scene renderables into [`FramePreparedDraw`] entries.
//!
//! Walks a scene space's static and skinned renderer lists in deterministic order, performs the
//! frame-scope filters (resident mesh, non-degenerate transform), precomputes per-renderer cull
//! geometry, and emits one entry per `(renderer, material slot)` pair.

use glam::Mat4;
use hashbrown::HashSet;
use rayon::prelude::*;
use std::ops::Range;

use crate::assets::mesh::GpuMesh;
use crate::gpu_pools::MeshPool;
use crate::scene::{
    MeshMaterialSlot, MeshRendererInstanceId, RenderSpaceId, SceneCoordinator, SkinnedMeshRenderer,
    StaticMeshRenderer,
};
use crate::shared::RenderingContext;
use crate::world_mesh::culling::{
    MeshCullGeometry, MeshCullTarget, mesh_world_geometry_for_cull_with_head,
};

use crate::world_mesh::draw_prep::collect::prepared::prepared_draws_share_renderer;

use super::super::item::stacked_material_submesh_range;
use super::{FramePreparedDraw, FramePreparedRun};

const MATERIAL_KEY_SIGNATURE_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const MATERIAL_KEY_SIGNATURE_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Renderer count in one render space above which expansion fans out across Rayon chunks.
pub(in crate::world_mesh::draw_prep) const PREPARED_EXPAND_PARALLEL_MIN_RENDERERS: usize = 256;
/// Renderer slice width used by aggressive prepared-renderable expansion.
pub(in crate::world_mesh::draw_prep) const PREPARED_EXPAND_RENDERER_CHUNK_SIZE: usize = 64;

#[derive(Clone, Copy)]
enum ExpansionChunkKind {
    Static,
    Skinned,
}

#[derive(Clone)]
struct ExpansionChunkSpec {
    kind: ExpansionChunkKind,
    range: Range<usize>,
}

/// One renderable's identity and mesh handles, threaded into [`expand_renderer_slots`].
///
/// Bundles the per-renderable fields that [`expand_renderers_for_space`] has already resolved so
/// the slot expander doesn't take seven independent parameters.
struct RenderableExpansion<'a> {
    /// Render space the renderable lives in.
    space_id: RenderSpaceId,
    /// Index of the renderable within its kind-specific list (static or skinned).
    renderable_index: usize,
    /// Renderer-local identity that survives dense table reindexing.
    instance_id: MeshRendererInstanceId,
    /// Renderer record (shared base for static and skinned variants).
    renderer: &'a StaticMeshRenderer,
    /// GPU mesh resolved from the mesh pool.
    mesh: &'a GpuMesh,
    /// Whether this renderable is on the skinned path.
    skinned: bool,
    /// Whether the skinned mesh deforms into world space via the skin cache.
    world_space_deformed: bool,
    /// Whether the mesh has active blendshape weights this frame.
    blendshape_deformed: bool,
    /// Whether active blendshape tangent deltas should run for tangent-reading materials.
    tangent_blendshape_deform_active: bool,
    /// Frame-time precomputed cull geometry for the renderer (`None` for overlay spaces).
    cull_geometry: Option<MeshCullGeometry>,
}

/// Signature for an empty prepared material live set.
#[inline]
pub(in crate::world_mesh::draw_prep) const fn empty_material_key_signature() -> u64 {
    MATERIAL_KEY_SIGNATURE_OFFSET
}

#[inline]
fn mix_material_key_signature(
    mut signature: u64,
    material_asset_id: i32,
    property_block_id: Option<i32>,
) -> u64 {
    let material_bits = material_asset_id as i64 as u64;
    let property_bits = property_block_id.map_or(0x9e37_79b9_7f4a_7c15, |id| id as i64 as u64);
    for part in [
        material_bits,
        property_bits,
        material_bits.rotate_left(17) ^ property_bits.rotate_right(11),
    ] {
        signature ^= part;
        signature = signature.wrapping_mul(MATERIAL_KEY_SIGNATURE_PRIME);
    }
    signature
}

/// Walks `draws` once and refreshes renderer-run ranges plus unique material/property keys.
///
/// Runs are detected post-build instead of plumbed through the parallel expansion so the
/// multi-space worker output can be merged with `Vec::append` without per-space offset adjustment.
///
/// Returns a deterministic signature of the first-seen unique material/property live set so
/// downstream caches can prove that an unchanged material generation also has unchanged
/// membership.
pub(in crate::world_mesh::draw_prep) fn populate_runs_and_material_keys(
    draws: &[FramePreparedDraw],
    runs: &mut Vec<FramePreparedRun>,
    material_property_keys: &mut Vec<(i32, Option<i32>)>,
    seen: &mut HashSet<(i32, Option<i32>)>,
) -> u64 {
    profiling::scope!("mesh::prepared_renderables::populate_run_starts");
    runs.clear();
    material_property_keys.clear();
    seen.clear();
    if draws.is_empty() {
        return empty_material_key_signature();
    }
    let mut signature = empty_material_key_signature();
    let mut run_start = 0usize;
    let mut prev = &draws[0];
    for (idx, d) in draws.iter().enumerate() {
        let key = (d.material_asset_id, d.property_block_id);
        if seen.insert(key) {
            material_property_keys.push(key);
            signature =
                mix_material_key_signature(signature, d.material_asset_id, d.property_block_id);
        }
        if idx > 0 && !prepared_draws_share_renderer(prev, d) {
            runs.push(FramePreparedRun {
                start: run_start as u32,
                end: idx as u32,
            });
            run_start = idx;
        }
        prev = d;
    }
    runs.push(FramePreparedRun {
        start: run_start as u32,
        end: draws.len() as u32,
    });
    signature ^ (material_property_keys.len() as u64)
}

/// Upper bound on prepared draws produced by `space_id`, used to pre-size per-space output
/// buffers. The 2x multiplier reflects the typical 2-slot-per-renderer expansion observed across
/// the existing scene corpus; over-estimation is cheap (`Vec::reserve` only grows), under-estimation
/// triggers the doubling growth path.
pub(in crate::world_mesh::draw_prep) fn estimated_draw_count(
    scene: &SceneCoordinator,
    space_id: RenderSpaceId,
) -> usize {
    scene.space(space_id).map_or(0, |s| {
        s.static_mesh_renderers()
            .iter()
            .filter(|renderer| renderer.participates_in_raster_draws())
            .count()
            .saturating_add(
                s.skinned_mesh_renderers()
                    .iter()
                    .filter(|skinned| skinned.base.participates_in_raster_draws())
                    .count(),
            )
            .saturating_mul(2)
    })
}

/// Total static + skinned renderer rows in one active render space.
pub(in crate::world_mesh::draw_prep) fn renderer_count_for_space(
    scene: &SceneCoordinator,
    space_id: RenderSpaceId,
) -> usize {
    scene.space(space_id).map_or(0, |s| {
        if s.is_active() {
            s.static_mesh_renderers()
                .len()
                .saturating_add(s.skinned_mesh_renderers().len())
        } else {
            0
        }
    })
}

/// Expands every valid renderer (static and skinned) in `space_id` into `out`.
pub(in crate::world_mesh::draw_prep) fn expand_space_into(
    out: &mut Vec<FramePreparedDraw>,
    scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
) {
    profiling::scope!("mesh::prepared_renderables::expand_space");
    let Some(space) = scene.space(space_id) else {
        return;
    };
    if !space.is_active() {
        return;
    }

    let space_is_overlay = space.is_overlay();
    let mut ctx = ExpandCtx {
        out,
        scene,
        mesh_pool,
        render_context,
        space_id,
        space_is_overlay,
    };
    expand_static_list(ctx.reborrow(), space.static_mesh_renderers());
    expand_skinned_list(ctx, space.skinned_mesh_renderers());
}

/// Expands every valid renderer in `space_id`, using chunked Rayon fan-out for large spaces.
pub(in crate::world_mesh::draw_prep) fn expand_space_into_aggressive(
    out: &mut Vec<FramePreparedDraw>,
    chunk_scratch: &mut Vec<Vec<FramePreparedDraw>>,
    scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
) {
    profiling::scope!("mesh::prepared_renderables::expand_space_aggressive");
    if renderer_count_for_space(scene, space_id) < PREPARED_EXPAND_PARALLEL_MIN_RENDERERS {
        expand_space_into(out, scene, mesh_pool, render_context, space_id);
        return;
    }
    expand_space_into_parallel_chunks(
        out,
        chunk_scratch,
        scene,
        mesh_pool,
        render_context,
        space_id,
    );
}

fn expand_space_into_parallel_chunks(
    out: &mut Vec<FramePreparedDraw>,
    chunk_scratch: &mut Vec<Vec<FramePreparedDraw>>,
    scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
) {
    profiling::scope!("mesh::prepared_renderables::expand_parallel_chunks");
    let Some(space) = scene.space(space_id) else {
        return;
    };
    if !space.is_active() {
        return;
    }
    let mut specs = Vec::new();
    {
        profiling::scope!("mesh::prepared_renderables::build_renderer_chunks");
        push_expansion_chunks(
            &mut specs,
            ExpansionChunkKind::Static,
            space.static_mesh_renderers().len(),
        );
        push_expansion_chunks(
            &mut specs,
            ExpansionChunkKind::Skinned,
            space.skinned_mesh_renderers().len(),
        );
    }
    if specs.len() < 2 {
        expand_space_into(out, scene, mesh_pool, render_context, space_id);
        return;
    }

    {
        profiling::scope!("mesh::prepared_renderables::resize_chunk_scratch");
        if chunk_scratch.len() < specs.len() {
            chunk_scratch.resize_with(specs.len(), Vec::new);
        }
    }
    chunk_scratch
        .par_iter_mut()
        .take(specs.len())
        .zip(specs.par_iter())
        .for_each(|(chunk_out, spec)| {
            profiling::scope!("mesh::prepared_renderables::renderer_chunk_worker");
            chunk_out.clear();
            chunk_out.reserve(spec.range.len().saturating_mul(2));
            expand_space_chunk_into(chunk_out, scene, mesh_pool, render_context, space_id, spec);
        });

    {
        profiling::scope!("mesh::prepared_renderables::merge_renderer_chunks");
        let total = chunk_scratch
            .iter()
            .take(specs.len())
            .map(Vec::len)
            .sum::<usize>();
        out.reserve(total);
        for chunk in chunk_scratch.iter_mut().take(specs.len()) {
            out.append(chunk);
        }
    }
}

fn push_expansion_chunks(
    chunks: &mut Vec<ExpansionChunkSpec>,
    kind: ExpansionChunkKind,
    len: usize,
) {
    let mut start = 0usize;
    while start < len {
        let end = len.min(start + PREPARED_EXPAND_RENDERER_CHUNK_SIZE);
        chunks.push(ExpansionChunkSpec {
            kind,
            range: start..end,
        });
        start = end;
    }
}

/// Frame-time inputs that stay constant across all renderers in one render space.
struct ExpandCtx<'a> {
    out: &'a mut Vec<FramePreparedDraw>,
    scene: &'a SceneCoordinator,
    mesh_pool: &'a MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
    space_is_overlay: bool,
}

impl<'a> ExpandCtx<'a> {
    fn reborrow(&mut self) -> ExpandCtx<'_> {
        ExpandCtx {
            out: self.out,
            scene: self.scene,
            mesh_pool: self.mesh_pool,
            render_context: self.render_context,
            space_id: self.space_id,
            space_is_overlay: self.space_is_overlay,
        }
    }
}

fn expand_static_list(mut ctx: ExpandCtx<'_>, renderers: &[StaticMeshRenderer]) {
    profiling::scope!("mesh::prepared_renderables::expand_static_list");
    for (renderable_index, r) in renderers.iter().enumerate() {
        try_expand_one_renderer(&mut ctx, renderable_index, r, /*skinned=*/ false, None);
    }
}

fn expand_skinned_list(mut ctx: ExpandCtx<'_>, renderers: &[SkinnedMeshRenderer]) {
    profiling::scope!("mesh::prepared_renderables::expand_skinned_list");
    for (renderable_index, sk) in renderers.iter().enumerate() {
        try_expand_one_renderer(
            &mut ctx,
            renderable_index,
            &sk.base,
            /*skinned=*/ true,
            Some(sk),
        );
    }
}

fn expand_space_chunk_into(
    out: &mut Vec<FramePreparedDraw>,
    scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
    spec: &ExpansionChunkSpec,
) {
    profiling::scope!("mesh::prepared_renderables::expand_renderer_chunk");
    let Some(space) = scene.space(space_id) else {
        return;
    };
    if !space.is_active() {
        return;
    }
    let mut ctx = ExpandCtx {
        out,
        scene,
        mesh_pool,
        render_context,
        space_id,
        space_is_overlay: space.is_overlay(),
    };
    match spec.kind {
        ExpansionChunkKind::Static => {
            for renderable_index in spec.range.clone() {
                let r = &space.static_mesh_renderers()[renderable_index];
                try_expand_one_renderer(
                    &mut ctx,
                    renderable_index,
                    r,
                    /*skinned=*/ false,
                    None,
                );
            }
        }
        ExpansionChunkKind::Skinned => {
            for renderable_index in spec.range.clone() {
                let sk = &space.skinned_mesh_renderers()[renderable_index];
                try_expand_one_renderer(
                    &mut ctx,
                    renderable_index,
                    &sk.base,
                    /*skinned=*/ true,
                    Some(sk),
                );
            }
        }
    }
}

/// Runs the shared per-renderer filters and emits draws for every valid material slot.
fn try_expand_one_renderer(
    ctx: &mut ExpandCtx<'_>,
    renderable_index: usize,
    base: &StaticMeshRenderer,
    skinned: bool,
    skinned_renderer: Option<&SkinnedMeshRenderer>,
) {
    if !base.participates_in_raster_draws() || base.mesh_asset_id < 0 || base.node_id < 0 {
        return;
    }
    if ctx.scene.transform_has_degenerate_scale_for_context(
        ctx.space_id,
        base.node_id as usize,
        ctx.render_context,
    ) {
        return;
    }
    let Some(mesh) = ctx.mesh_pool.get(base.mesh_asset_id) else {
        return;
    };
    if mesh.submeshes.is_empty() {
        return;
    }

    let world_space_deformed = skinned_renderer.is_some_and(|sk| {
        mesh.supports_world_space_skin_deform(Some(sk.bone_transform_indices.as_slice()))
    });
    let blendshape_deformed = mesh.supports_active_blendshape_deform(&base.blend_shape_weights);
    let tangent_blendshape_deform_active =
        mesh.supports_active_tangent_blendshape_deform(&base.blend_shape_weights);

    let cull_geometry =
        precompute_cull_geometry(ctx, mesh, skinned, skinned_renderer, base.node_id);

    expand_renderer_slots(
        ctx.out,
        ctx.scene,
        ctx.render_context,
        RenderableExpansion {
            space_id: ctx.space_id,
            renderable_index,
            instance_id: base.instance_id,
            renderer: base,
            mesh,
            skinned,
            world_space_deformed,
            blendshape_deformed,
            tangent_blendshape_deform_active,
            cull_geometry,
        },
    );
}

/// Computes per-renderer cull geometry once per frame for non-overlay spaces.
///
/// Returns `None` when the source space is overlay (its world matrix re-roots against the
/// per-view `head_output_transform`, so the geometry is genuinely view-dependent and must stay
/// per-view). For non-overlay spaces, [`mesh_world_geometry_for_cull_with_head`] is invoked with
/// `Mat4::IDENTITY` because the matrix path it follows
/// ([`SceneCoordinator::world_matrix_for_render_context`]) only multiplies by
/// `head_output_transform` for overlay spaces.
fn precompute_cull_geometry(
    ctx: &ExpandCtx<'_>,
    mesh: &GpuMesh,
    skinned: bool,
    skinned_renderer: Option<&SkinnedMeshRenderer>,
    node_id: i32,
) -> Option<MeshCullGeometry> {
    if ctx.space_is_overlay {
        return None;
    }
    let target = MeshCullTarget {
        scene: ctx.scene,
        space_id: ctx.space_id,
        mesh,
        skinned,
        skinned_renderer,
        node_id,
    };
    Some(mesh_world_geometry_for_cull_with_head(
        &target,
        Mat4::IDENTITY,
        ctx.render_context,
    ))
}

/// Expands one renderer's material slots mapped to submesh ranges into prepared draws.
///
/// Mirrors the scene-walk path's slot resolution and override / validity guards so the per-view
/// collection path can iterate prepared draws unconditionally.
fn expand_renderer_slots(
    out: &mut Vec<FramePreparedDraw>,
    scene: &SceneCoordinator,
    render_context: RenderingContext,
    renderable: RenderableExpansion<'_>,
) {
    let RenderableExpansion {
        space_id,
        renderable_index,
        instance_id,
        renderer,
        mesh,
        skinned,
        world_space_deformed,
        blendshape_deformed,
        tangent_blendshape_deform_active,
        cull_geometry,
    } = renderable;
    let fallback_slot;
    let slots: &[MeshMaterialSlot] = if !renderer.material_slots.is_empty() {
        &renderer.material_slots
    } else if let Some(mat_id) = renderer.primary_material_asset_id {
        fallback_slot = MeshMaterialSlot {
            material_asset_id: mat_id,
            property_block_id: renderer.primary_property_block_id,
        };
        std::slice::from_ref(&fallback_slot)
    } else {
        return;
    };

    if slots.is_empty() {
        return;
    }
    let submeshes: &[(u32, u32)] = &mesh.submeshes;
    if submeshes.is_empty() {
        return;
    }

    let is_overlay = renderer.node_id >= 0
        && scene.transform_is_in_overlay_layer(space_id, renderer.node_id as usize);

    for (slot_index, slot) in slots.iter().enumerate() {
        let Some((first_index, index_count)) =
            stacked_material_submesh_range(slot_index, submeshes)
        else {
            continue;
        };
        if index_count == 0 {
            continue;
        }
        let material_asset_id = scene
            .overridden_material_asset_id(
                space_id,
                render_context,
                skinned,
                renderable_index,
                slot_index,
            )
            .unwrap_or(slot.material_asset_id);
        if material_asset_id < 0 {
            continue;
        }
        out.push(FramePreparedDraw {
            space_id,
            renderable_index,
            instance_id,
            node_id: renderer.node_id,
            mesh_asset_id: renderer.mesh_asset_id,
            is_overlay,
            sorting_order: renderer.sorting_order,
            skinned,
            world_space_deformed,
            blendshape_deformed,
            tangent_blendshape_deform_active,
            shadow_cast_mode: renderer.shadow_cast_mode,
            slot_index,
            first_index,
            index_count,
            material_asset_id,
            property_block_id: slot.property_block_id,
            cull_geometry,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_chunk(spec: &ExpansionChunkSpec, kind: ExpansionChunkKind, range: Range<usize>) {
        match (spec.kind, kind) {
            (ExpansionChunkKind::Static, ExpansionChunkKind::Static)
            | (ExpansionChunkKind::Skinned, ExpansionChunkKind::Skinned) => {}
            _ => panic!("unexpected expansion chunk kind"),
        }
        assert_eq!(spec.range, range);
    }

    #[test]
    fn expansion_chunks_preserve_static_then_skinned_order() {
        let mut specs = Vec::new();
        push_expansion_chunks(&mut specs, ExpansionChunkKind::Static, 130);
        push_expansion_chunks(&mut specs, ExpansionChunkKind::Skinned, 70);

        assert_eq!(specs.len(), 5);
        assert_chunk(&specs[0], ExpansionChunkKind::Static, 0..64);
        assert_chunk(&specs[1], ExpansionChunkKind::Static, 64..128);
        assert_chunk(&specs[2], ExpansionChunkKind::Static, 128..130);
        assert_chunk(&specs[3], ExpansionChunkKind::Skinned, 0..64);
        assert_chunk(&specs[4], ExpansionChunkKind::Skinned, 64..70);
    }
}
