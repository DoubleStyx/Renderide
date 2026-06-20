//! Static and skinned mesh renderer expansion.

use glam::Mat4;

use crate::assets::mesh::GpuMesh;
use crate::gpu_pools::MeshPool;
use crate::particles::ParticleDrawParams;
use crate::scene::{
    MeshMaterialSlot, MeshRendererInstanceId, RenderSpaceId, RenderSpaceRead, SkinnedMeshRenderer,
    StaticMeshRenderer, WorldMeshSceneRead,
};
use crate::shared::{LayerType, RenderingContext};
use crate::world_mesh::culling::{
    MeshCullGeometry, MeshCullTarget, mesh_world_geometry_for_cull_with_head,
};
use crate::world_mesh::materials::normalized_material_slot;

use super::super::super::item::{MaterialStackOrder, stacked_material_submesh_range};
use super::super::FramePreparedDraw;
use super::context::ExpandCtx;

/// One renderable's identity and mesh handles, threaded into [`expand_renderer_slots`].
///
/// Bundles the per-renderable fields that [`try_expand_one_renderer`] has already resolved so the
/// slot expander doesn't take seven independent parameters.
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

/// Upper bound on prepared draws produced by `space_id`, used to pre-size per-space output
/// buffers. The 2x multiplier reflects the typical 2-slot-per-renderer expansion observed across
/// the existing scene corpus; over-estimation is cheap (`Vec::reserve` only grows), under-estimation
/// triggers the doubling growth path.
#[cfg(test)]
pub(in crate::world_mesh::draw_prep) fn estimated_draw_count(
    scene: &(impl WorldMeshSceneRead + ?Sized),
    space_id: RenderSpaceId,
) -> usize {
    scene
        .static_mesh_renderers(space_id)
        .map_or(0, |renderers| {
            renderers
                .iter()
                .filter(|renderer| renderer.emits_visible_color_draws() || renderer.casts_shadows())
                .count()
        })
        .saturating_add(
            scene
                .skinned_mesh_renderers(space_id)
                .map_or(0, |renderers| {
                    renderers
                        .iter()
                        .filter(|skinned| {
                            skinned.base.emits_visible_color_draws() || skinned.base.casts_shadows()
                        })
                        .count()
                }),
        )
        .saturating_mul(2)
}

/// Total static + skinned renderer rows in one active render space.
#[cfg(test)]
pub(super) fn renderer_count_for_space(
    scene: &(impl WorldMeshSceneRead + ?Sized),
    space_id: RenderSpaceId,
) -> usize {
    let Some(space) = scene.space(space_id) else {
        return 0;
    };
    if !space.is_active() {
        return 0;
    }
    scene
        .static_mesh_renderers(space_id)
        .map_or(0, <[StaticMeshRenderer]>::len)
        .saturating_add(
            scene
                .skinned_mesh_renderers(space_id)
                .map_or(0, <[SkinnedMeshRenderer]>::len),
        )
}

/// Expands every valid renderer (static and skinned) in `space_id` into `out`.
#[cfg(test)]
pub(in crate::world_mesh::draw_prep) fn expand_space_into<S>(
    out: &mut Vec<FramePreparedDraw>,
    scene: &S,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
) where
    S: WorldMeshSceneRead + ?Sized,
{
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
    expand_static_list(
        ctx.reborrow(),
        scene.static_mesh_renderers(space_id).unwrap_or_default(),
    );
    expand_skinned_list(
        ctx,
        scene.skinned_mesh_renderers(space_id).unwrap_or_default(),
    );
}

/// Expands one static renderer row into retained draw-template entries.
pub(in crate::world_mesh::draw_prep) fn expand_static_renderer_into<S>(
    out: &mut Vec<FramePreparedDraw>,
    scene: &S,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
    renderable_index: usize,
) where
    S: WorldMeshSceneRead + ?Sized,
{
    profiling::scope!("mesh::prepared_renderables::expand_static_renderer");
    let Some(space) = scene.space(space_id) else {
        return;
    };
    if !space.is_active() {
        return;
    }
    let Some(renderer) = scene
        .static_mesh_renderers(space_id)
        .and_then(|renderers| renderers.get(renderable_index))
    else {
        return;
    };
    let mut ctx = ExpandCtx {
        out,
        scene,
        mesh_pool,
        render_context,
        space_id,
        space_is_overlay: space.is_overlay(),
    };
    try_expand_one_renderer(&mut ctx, renderable_index, renderer, false, None);
}

/// Expands one skinned renderer row into retained draw-template entries.
pub(in crate::world_mesh::draw_prep) fn expand_skinned_renderer_into<S>(
    out: &mut Vec<FramePreparedDraw>,
    scene: &S,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
    renderable_index: usize,
) where
    S: WorldMeshSceneRead + ?Sized,
{
    profiling::scope!("mesh::prepared_renderables::expand_skinned_renderer");
    let Some(space) = scene.space(space_id) else {
        return;
    };
    if !space.is_active() {
        return;
    }
    let Some(renderer) = scene
        .skinned_mesh_renderers(space_id)
        .and_then(|renderers| renderers.get(renderable_index))
    else {
        return;
    };
    let mut ctx = ExpandCtx {
        out,
        scene,
        mesh_pool,
        render_context,
        space_id,
        space_is_overlay: space.is_overlay(),
    };
    try_expand_one_renderer(
        &mut ctx,
        renderable_index,
        &renderer.base,
        true,
        Some(renderer),
    );
}

/// Expands all static renderers in one render-space slice.
#[cfg(test)]
fn expand_static_list<S>(mut ctx: ExpandCtx<'_, S>, renderers: &[StaticMeshRenderer])
where
    S: WorldMeshSceneRead + ?Sized,
{
    profiling::scope!("mesh::prepared_renderables::expand_static_list");
    for (renderable_index, r) in renderers.iter().enumerate() {
        try_expand_one_renderer(&mut ctx, renderable_index, r, /*skinned=*/ false, None);
    }
}

/// Expands all skinned renderers in one render-space slice.
#[cfg(test)]
fn expand_skinned_list<S>(mut ctx: ExpandCtx<'_, S>, renderers: &[SkinnedMeshRenderer])
where
    S: WorldMeshSceneRead + ?Sized,
{
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

/// Runs the shared per-renderer filters and emits draws for every valid material slot.
pub(super) fn try_expand_one_renderer<S>(
    ctx: &mut ExpandCtx<'_, S>,
    renderable_index: usize,
    base: &StaticMeshRenderer,
    skinned: bool,
    skinned_renderer: Option<&SkinnedMeshRenderer>,
) where
    S: WorldMeshSceneRead + ?Sized,
{
    if (!base.emits_visible_color_draws() && !base.casts_shadows())
        || base.mesh_asset_id < 0
        || base.node_id < 0
    {
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
/// ([`WorldMeshSceneRead::world_matrix_for_render_context`]) only multiplies by
/// `head_output_transform` for overlay spaces.
fn precompute_cull_geometry<S>(
    ctx: &ExpandCtx<'_, S>,
    mesh: &GpuMesh,
    skinned: bool,
    skinned_renderer: Option<&SkinnedMeshRenderer>,
    node_id: i32,
) -> Option<MeshCullGeometry>
where
    S: WorldMeshSceneRead + ?Sized,
{
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
fn expand_renderer_slots<S>(
    out: &mut Vec<FramePreparedDraw>,
    scene: &S,
    render_context: RenderingContext,
    renderable: RenderableExpansion<'_>,
) where
    S: WorldMeshSceneRead + ?Sized,
{
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

    let special_layer = if renderer.node_id >= 0 {
        scene.transform_special_layer(space_id, renderer.node_id as usize)
    } else {
        None
    };
    let is_overlay = matches!(special_layer, Some(LayerType::Overlay));
    let is_hidden = matches!(special_layer, Some(LayerType::Hidden));

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
        let Some((material_asset_id, property_block_id)) =
            normalized_material_slot(material_asset_id, slot.property_block_id)
        else {
            continue;
        };
        out.push(FramePreparedDraw {
            space_id,
            renderable_index,
            instance_id,
            renderer_ordinal: 0,
            node_id: renderer.node_id,
            mesh_asset_id: renderer.mesh_asset_id,
            is_overlay,
            is_hidden,
            sorting_order: renderer.sorting_order,
            shadow_cast_mode: renderer.shadow_cast_mode,
            skinned,
            world_space_deformed,
            blendshape_deformed,
            tangent_blendshape_deform_active,
            slot_index,
            material_stack_order: MaterialStackOrder::from_slot_counts(
                slot_index,
                slots.len(),
                submeshes.len(),
            ),
            first_index,
            index_count,
            material_asset_id,
            property_block_id,
            cull_geometry,
            rigid_world_matrix_override: None,
            particle_draw: ParticleDrawParams::default(),
        });
    }
}
