//! Frame-scope dense expansion of scene mesh renderables into one entry per
//! `(renderer, submesh slot)` pair.
//!
//! This is the Stage 3 amortization of [`super::collect::collect_and_sort_world_mesh_draws_with_parallelism`]:
//! every per-view collection used to walk each active render space, look up the resident
//! [`crate::assets::mesh::GpuMesh`] per renderer, expand material slots × submeshes, and resolve
//! render-context material overrides — all of which are functions of frame-global state, not the
//! view. Doing that work once per frame and reusing the dense list across every view (desktop
//! multi-view secondary render-texture cameras + main swapchain) removes the N+1 scene walk that
//! dominated frame cost.
//!
//! The cull step and [`super::types::WorldMeshDrawItem`] construction stay per-view because they
//! depend on the view's camera, filter, and Hi-Z snapshot.

use rayon::prelude::*;

use crate::assets::material::MaterialPropertyLookupIds;
use crate::resources::MeshPool;
use crate::scene::{MeshMaterialSlot, RenderSpaceId, SceneCoordinator};
use crate::shared::{LayerType, RenderingContext};

/// One fully-resolved draw slot (renderer × material slot × submesh) for the current frame.
///
/// All fields here are functions of `(scene, mesh_pool, render_context)` and are therefore safe
/// to share across every view in a frame. Per-view data (camera transform, frustum / Hi-Z cull
/// outcome, transparent sort distance) is computed while consuming this list, not here.
///
/// [`Self::skinned`] implicitly selects which renderer list [`Self::renderable_index`] targets
/// ([`crate::scene::RenderSpaceState::static_mesh_renderers`] when `false`,
/// [`crate::scene::RenderSpaceState::skinned_mesh_renderers`] when `true`).
#[derive(Clone, Debug)]
pub(super) struct FramePreparedDraw {
    /// Host render space that owns the source renderer.
    pub space_id: RenderSpaceId,
    /// Index into the static or skinned renderer list (selected by [`Self::skinned`]), used by
    /// per-view cull to build [`super::super::world_mesh_cull_eval::MeshCullTarget`].
    pub renderable_index: usize,
    /// Scene node id for rigid transform lookup and filter-mask indexing.
    pub node_id: i32,
    /// Resident mesh asset id (always matches `mesh_pool.get_mesh(...)` being `Some`).
    pub mesh_asset_id: i32,
    /// Precomputed overlay flag from the renderer's [`LayerType`].
    pub is_overlay: bool,
    /// Host-side sorting order propagated to [`super::types::WorldMeshDrawItem::sorting_order`].
    pub sorting_order: i32,
    /// `true` when the source came from the skinned renderer list.
    pub skinned: bool,
    /// Cached result of [`crate::assets::mesh::GpuMesh::supports_world_space_skin_deform`] for
    /// skinned renderers (resolved once per frame against the mesh's bone layout).
    pub world_space_deformed: bool,
    /// Material-slot index within the renderer's slot / primary fallback list.
    pub slot_index: usize,
    /// First index into [`crate::assets::mesh::GpuMesh::submeshes`] for this draw.
    pub first_index: u32,
    /// Number of indices for this submesh draw (always `> 0`).
    pub index_count: u32,
    /// Material id after [`SceneCoordinator::overridden_material_asset_id`] resolution (always `>= 0`).
    pub material_asset_id: i32,
    /// Per-slot property block id when present (distinct from `Some` for batching).
    pub property_block_id: Option<i32>,
    /// Material / property-block lookup ids for [`super::sort::batch_key_for_slot_cached`].
    pub lookup_ids: MaterialPropertyLookupIds,
}

/// Frame-scope dense list of [`FramePreparedDraw`] entries across every active render space.
///
/// Build once per frame via [`FramePreparedRenderables::build_for_frame`] and hand as a borrow to
/// every per-view [`super::collect::DrawCollectionContext`]. Per-view collection walks this list,
/// applies frustum / Hi-Z culling, and emits [`super::types::WorldMeshDrawItem`]s — no scene
/// walk, no repeated mesh-pool lookup, no repeated material-override resolution.
pub struct FramePreparedRenderables {
    /// Dense expanded draws. Order is deterministic: render spaces in
    /// [`SceneCoordinator::render_space_ids`] order, then static renderers (ascending index),
    /// then skinned renderers (ascending index), then slot / submesh pairs in ascending index.
    pub(super) draws: Vec<FramePreparedDraw>,
    /// Render context used when resolving material overrides; must match the per-view contexts
    /// (the main renderer uses [`SceneCoordinator::active_main_render_context`] for every view
    /// in the same frame).
    pub(super) render_context: RenderingContext,
}

impl FramePreparedRenderables {
    /// Empty list (no active spaces / no valid renderers); used by tests and scenes where every
    /// mesh is non-resident.
    pub fn empty(render_context: RenderingContext) -> Self {
        Self {
            draws: Vec::new(),
            render_context,
        }
    }

    /// Builds the dense draw list for every active render space in `scene`.
    ///
    /// Per-space expansion runs in parallel via [`rayon`] and the per-space outputs are
    /// concatenated in render-space-id order. Every entry is filtered to only include draws that
    /// would survive [`super::collect::collect_chunk`]'s resident-mesh / slot-validity checks —
    /// per-view collection can iterate unconditionally without duplicating those guards.
    pub fn build_for_frame(
        scene: &SceneCoordinator,
        mesh_pool: &MeshPool,
        render_context: RenderingContext,
    ) -> Self {
        profiling::scope!("mesh::prepared_renderables_build_for_frame");
        let active_space_ids: Vec<RenderSpaceId> = scene
            .render_space_ids()
            .filter(|id| scene.space(*id).map(|s| s.is_active).unwrap_or(false))
            .collect();

        if active_space_ids.is_empty() {
            return Self::empty(render_context);
        }

        if active_space_ids.len() == 1 {
            let mut draws = Vec::new();
            expand_space_into(
                &mut draws,
                scene,
                mesh_pool,
                render_context,
                active_space_ids[0],
            );
            return Self {
                draws,
                render_context,
            };
        }

        let per_space: Vec<Vec<FramePreparedDraw>> = active_space_ids
            .par_iter()
            .map(|&space_id| {
                let mut local = Vec::new();
                expand_space_into(&mut local, scene, mesh_pool, render_context, space_id);
                local
            })
            .collect();

        let total: usize = per_space.iter().map(Vec::len).sum();
        let mut draws = Vec::with_capacity(total);
        for mut local in per_space {
            draws.append(&mut local);
        }
        Self {
            draws,
            render_context,
        }
    }

    /// Number of expanded draws across all active render spaces.
    #[inline]
    pub fn len(&self) -> usize {
        self.draws.len()
    }

    /// `true` when no renderers expanded to any draw (no active space, no resident meshes).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.draws.is_empty()
    }

    /// Render context the list was built against (used for `debug_assert` parity with the
    /// per-view [`super::collect::DrawCollectionContext::render_context`] so material-override
    /// resolution matches downstream culling).
    #[inline]
    pub fn render_context(&self) -> RenderingContext {
        self.render_context
    }
}

/// Expands every valid renderer (static and skinned) in `space_id` into `out`.
fn expand_space_into(
    out: &mut Vec<FramePreparedDraw>,
    scene: &SceneCoordinator,
    mesh_pool: &MeshPool,
    render_context: RenderingContext,
    space_id: RenderSpaceId,
) {
    let Some(space) = scene.space(space_id) else {
        return;
    };
    if !space.is_active {
        return;
    }

    for (renderable_index, r) in space.static_mesh_renderers.iter().enumerate() {
        if r.mesh_asset_id < 0 || r.node_id < 0 {
            continue;
        }
        let Some(mesh) = mesh_pool.get_mesh(r.mesh_asset_id) else {
            continue;
        };
        if mesh.submeshes.is_empty() {
            continue;
        }
        expand_renderer_slots(
            out,
            scene,
            space_id,
            renderable_index,
            r,
            mesh,
            render_context,
            /* skinned = */ false,
            /* world_space_deformed = */ false,
        );
    }

    for (renderable_index, sk) in space.skinned_mesh_renderers.iter().enumerate() {
        let r = &sk.base;
        if r.mesh_asset_id < 0 || r.node_id < 0 {
            continue;
        }
        let Some(mesh) = mesh_pool.get_mesh(r.mesh_asset_id) else {
            continue;
        };
        if mesh.submeshes.is_empty() {
            continue;
        }
        let world_space_deformed =
            mesh.supports_world_space_skin_deform(Some(sk.bone_transform_indices.as_slice()));
        expand_renderer_slots(
            out,
            scene,
            space_id,
            renderable_index,
            r,
            mesh,
            render_context,
            /* skinned = */ true,
            world_space_deformed,
        );
    }
}

/// Expands one renderer's material slots × submeshes into prepared draws.
///
/// Mirrors [`super::collect::push_draws_for_renderer`]'s slot resolution and
/// [`super::collect::push_one_slot_draw`]'s override / validity guards so the per-view collection
/// path can iterate prepared draws unconditionally.
#[allow(clippy::too_many_arguments)]
fn expand_renderer_slots(
    out: &mut Vec<FramePreparedDraw>,
    scene: &SceneCoordinator,
    space_id: RenderSpaceId,
    renderable_index: usize,
    renderer: &crate::scene::StaticMeshRenderer,
    mesh: &crate::assets::mesh::GpuMesh,
    render_context: RenderingContext,
    skinned: bool,
    world_space_deformed: bool,
) {
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
    let n = submeshes.len().min(slots.len());
    if n == 0 {
        return;
    }

    let is_overlay = renderer.layer == LayerType::Overlay;

    for (slot_index, (slot, &(first_index, index_count))) in
        slots[..n].iter().zip(submeshes[..n].iter()).enumerate()
    {
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
        let lookup_ids = MaterialPropertyLookupIds {
            material_asset_id,
            mesh_property_block_slot0: slot.property_block_id,
        };
        out.push(FramePreparedDraw {
            space_id,
            renderable_index,
            node_id: renderer.node_id,
            mesh_asset_id: renderer.mesh_asset_id,
            is_overlay,
            sorting_order: renderer.sorting_order,
            skinned,
            world_space_deformed,
            slot_index,
            first_index,
            index_count,
            material_asset_id,
            property_block_id: slot.property_block_id,
            lookup_ids,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::MeshPool;
    use crate::scene::{RenderSpaceId, SceneCoordinator};
    use crate::shared::RenderTransform;

    fn empty_scene() -> SceneCoordinator {
        SceneCoordinator::new()
    }

    #[test]
    fn build_for_frame_on_empty_scene_is_empty() {
        let scene = empty_scene();
        let mesh_pool = MeshPool::default_pool();
        let prepared = FramePreparedRenderables::build_for_frame(
            &scene,
            &mesh_pool,
            RenderingContext::default(),
        );
        assert!(prepared.is_empty());
        assert_eq!(prepared.len(), 0);
    }

    /// Active space with no mesh renderers still produces an empty prepared list.
    #[test]
    fn build_for_frame_with_empty_active_space_is_empty() {
        let mut scene = empty_scene();
        scene.test_seed_space_identity_worlds(
            RenderSpaceId(1),
            vec![RenderTransform::default()],
            vec![-1],
        );
        let mesh_pool = MeshPool::default_pool();
        let prepared = FramePreparedRenderables::build_for_frame(
            &scene,
            &mesh_pool,
            RenderingContext::default(),
        );
        assert!(prepared.is_empty());
    }
}
