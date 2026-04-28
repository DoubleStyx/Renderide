//! Frame-scope dense expansion of scene mesh renderables into one entry per
//! `(renderer, material slot)` pair.
//!
//! This is the Stage 3 amortization of [`super::collect::collect_and_sort_world_mesh_draws_with_parallelism`]:
//! every per-view collection used to walk each active render space, look up the resident
//! [`crate::assets::mesh::GpuMesh`] per renderer, expand material slots onto submesh ranges, and resolve
//! render-context material overrides — all of which are functions of frame-global state, not the
//! view. Doing that work once per frame and reusing the dense list across every view (desktop
//! multi-view secondary render-texture cameras + main swapchain) removes the N+1 scene walk that
//! dominated frame cost.
//!
//! The cull step and [`super::types::WorldMeshDrawItem`] construction stay per-view because they
//! depend on the view's camera, filter, and Hi-Z snapshot.

use rayon::prelude::*;

use crate::resources::MeshPool;
use crate::scene::{MeshMaterialSlot, MeshRendererInstanceId, RenderSpaceId, SceneCoordinator};
use crate::shared::{LayerType, RenderingContext};

use super::types::stacked_material_submesh_range;

/// One fully-resolved draw slot (renderer × material slot mapped to a submesh range) for the current frame.
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
    /// Renderer-local identity used for persistent GPU skin-cache ownership.
    pub instance_id: MeshRendererInstanceId,
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
    /// Cached result of [`crate::assets::mesh::GpuMesh::supports_active_blendshape_deform`].
    pub blendshape_deformed: bool,
    /// Material-slot index within the renderer's slot / primary fallback list.
    pub slot_index: usize,
    /// First index in the mesh index buffer for the selected submesh range.
    pub first_index: u32,
    /// Number of indices for this submesh draw (always `> 0`).
    pub index_count: u32,
    /// Material id after [`SceneCoordinator::overridden_material_asset_id`] resolution (always `>= 0`).
    pub material_asset_id: i32,
    /// Per-slot property block id when present (distinct from `Some` for batching).
    pub property_block_id: Option<i32>,
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
    /// then skinned renderers (ascending index), then material slots in ascending index.
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
    /// would survive [`super::collect::collect_chunk`]'s transform-scale, resident-mesh, and
    /// slot-validity checks — per-view collection can iterate unconditionally without duplicating
    /// those guards.
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

    /// Iterator of `(mesh_asset_id, material_asset_id)` pairs for every prepared draw.
    #[inline]
    pub fn mesh_material_pairs(&self) -> impl Iterator<Item = (i32, i32)> + '_ {
        self.draws
            .iter()
            .map(|d| (d.mesh_asset_id, d.material_asset_id))
    }
}

/// One renderable's identity and mesh handles, threaded into [`expand_renderer_slots`].
///
/// Bundles the per-renderable fields that `expand_space_into` has already resolved so the slot
/// expander doesn't take seven independent parameters.
struct RenderableExpansion<'a> {
    /// Render space the renderable lives in.
    space_id: RenderSpaceId,
    /// Index of the renderable within its kind-specific list (static or skinned).
    renderable_index: usize,
    /// Renderer-local identity that survives dense table reindexing.
    instance_id: MeshRendererInstanceId,
    /// Renderer record (shared base for static and skinned variants).
    renderer: &'a crate::scene::StaticMeshRenderer,
    /// GPU mesh resolved from the mesh pool.
    mesh: &'a crate::assets::mesh::GpuMesh,
    /// Whether this renderable is on the skinned path.
    skinned: bool,
    /// Whether the skinned mesh deforms into world space via the skin cache.
    world_space_deformed: bool,
    /// Whether the mesh has active blendshape weights this frame.
    blendshape_deformed: bool,
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
        if scene.transform_has_degenerate_scale_for_context(
            space_id,
            r.node_id as usize,
            render_context,
        ) {
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
            render_context,
            RenderableExpansion {
                space_id,
                renderable_index,
                instance_id: r.instance_id,
                renderer: r,
                mesh,
                skinned: false,
                world_space_deformed: false,
                blendshape_deformed: mesh.supports_active_blendshape_deform(&r.blend_shape_weights),
            },
        );
    }

    for (renderable_index, sk) in space.skinned_mesh_renderers.iter().enumerate() {
        let r = &sk.base;
        if r.mesh_asset_id < 0 || r.node_id < 0 {
            continue;
        }
        if scene.transform_has_degenerate_scale_for_context(
            space_id,
            r.node_id as usize,
            render_context,
        ) {
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
        let blendshape_deformed = mesh.supports_active_blendshape_deform(&r.blend_shape_weights);
        expand_renderer_slots(
            out,
            scene,
            render_context,
            RenderableExpansion {
                space_id,
                renderable_index,
                instance_id: r.instance_id,
                renderer: r,
                mesh,
                skinned: true,
                world_space_deformed,
                blendshape_deformed,
            },
        );
    }
}

/// Expands one renderer's material slots mapped to submesh ranges into prepared draws.
///
/// Mirrors [`super::collect::push_draws_for_renderer`]'s slot resolution and
/// [`super::collect::push_one_slot_draw`]'s override / validity guards so the per-view collection
/// path can iterate prepared draws unconditionally.
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

    let is_overlay = renderer.layer == LayerType::Overlay;

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
            slot_index,
            first_index,
            index_count,
            material_asset_id,
            property_block_id: slot.property_block_id,
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

    /// `mesh_material_pairs` is called from the compiled-render-graph pre-warm fallback that
    /// restores VR (OpenXR multiview) rendering of materials needing extended vertex streams;
    /// the accessor must exist and be empty for an empty scene.
    #[test]
    fn mesh_material_pairs_empty_scene_yields_nothing() {
        let scene = empty_scene();
        let mesh_pool = MeshPool::default_pool();
        let prepared = FramePreparedRenderables::build_for_frame(
            &scene,
            &mesh_pool,
            RenderingContext::default(),
        );
        assert_eq!(prepared.mesh_material_pairs().count(), 0);
    }
}
