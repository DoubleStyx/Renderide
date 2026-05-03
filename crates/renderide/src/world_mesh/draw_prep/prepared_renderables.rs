//! Frame-scope dense expansion of scene mesh renderables into one entry per
//! `(renderer, material slot)` pair.
//!
//! This is the Stage 3 amortization of [`super::collect::collect_and_sort_draws_with_parallelism`]:
//! every per-view collection used to walk each active render space, look up the resident
//! [`crate::assets::mesh::GpuMesh`] per renderer, expand material slots onto submesh ranges, and resolve
//! render-context material overrides -- all of which are functions of frame-global state, not the
//! view. Doing that work once per frame and reusing the dense list across every view (desktop
//! multi-view secondary render-texture cameras + main swapchain) removes the N+1 scene walk that
//! dominated frame cost.
//!
//! The cull step and [`super::item::WorldMeshDrawItem`] construction stay per-view because they
//! depend on the view's camera, filter, and Hi-Z snapshot.

use glam::Mat4;
use rayon::prelude::*;

use crate::gpu_pools::MeshPool;
use crate::scene::{MeshMaterialSlot, MeshRendererInstanceId, RenderSpaceId, SceneCoordinator};
use crate::shared::{LayerType, RenderingContext};
use crate::world_mesh::culling::{
    MeshCullGeometry, MeshCullTarget, mesh_world_geometry_for_cull_with_head,
};

use super::item::stacked_material_submesh_range;

/// One fully-resolved draw slot (renderer x material slot mapped to a submesh range) for the current frame.
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
    /// per-view cull to build [`super::super::culling::MeshCullTarget`].
    pub renderable_index: usize,
    /// Renderer-local identity used for persistent GPU skin-cache ownership.
    pub instance_id: MeshRendererInstanceId,
    /// Scene node id for rigid transform lookup and filter-mask indexing.
    pub node_id: i32,
    /// Resident mesh asset id (always matches `mesh_pool.get(...)` being `Some`).
    pub mesh_asset_id: i32,
    /// Precomputed overlay flag from the renderer's [`LayerType`].
    pub is_overlay: bool,
    /// Host-side sorting order propagated to [`super::item::WorldMeshDrawItem::sorting_order`].
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
    /// Frame-time precomputed cull geometry (world AABB + rigid world matrix), shared across all
    /// material slots of the same source renderer. `Some` when the source space is non-overlay
    /// and therefore the geometry is view-invariant; `None` for overlay spaces (their world
    /// matrix re-roots against the per-view `head_output_transform`, so cull recomputes per-view).
    pub cull_geometry: Option<MeshCullGeometry>,
}

/// Frame-scope dense list of [`FramePreparedDraw`] entries across every active render space.
///
/// Build once per frame via [`FramePreparedRenderables::build_for_frame`] and hand as a borrow to
/// every per-view [`super::collect::DrawCollectionContext`]. Per-view collection walks this list,
/// applies frustum / Hi-Z culling, and emits [`super::item::WorldMeshDrawItem`]s -- no scene
/// walk, no repeated mesh-pool lookup, no repeated material-override resolution.
pub struct FramePreparedRenderables {
    /// Active render spaces captured while building this frame snapshot.
    pub(super) active_space_ids: Vec<RenderSpaceId>,
    /// Dense expanded draws. Order is deterministic: render spaces in
    /// [`SceneCoordinator::render_space_ids`] order, then static renderers (ascending index),
    /// then skinned renderers (ascending index), then material slots in ascending index.
    pub(super) draws: Vec<FramePreparedDraw>,
    /// Indices into [`Self::draws`] marking the first slot of each renderer run. Always starts
    /// with `0` when [`Self::draws`] is non-empty. Lets per-view collection chunk the prepared
    /// list on run boundaries via [`Self::run_aligned_chunks`] so a single renderer's slots are
    /// never split across two chunks (the prior `par_chunks(PREPARED_CHUNK_SIZE)` path
    /// duplicated per-renderer CPU cull work whenever a chunk seam fell inside a renderer run).
    pub(super) run_starts: Vec<u32>,
    /// Render context used when resolving material overrides; must match the per-view contexts
    /// (the main renderer uses [`SceneCoordinator::active_main_render_context`] for every view
    /// in the same frame).
    pub(super) render_context: RenderingContext,
    /// Reused per-worker output buffers for the multi-space parallel expansion path. Outer
    /// [`Vec`] is resized to [`Self::active_space_ids`] length; each inner [`Vec`] is cleared and
    /// re-filled inside the rayon worker before [`expand_space_into`] runs. Capacities persist
    /// across frames so the steady-state path does not reallocate the per-space buffers.
    space_scratch: Vec<Vec<FramePreparedDraw>>,
}

impl FramePreparedRenderables {
    /// Empty list (no active spaces / no valid renderers); used by tests and scenes where every
    /// mesh is non-resident.
    pub fn empty(render_context: RenderingContext) -> Self {
        Self {
            active_space_ids: Vec::new(),
            draws: Vec::new(),
            run_starts: Vec::new(),
            render_context,
            space_scratch: Vec::new(),
        }
    }

    /// Builds the dense draw list for every active render space in `scene`.
    ///
    /// Per-space expansion runs in parallel via [`rayon`] and the per-space outputs are
    /// concatenated in render-space-id order. Every entry is filtered to only include draws that
    /// would survive [`super::collect::collect_chunk`]'s transform-scale, resident-mesh, and
    /// slot-validity checks -- per-view collection can iterate unconditionally without duplicating
    /// those guards.
    pub fn build_for_frame(
        scene: &SceneCoordinator,
        mesh_pool: &MeshPool,
        render_context: RenderingContext,
    ) -> Self {
        let mut out = Self::empty(render_context);
        out.rebuild_for_frame(scene, mesh_pool, render_context);
        out
    }

    /// Rebuilds this snapshot in place, reusing the `draws` and `active_space_ids` Vec
    /// capacities across frames. Same semantics and parallelization as [`Self::build_for_frame`].
    ///
    /// Pooling matters because every frame produces a fresh dense list of every renderable's
    /// material slots -- typically hundreds to thousands of entries. Allocating and freeing the
    /// backing buffer each frame shows up in `extract_frame_shared` zone profiles; clearing in
    /// place keeps the allocation count flat in steady state.
    pub fn rebuild_for_frame(
        &mut self,
        scene: &SceneCoordinator,
        mesh_pool: &MeshPool,
        render_context: RenderingContext,
    ) {
        profiling::scope!("mesh::prepared_renderables_build_for_frame");
        self.render_context = render_context;
        self.active_space_ids.clear();
        self.draws.clear();
        self.run_starts.clear();

        {
            profiling::scope!("mesh::prepared_renderables::collect_active_spaces");
            self.active_space_ids.extend(
                scene
                    .render_space_ids()
                    .filter(|id| scene.space(*id).is_some_and(|s| s.is_active)),
            );
        }

        if self.active_space_ids.is_empty() {
            return;
        }

        if self.active_space_ids.len() == 1 {
            let space_id = self.active_space_ids[0];
            {
                profiling::scope!("mesh::prepared_renderables::single_space_expand");
                self.draws.reserve(estimated_draw_count(scene, space_id));
                expand_space_into(&mut self.draws, scene, mesh_pool, render_context, space_id);
            }
            populate_run_starts(&self.draws, &mut self.run_starts);
            return;
        }

        // Reuse a long-lived per-space scratch so each frame's parallel expansion does not
        // allocate a fresh outer `Vec` (the prior `par_iter().map(...).collect()` pattern) or a
        // fresh inner `Vec` per worker (`let mut local = Vec::new();`). Capacities persist across
        // frames; only the contents get cleared and refilled.
        let mut space_scratch = std::mem::take(&mut self.space_scratch);
        {
            profiling::scope!("mesh::prepared_renderables::prepare_space_scratch");
            space_scratch.resize_with(self.active_space_ids.len(), Vec::new);
        }
        let active_space_ids = &self.active_space_ids;

        {
            profiling::scope!("mesh::prepared_renderables::parallel_expand");
            space_scratch
                .par_iter_mut()
                .zip(active_space_ids.par_iter())
                .for_each(|(out, &space_id)| {
                    out.clear();
                    let estimate = estimated_draw_count(scene, space_id);
                    if estimate > out.capacity() {
                        out.reserve(estimate - out.capacity());
                    }
                    expand_space_into(out, scene, mesh_pool, render_context, space_id);
                });
        }

        {
            profiling::scope!("mesh::prepared_renderables::merge_space_scratch");
            let total: usize = space_scratch.iter().map(Vec::len).sum();
            self.draws.reserve(total);
            for buf in &mut space_scratch {
                self.draws.append(buf);
            }
        }
        self.space_scratch = space_scratch;
        populate_run_starts(&self.draws, &mut self.run_starts);
    }

    /// Returns slices of [`Self::draws`] aligned to renderer-run boundaries with each chunk
    /// covering at least `target_chunk_size` draws (the last chunk may be smaller). Used by
    /// per-view collection so the per-renderer CPU cull and material-batch lookup happens at
    /// most once per renderer per frame even on parallel workers.
    pub(super) fn run_aligned_chunks(&self, target_chunk_size: usize) -> Vec<&[FramePreparedDraw]> {
        if self.draws.is_empty() {
            return Vec::new();
        }
        let target_chunk_size = target_chunk_size.max(1);
        let n = self.draws.len();
        let starts = self.run_starts.as_slice();
        let mut chunks: Vec<&[FramePreparedDraw]> = Vec::new();

        let mut cursor = 0usize;
        let mut next_run_idx = 1usize;
        while cursor < n {
            let target = cursor.saturating_add(target_chunk_size);
            // Walk past every run boundary that is still inside the target window so the chunk
            // ends at the smallest run boundary that meets the target chunk size.
            while next_run_idx < starts.len() && (starts[next_run_idx] as usize) < target {
                next_run_idx += 1;
            }
            let chunk_end = if next_run_idx < starts.len() {
                starts[next_run_idx] as usize
            } else {
                n
            };
            chunks.push(&self.draws[cursor..chunk_end]);
            cursor = chunk_end;
            next_run_idx += 1;
        }
        chunks
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

    /// Active render spaces captured by this prepared snapshot.
    #[inline]
    pub fn active_space_ids(&self) -> &[RenderSpaceId] {
        &self.active_space_ids
    }

    /// Iterator of `(mesh_asset_id, material_asset_id)` pairs for every prepared draw.
    #[inline]
    pub fn mesh_material_pairs(&self) -> impl Iterator<Item = (i32, i32)> + '_ {
        self.draws
            .iter()
            .map(|d| (d.mesh_asset_id, d.material_asset_id))
    }

    /// Iterator of `(material_asset_id, property_block_id)` pairs for every prepared draw.
    #[inline]
    pub fn material_property_pairs(&self) -> impl Iterator<Item = (i32, Option<i32>)> + '_ {
        self.draws
            .iter()
            .map(|d| (d.material_asset_id, d.property_block_id))
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
    /// Frame-time precomputed cull geometry for the renderer (`None` for overlay spaces).
    cull_geometry: Option<MeshCullGeometry>,
}

/// Walks `draws` once and writes the index of each renderer-run start into `run_starts` (cleared
/// on entry). Two adjacent draws share a run when `prepared_draws_share_renderer` returns `true`;
/// every other index marks a boundary. Runs are detected post-build instead of plumbed through
/// the parallel expansion so the multi-space worker output can be merged with `Vec::append` without
/// per-space offset adjustment.
fn populate_run_starts(draws: &[FramePreparedDraw], run_starts: &mut Vec<u32>) {
    profiling::scope!("mesh::prepared_renderables::populate_run_starts");
    run_starts.clear();
    if draws.is_empty() {
        return;
    }
    run_starts.push(0);
    let mut prev = &draws[0];
    for (idx, d) in draws.iter().enumerate().skip(1) {
        if !super::collect::prepared::prepared_draws_share_renderer(prev, d) {
            run_starts.push(idx as u32);
        }
        prev = d;
    }
}

/// Upper bound on prepared draws produced by `space_id`, used to pre-size per-space output
/// buffers. The 2x multiplier reflects the typical 2-slot-per-renderer expansion observed across
/// the existing scene corpus; over-estimation is cheap (`Vec::reserve` only grows), under-estimation
/// triggers the doubling growth path.
fn estimated_draw_count(scene: &SceneCoordinator, space_id: RenderSpaceId) -> usize {
    scene.space(space_id).map_or(0, |s| {
        s.static_mesh_renderers
            .len()
            .saturating_add(s.skinned_mesh_renderers.len())
            .saturating_mul(2)
    })
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

    let space_is_overlay = space.is_overlay;

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
        let Some(mesh) = mesh_pool.get(r.mesh_asset_id) else {
            continue;
        };
        if mesh.submeshes.is_empty() {
            continue;
        }
        let cull_geometry = precompute_cull_geometry(PrecomputeCullInputs {
            scene,
            space_id,
            space_is_overlay,
            mesh,
            skinned: false,
            skinned_renderer: None,
            node_id: r.node_id,
            render_context,
        });
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
                cull_geometry,
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
        let Some(mesh) = mesh_pool.get(r.mesh_asset_id) else {
            continue;
        };
        if mesh.submeshes.is_empty() {
            continue;
        }
        let world_space_deformed =
            mesh.supports_world_space_skin_deform(Some(sk.bone_transform_indices.as_slice()));
        let blendshape_deformed = mesh.supports_active_blendshape_deform(&r.blend_shape_weights);
        let cull_geometry = precompute_cull_geometry(PrecomputeCullInputs {
            scene,
            space_id,
            space_is_overlay,
            mesh,
            skinned: true,
            skinned_renderer: Some(sk),
            node_id: r.node_id,
            render_context,
        });
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
                cull_geometry,
            },
        );
    }
}

/// Bundle of inputs needed to precompute one renderer's cull geometry for the frame.
struct PrecomputeCullInputs<'a> {
    scene: &'a SceneCoordinator,
    space_id: RenderSpaceId,
    space_is_overlay: bool,
    mesh: &'a crate::assets::mesh::GpuMesh,
    skinned: bool,
    skinned_renderer: Option<&'a crate::scene::SkinnedMeshRenderer>,
    node_id: i32,
    render_context: RenderingContext,
}

/// Computes per-renderer cull geometry once per frame for non-overlay spaces.
///
/// Returns `None` when the source space is overlay (its world matrix re-roots against the
/// per-view `head_output_transform`, so the geometry is genuinely view-dependent and must stay
/// per-view). For non-overlay spaces, [`mesh_world_geometry_for_cull_with_head`] is invoked with
/// `Mat4::IDENTITY` because the matrix path it follows
/// ([`SceneCoordinator::world_matrix_for_render_context`]) only multiplies by
/// `head_output_transform` for overlay spaces.
fn precompute_cull_geometry(inputs: PrecomputeCullInputs<'_>) -> Option<MeshCullGeometry> {
    let PrecomputeCullInputs {
        scene,
        space_id,
        space_is_overlay,
        mesh,
        skinned,
        skinned_renderer,
        node_id,
        render_context,
    } = inputs;
    if space_is_overlay {
        return None;
    }
    let target = MeshCullTarget {
        scene,
        space_id,
        mesh,
        skinned,
        skinned_renderer,
        node_id,
    };
    Some(mesh_world_geometry_for_cull_with_head(
        &target,
        Mat4::IDENTITY,
        render_context,
    ))
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
            cull_geometry,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_pools::MeshPool;
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
