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

mod expand;

use hashbrown::HashSet;
#[cfg(test)]
use rayon::prelude::*;

#[cfg(test)]
use crate::gpu_pools::MeshPool;
#[cfg(test)]
use crate::scene::SceneCoordinator;
use crate::scene::{MeshRendererInstanceId, RenderSpaceId};
use crate::shared::RenderingContext;
use crate::world_mesh::culling::MeshCullGeometry;

use expand::{empty_material_key_signature, populate_runs_and_material_keys};

pub(in crate::world_mesh::draw_prep) use expand::estimated_draw_count;
#[cfg(test)]
pub(in crate::world_mesh::draw_prep) use expand::expand_space_into;
pub(in crate::world_mesh::draw_prep) use expand::expand_space_into_aggressive;

/// Target draw count for one prepared renderer-run chunk.
pub(super) const PREPARED_RUN_CHUNK_DRAW_TARGET: usize = 64;

/// One fully-resolved draw slot (renderer x material slot mapped to a submesh range) for the current frame.
///
/// All fields here are functions of `(scene, mesh_pool, render_context)` and are therefore safe
/// to share across every view in a frame. Per-view data (camera transform, frustum / Hi-Z cull
/// outcome, transparent sort distance) is computed while consuming this list, not here.
///
/// [`Self::skinned`] implicitly selects which renderer list [`Self::renderable_index`] targets
/// (static renderers when `false`, skinned renderers when `true`).
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
    /// Precomputed overlay flag from the renderer's inherited layer state.
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
    /// Cached active tangent-blendshape state used when a material needs tangent-space shading.
    pub tangent_blendshape_deform_active: bool,
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

/// Contiguous range of [`FramePreparedRenderables::draws`] produced by one source renderer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FramePreparedRun {
    /// First draw index in this renderer run.
    pub start: u32,
    /// One-past-last draw index in this renderer run.
    pub end: u32,
}

/// Contiguous range of [`FramePreparedRenderables::runs`] consumed as one collection task.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FramePreparedRunChunk {
    /// First run index in this chunk.
    start: usize,
    /// One-past-last run index in this chunk.
    end: usize,
}

/// Rebuilds cached run-chunk ranges from renderer-run metadata.
fn populate_run_chunks(
    runs: &[FramePreparedRun],
    run_chunks: &mut Vec<FramePreparedRunChunk>,
    target_chunk_size: usize,
) {
    run_chunks.clear();
    if runs.is_empty() {
        return;
    }
    let target_chunk_size = target_chunk_size.max(1);
    let mut run_start = 0usize;
    while run_start < runs.len() {
        let draw_start = runs[run_start].start as usize;
        let mut run_end = run_start + 1;
        while run_end < runs.len()
            && (runs[run_end - 1].end as usize).saturating_sub(draw_start) < target_chunk_size
        {
            run_end += 1;
        }
        run_chunks.push(FramePreparedRunChunk {
            start: run_start,
            end: run_end,
        });
        run_start = run_end;
    }
}

/// Frame-scope dense list of [`FramePreparedDraw`] entries across every active render space.
///
/// Build once per frame via [`FramePreparedRenderables::build_for_frame`] and hand as a borrow to
/// every per-view [`super::collect::DrawCollectionContext`]. Per-view collection walks this list,
/// applies frustum / Hi-Z culling, and emits [`super::item::WorldMeshDrawItem`]s -- no scene
/// walk, no repeated mesh-pool lookup, no repeated material-override resolution.
pub struct FramePreparedRenderables {
    /// Active render spaces captured while building this frame snapshot.
    active_space_ids: Vec<RenderSpaceId>,
    /// Dense expanded draws. Order is deterministic: render spaces in
    /// [`SceneCoordinator::render_space_ids`] order, then static renderers (ascending index),
    /// then skinned renderers (ascending index), then material slots in ascending index.
    draws: Vec<FramePreparedDraw>,
    /// Contiguous renderer runs in [`Self::draws`]. Lets per-view collection chunk the prepared
    /// list on run boundaries and then consume precomputed run ranges directly instead of
    /// rediscovering boundaries inside every view/chunk.
    runs: Vec<FramePreparedRun>,
    /// Cached chunks over [`Self::runs`] so per-view collection can fan out without allocating a
    /// chunk-list vector per view.
    run_chunks: Vec<FramePreparedRunChunk>,
    /// First-seen unique `(material_asset_id, property_block_id)` keys referenced by
    /// [`Self::draws`]. Material caches consume this list once per shader permutation instead of
    /// materializing and deduping every prepared draw.
    material_property_keys: Vec<(i32, Option<i32>)>,
    /// Deterministic signature of [`Self::material_property_keys`] membership and order.
    material_property_key_signature: u64,
    /// Render context used when resolving material overrides; must match the per-view context.
    render_context: RenderingContext,
    /// Reused per-worker output buffers for the multi-space parallel expansion path. Outer
    /// [`Vec`] is resized to [`Self::active_space_ids`] length; each inner [`Vec`] is cleared and
    /// re-filled inside the rayon worker before [`expand_space_into`] runs. Capacities persist
    /// across frames so the steady-state path does not reallocate the per-space buffers.
    #[cfg(test)]
    space_scratch: Vec<Vec<FramePreparedDraw>>,
    /// Reused dedup set for rebuilding [`Self::material_property_keys`].
    material_property_seen_scratch: HashSet<(i32, Option<i32>)>,
}

impl FramePreparedRenderables {
    /// Empty list (no active spaces / no valid renderers); used by tests and scenes where every
    /// mesh is non-resident.
    pub fn empty(render_context: RenderingContext) -> Self {
        Self {
            active_space_ids: Vec::new(),
            draws: Vec::new(),
            runs: Vec::new(),
            run_chunks: Vec::new(),
            material_property_keys: Vec::new(),
            material_property_key_signature: empty_material_key_signature(),
            render_context,
            #[cfg(test)]
            space_scratch: Vec::new(),
            material_property_seen_scratch: HashSet::new(),
        }
    }

    /// Builds the dense draw list for every active render space in `scene`.
    ///
    /// Per-space expansion runs in parallel via [`rayon`] and the per-space outputs are
    /// concatenated in render-space-id order. Every entry is filtered to only include draws that
    /// would survive [`super::collect::collect_chunk`]'s transform-scale, resident-mesh, and
    /// slot-validity checks -- per-view collection can iterate unconditionally without duplicating
    /// those guards.
    #[cfg(test)]
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
    #[cfg(test)]
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
        self.runs.clear();
        self.run_chunks.clear();
        self.material_property_keys.clear();

        {
            profiling::scope!("mesh::prepared_renderables::collect_active_spaces");
            self.active_space_ids.extend(
                scene
                    .render_space_ids()
                    .filter(|id| scene.space(*id).is_some_and(|s| s.is_active())),
            );
        }

        if self.active_space_ids.is_empty() {
            self.material_property_key_signature = empty_material_key_signature();
            return;
        }

        if self.active_space_ids.len() == 1 {
            let space_id = self.active_space_ids[0];
            {
                profiling::scope!("mesh::prepared_renderables::single_space_expand");
                self.draws.reserve(estimated_draw_count(scene, space_id));
                expand_space_into_aggressive(
                    &mut self.draws,
                    &mut self.space_scratch,
                    scene,
                    mesh_pool,
                    render_context,
                    space_id,
                );
            }
            self.refresh_runs_material_keys_and_chunks();
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
                    profiling::scope!("mesh::prepared_renderables::space_worker");
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
        self.refresh_runs_material_keys_and_chunks();
    }

    /// Refreshes renderer runs, run chunks, and material keys from the current draw list.
    fn refresh_runs_material_keys_and_chunks(&mut self) {
        self.material_property_key_signature = populate_runs_and_material_keys(
            &self.draws,
            &mut self.runs,
            &mut self.material_property_keys,
            &mut self.material_property_seen_scratch,
        );
        populate_run_chunks(
            &self.runs,
            &mut self.run_chunks,
            PREPARED_RUN_CHUNK_DRAW_TARGET,
        );
    }

    /// Dense prepared draw slice backing [`Self::runs`].
    #[inline]
    pub(super) fn draws(&self) -> &[FramePreparedDraw] {
        &self.draws
    }

    /// Cached run chunks consumed by per-view collection.
    #[inline]
    pub(super) fn run_chunks(&self) -> &[FramePreparedRunChunk] {
        &self.run_chunks
    }

    /// Resolves a cached run chunk into the backing run slice.
    #[inline]
    pub(super) fn runs_for_chunk(&self, chunk: FramePreparedRunChunk) -> &[FramePreparedRun] {
        &self.runs[chunk.start..chunk.end]
    }

    /// Number of expanded draws across all active render spaces.
    #[inline]
    pub fn len(&self) -> usize {
        self.draws.len()
    }

    /// `true` when no renderers expanded to any draw (no active space, no resident meshes).
    #[inline]
    #[cfg(test)]
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
    #[cfg(test)]
    pub fn mesh_material_pairs(&self) -> impl Iterator<Item = (i32, i32)> + '_ {
        self.draws
            .iter()
            .map(|d| (d.mesh_asset_id, d.material_asset_id))
    }

    /// Unique `(material_asset_id, property_block_id)` pairs referenced by this prepared snapshot.
    #[inline]
    pub fn unique_material_property_pairs(&self) -> &[(i32, Option<i32>)] {
        &self.material_property_keys
    }

    /// Signature of [`Self::unique_material_property_pairs`] used by frame caches to detect
    /// unchanged prepared material membership without touching every key.
    #[inline]
    pub fn material_property_key_signature(&self) -> u64 {
        self.material_property_key_signature
    }

    /// Rebuilds this snapshot in place from the supplied iterator of cached `(space, draws)`
    /// pairs. Used by [`super::render_world::RenderWorld`] when refreshing its persistent cache.
    ///
    /// Keeps the underlying `Vec` capacities so the steady-state rebuild path does not drop the
    /// backing buffers each frame.
    pub(super) fn rebuild_from_cached_spaces<'a, I>(
        &mut self,
        render_context: RenderingContext,
        active_with_draws: I,
    ) where
        I: IntoIterator<Item = (RenderSpaceId, &'a [FramePreparedDraw])>,
    {
        self.render_context = render_context;
        self.active_space_ids.clear();
        self.draws.clear();
        self.runs.clear();
        self.run_chunks.clear();
        for (id, draws) in active_with_draws {
            self.active_space_ids.push(id);
            self.draws.extend(draws.iter().cloned());
        }
        self.refresh_runs_material_keys_and_chunks();
    }
}

#[cfg(test)]
mod tests {
    use super::expand::populate_runs_and_material_keys;
    use super::*;
    use crate::gpu_pools::MeshPool;
    use crate::scene::{RenderSpaceId, SceneCoordinator, SkinnedMeshRenderer, StaticMeshRenderer};
    use crate::shared::{RenderTransform, ShadowCastMode};

    fn empty_scene() -> SceneCoordinator {
        SceneCoordinator::new()
    }

    fn prepared_draw(
        renderable_index: usize,
        material_asset_id: i32,
        property_block_id: Option<i32>,
    ) -> FramePreparedDraw {
        FramePreparedDraw {
            space_id: RenderSpaceId(1),
            renderable_index,
            instance_id: MeshRendererInstanceId(renderable_index as u64 + 1),
            node_id: renderable_index as i32,
            mesh_asset_id: 10,
            is_overlay: false,
            sorting_order: 0,
            skinned: false,
            world_space_deformed: false,
            blendshape_deformed: false,
            tangent_blendshape_deform_active: false,
            slot_index: 0,
            first_index: 0,
            index_count: 3,
            material_asset_id,
            property_block_id,
            cull_geometry: None,
        }
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

    #[test]
    fn populate_runs_also_deduplicates_material_property_keys() {
        let draws = vec![
            prepared_draw(0, 7, None),
            prepared_draw(0, 7, None),
            prepared_draw(1, 9, Some(3)),
            prepared_draw(1, 7, None),
        ];
        let mut runs = Vec::new();
        let mut keys = Vec::new();
        let mut seen = HashSet::new();

        let signature = populate_runs_and_material_keys(&draws, &mut runs, &mut keys, &mut seen);

        assert_eq!(
            runs,
            vec![
                FramePreparedRun { start: 0, end: 2 },
                FramePreparedRun { start: 2, end: 4 },
            ]
        );
        assert_eq!(keys, vec![(7, None), (9, Some(3))]);
        assert_ne!(signature, empty_material_key_signature());
    }

    #[test]
    fn populate_run_chunks_keeps_renderer_runs_intact() {
        let runs = vec![
            FramePreparedRun { start: 0, end: 2 },
            FramePreparedRun { start: 2, end: 5 },
            FramePreparedRun { start: 5, end: 9 },
            FramePreparedRun { start: 9, end: 10 },
        ];
        let mut chunks = Vec::new();

        populate_run_chunks(&runs, &mut chunks, 4);

        assert_eq!(
            chunks,
            vec![
                FramePreparedRunChunk { start: 0, end: 2 },
                FramePreparedRunChunk { start: 2, end: 3 },
                FramePreparedRunChunk { start: 3, end: 4 },
            ]
        );
    }

    #[test]
    fn estimated_draw_count_excludes_static_shadow_only_renderers() {
        let mut scene = empty_scene();
        let id = RenderSpaceId(1);
        scene.test_insert_static_mesh_renderers(
            id,
            vec![
                StaticMeshRenderer {
                    shadow_cast_mode: ShadowCastMode::On,
                    ..Default::default()
                },
                StaticMeshRenderer {
                    shadow_cast_mode: ShadowCastMode::ShadowOnly,
                    ..Default::default()
                },
            ],
        );

        assert_eq!(estimated_draw_count(&scene, id), 2);
    }

    #[test]
    fn estimated_draw_count_excludes_skinned_shadow_only_renderers() {
        let mut scene = empty_scene();
        let id = RenderSpaceId(1);
        let mut visible = SkinnedMeshRenderer::default();
        visible.base.shadow_cast_mode = ShadowCastMode::DoubleSided;
        let mut shadow_only = SkinnedMeshRenderer::default();
        shadow_only.base.shadow_cast_mode = ShadowCastMode::ShadowOnly;
        scene.test_insert_skinned_mesh_renderers(id, vec![visible, shadow_only]);

        assert_eq!(estimated_draw_count(&scene, id), 2);
    }
}
