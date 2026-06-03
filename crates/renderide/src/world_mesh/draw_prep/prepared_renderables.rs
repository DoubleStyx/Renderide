//! Frame-scope dense expansion of scene mesh renderables into one entry per
//! `(renderer, material slot)` pair.
//!
//! This is the Stage 3 amortization of [`super::collect::queue_draws_with_parallelism`]:
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
mod spatial;

use hashbrown::{HashMap, HashSet};
#[cfg(test)]
use rayon::prelude::*;
use std::ops::Range;

use crate::cpu_parallelism::RENDER_COMMAND_CHUNK_DRAWS;
#[cfg(test)]
use crate::gpu_pools::MeshPool;
use crate::particles::ParticleDrawParams;
use crate::scene::{MeshRendererInstanceId, RenderSpaceId, SceneCoordinator};
use crate::shared::RenderingContext;
use crate::world_mesh::culling::{MeshCullGeometry, WorldMeshCullInput};

use expand::{empty_material_key_signature, populate_runs_and_material_keys};
use spatial::{PreparedSpatialIndex, PreparedSpatialRunCandidates};

#[cfg(test)]
pub(in crate::world_mesh::draw_prep) use expand::estimated_draw_count;
#[cfg(test)]
pub(in crate::world_mesh::draw_prep) use expand::expand_space_into;
#[cfg(test)]
pub(in crate::world_mesh::draw_prep) use expand::expand_space_into_aggressive;
pub(in crate::world_mesh::draw_prep) use expand::{
    expand_render_buffer_renderers_into, expand_skinned_renderer_into, expand_static_renderer_into,
};

/// Target draw count for one prepared renderer-run chunk.
pub(super) const PREPARED_RUN_CHUNK_DRAW_TARGET: usize = RENDER_COMMAND_CHUNK_DRAWS;
/// Active render spaces assigned to one prepared-renderable expansion worker.
#[cfg(test)]
const PREPARED_EXPAND_PARALLEL_CHUNK_SPACES: usize = 1;
/// Active render-space count required before prepared-renderable expansion fans out.
#[cfg(test)]
const PREPARED_EXPAND_PARALLEL_MIN_SPACES: usize = PREPARED_EXPAND_PARALLEL_CHUNK_SPACES * 2;

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
    /// Dense per-space renderer ordinal assigned after prepared runs are finalized.
    pub renderer_ordinal: usize,
    /// Scene node id for rigid transform lookup and filter-mask indexing.
    pub node_id: i32,
    /// Resident mesh asset id (always matches `mesh_pool.get(...)` being `Some`).
    pub mesh_asset_id: i32,
    /// Precomputed overlay flag from the renderer's inherited layer state.
    pub is_overlay: bool,
    /// Precomputed hidden flag from the renderer's inherited layer state.
    pub is_hidden: bool,
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
    /// Optional final rigid world matrix for generated draw sources that are not represented by a
    /// scene transform alone.
    pub rigid_world_matrix_override: Option<glam::Mat4>,
    /// Particle renderer metadata for generated render-buffer draw sources.
    pub particle_draw: ParticleDrawParams,
}

/// Contiguous range of [`FramePreparedRenderables::draws`] produced by one source renderer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FramePreparedRun {
    /// First draw index in this renderer run.
    pub start: u32,
    /// One-past-last draw index in this renderer run.
    pub end: u32,
}

/// Stable renderer identity used to patch one prepared run without scanning all draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct FramePreparedRunLookupKey {
    /// Host render space that owns the renderer.
    space_id: RenderSpaceId,
    /// `true` when the renderer came from the skinned renderer table.
    skinned: bool,
    /// Dense renderer index in the source scene table.
    renderable_index: usize,
    /// Renderer-local identity that survives dense-table reindexing.
    instance_id: MeshRendererInstanceId,
}

/// Contiguous range of [`FramePreparedRenderables::runs`] consumed as one collection task.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FramePreparedRunChunk {
    /// First run index in this chunk.
    start: usize,
    /// One-past-last run index in this chunk.
    end: usize,
}

/// One renderer referenced by a prepared LOD entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FramePreparedLodRenderer {
    /// Stable per-space renderer ordinal used by dense visibility bitsets.
    pub(super) renderer_ordinal: usize,
}

/// One prepared LOD row with live renderer membership pre-resolved.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct FramePreparedLodEntry {
    /// Threshold copied from scene LOD state.
    pub(super) screen_relative_transition_height: f32,
    /// Live renderer ordinals selected by this LOD row.
    pub(super) renderers: Vec<FramePreparedLodRenderer>,
}

/// One prepared LOD group with membership and view-invariant bounds pre-resolved.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct FramePreparedLodGroup {
    /// Render space that owns the LOD group.
    pub(super) space_id: RenderSpaceId,
    /// Index into the scene render space's LOD group table.
    pub(super) scene_group_index: usize,
    /// Whether any referenced renderer is in the overlay layer.
    pub(super) any_overlay: bool,
    /// Cached group bounds when every referenced renderer has view-invariant geometry.
    pub(super) world_aabb: Option<(glam::Vec3, glam::Vec3)>,
    /// Ordered LOD entries with stale renderer references removed.
    pub(super) lods: Vec<FramePreparedLodEntry>,
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

/// Rebuilds direct renderer-run lookup entries from finalized run ranges.
fn populate_renderer_run_lookup(
    draws: &[FramePreparedDraw],
    runs: &[FramePreparedRun],
    lookup: &mut HashMap<FramePreparedRunLookupKey, FramePreparedRun>,
) {
    lookup.clear();
    lookup.reserve(runs.len());
    for &run in runs {
        let Some(first) = draws.get(run.start as usize) else {
            continue;
        };
        lookup.insert(
            FramePreparedRunLookupKey {
                space_id: first.space_id,
                skinned: first.skinned,
                renderable_index: first.renderable_index,
                instance_id: first.instance_id,
            },
            run,
        );
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
    /// Draw ranges per active render space in [`Self::draws`].
    cached_space_draw_ranges: HashMap<RenderSpaceId, Range<usize>>,
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
    /// Direct lookup from renderer identity to its prepared run.
    renderer_run_lookup: HashMap<FramePreparedRunLookupKey, FramePreparedRun>,
    /// First-seen unique `(material_asset_id, property_block_id)` keys referenced by
    /// [`Self::draws`]. Material caches consume this list once per shader permutation instead of
    /// materializing and deduping every prepared draw.
    material_property_keys: Vec<(i32, Option<i32>)>,
    /// Deterministic signature of [`Self::material_property_keys`] membership and order.
    material_property_key_signature: u64,
    /// Per-render-space BVH and linear fallback buckets over renderer runs.
    spatial: PreparedSpatialIndex,
    /// Prepared LOD groups resolved against the current draw snapshot.
    lod_groups: Vec<FramePreparedLodGroup>,
    /// Render context used when resolving material overrides; must match the per-view context.
    render_context: RenderingContext,
    /// Whether this snapshot was built for a context with no draw-prep overrides and can be used by any such context.
    context_invariant: bool,
    /// Previous rebuild's draw buffer, used for range-based partial snapshot reuse.
    previous_draws: Vec<FramePreparedDraw>,
    /// Previous rebuild's per-space draw ranges, paired with [`Self::previous_draws`].
    previous_cached_space_draw_ranges: HashMap<RenderSpaceId, Range<usize>>,
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
        Self::empty_with_context_mode(render_context, false)
    }

    /// Empty list that may be reused for any render context without draw-prep overrides.
    pub(super) fn empty_context_invariant(render_context: RenderingContext) -> Self {
        Self::empty_with_context_mode(render_context, true)
    }

    /// Empty list with an explicit context-compatibility mode.
    fn empty_with_context_mode(render_context: RenderingContext, context_invariant: bool) -> Self {
        Self {
            active_space_ids: Vec::new(),
            cached_space_draw_ranges: HashMap::new(),
            draws: Vec::new(),
            runs: Vec::new(),
            run_chunks: Vec::new(),
            renderer_run_lookup: HashMap::new(),
            material_property_keys: Vec::new(),
            material_property_key_signature: empty_material_key_signature(),
            spatial: PreparedSpatialIndex::default(),
            lod_groups: Vec::new(),
            render_context,
            context_invariant,
            previous_draws: Vec::new(),
            previous_cached_space_draw_ranges: HashMap::new(),
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
        self.cached_space_draw_ranges.clear();
        self.draws.clear();
        self.runs.clear();
        self.run_chunks.clear();
        self.renderer_run_lookup.clear();
        self.material_property_keys.clear();
        self.lod_groups.clear();

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

        if self.active_space_ids.len() < PREPARED_EXPAND_PARALLEL_MIN_SPACES {
            profiling::scope!("mesh::prepared_renderables::serial_space_expand");
            for &space_id in &self.active_space_ids {
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
            self.refresh_runs_material_keys_and_chunks(Some(scene));
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
                .with_min_len(PREPARED_EXPAND_PARALLEL_CHUNK_SPACES)
                .zip(
                    active_space_ids
                        .par_iter()
                        .with_min_len(PREPARED_EXPAND_PARALLEL_CHUNK_SPACES),
                )
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
        self.refresh_runs_material_keys_and_chunks(Some(scene));
    }

    /// Refreshes renderer runs, run chunks, material keys, and prepared LOD groups from the current draw list.
    fn refresh_runs_material_keys_and_chunks(&mut self, scene: Option<&SceneCoordinator>) {
        self.refresh_cached_space_draw_ranges();
        self.material_property_key_signature = populate_runs_and_material_keys(
            &self.draws,
            &mut self.runs,
            &mut self.material_property_keys,
            &mut self.material_property_seen_scratch,
        );
        if let Some(scene) = scene {
            populate_renderer_ordinals_from_scene(&mut self.draws, scene);
        } else {
            populate_renderer_ordinals_from_runs(&mut self.draws, &self.runs);
        }
        populate_run_chunks(
            &self.runs,
            &mut self.run_chunks,
            PREPARED_RUN_CHUNK_DRAW_TARGET,
        );
        populate_renderer_run_lookup(&self.draws, &self.runs, &mut self.renderer_run_lookup);
        self.rebuild_lod_groups(scene);
        self.spatial.rebuild(&self.draws, &self.runs);
    }

    /// Rebuilds cached per-space draw ranges from the current active-space ordering.
    fn refresh_cached_space_draw_ranges(&mut self) {
        self.cached_space_draw_ranges.clear();
        let mut cursor = 0usize;
        for &space_id in &self.active_space_ids {
            let start = cursor;
            while self
                .draws
                .get(cursor)
                .is_some_and(|draw| draw.space_id == space_id)
            {
                cursor += 1;
            }
            self.cached_space_draw_ranges
                .insert(space_id, start..cursor);
        }
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

    /// Returns run candidates for the requested render spaces after spatial frustum filtering.
    #[inline]
    pub(super) fn spatial_run_candidates(
        &self,
        space_ids: &[RenderSpaceId],
        scene: &SceneCoordinator,
        culling: Option<&WorldMeshCullInput<'_>>,
    ) -> PreparedSpatialRunCandidates {
        self.spatial
            .query_runs(&self.runs, space_ids, scene, culling)
    }

    /// Prepared LOD groups for per-view selection.
    #[inline]
    pub(super) fn lod_groups(&self) -> &[FramePreparedLodGroup] {
        &self.lod_groups
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

    /// Returns whether this snapshot can be consumed by `render_context`.
    #[inline]
    pub fn is_compatible_with_render_context(&self, render_context: RenderingContext) -> bool {
        self.context_invariant || self.render_context == render_context
    }

    /// Active render spaces captured by this prepared snapshot.
    #[inline]
    pub fn active_space_ids(&self) -> &[RenderSpaceId] {
        &self.active_space_ids
    }

    /// Returns whether the previous rebuild has retained draw rows for `id`.
    #[inline]
    pub(super) fn has_previous_cached_draws_for_space(&self, id: RenderSpaceId) -> bool {
        self.previous_cached_space_draw_ranges.contains_key(&id)
    }

    /// Returns whether `space_id` uses a BVH instead of only linear buckets.
    #[inline]
    #[cfg(test)]
    pub fn space_uses_bvh_for_tests(&self, space_id: RenderSpaceId) -> bool {
        self.spatial.space_uses_bvh_for_tests(space_id)
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

    /// Starts a retained render-world snapshot rebuild, preserving backing buffer capacity.
    pub(super) fn begin_cached_rebuild(&mut self, render_context: RenderingContext) {
        self.render_context = render_context;
        self.previous_draws.clear();
        std::mem::swap(&mut self.draws, &mut self.previous_draws);
        self.previous_cached_space_draw_ranges.clear();
        std::mem::swap(
            &mut self.cached_space_draw_ranges,
            &mut self.previous_cached_space_draw_ranges,
        );
        self.active_space_ids.clear();
        self.cached_space_draw_ranges.clear();
        self.draws.clear();
        self.runs.clear();
        self.run_chunks.clear();
        self.renderer_run_lookup.clear();
        self.lod_groups.clear();
    }

    /// Appends an active render space id to the retained snapshot under construction.
    pub(super) fn push_cached_space(&mut self, id: RenderSpaceId) {
        self.active_space_ids.push(id);
    }

    /// Appends retained draw-template rows to the snapshot under construction.
    pub(super) fn extend_cached_draws(&mut self, draws: &[FramePreparedDraw]) {
        self.draws.extend(draws.iter().cloned());
    }

    /// Appends retained draw rows for `id` from the previous rebuild, if available.
    pub(super) fn extend_previous_cached_draws_for_space(&mut self, id: RenderSpaceId) -> bool {
        let Some(range) = self.previous_cached_space_draw_ranges.get(&id).cloned() else {
            return false;
        };
        let Some(draws) = self.previous_draws.get(range) else {
            return false;
        };
        self.draws.extend(draws.iter().cloned());
        true
    }

    /// Appends retained draw-template rows with dynamic cull geometry filled from renderer state.
    pub(super) fn extend_cached_draws_with_cull_geometry(
        &mut self,
        draws: &[FramePreparedDraw],
        cull_geometry: Option<MeshCullGeometry>,
    ) {
        self.draws.extend(draws.iter().cloned().map(|mut draw| {
            draw.cull_geometry = cull_geometry;
            draw
        }));
    }

    /// Mutable draw buffer used while a retained snapshot rebuild is in progress.
    pub(super) fn draws_mut_for_cached_rebuild(&mut self) -> &mut Vec<FramePreparedDraw> {
        &mut self.draws
    }

    /// Updates dynamic cull geometry for an already prepared renderer run.
    pub(super) fn update_cached_renderer_cull_geometry(
        &mut self,
        space_id: RenderSpaceId,
        skinned: bool,
        renderable_index: usize,
        instance_id: MeshRendererInstanceId,
        cull_geometry: Option<MeshCullGeometry>,
    ) {
        let key = FramePreparedRunLookupKey {
            space_id,
            skinned,
            renderable_index,
            instance_id,
        };
        let Some(run) = self.renderer_run_lookup.get(&key).copied() else {
            return;
        };
        let start = run.start as usize;
        let end = run.end as usize;
        if let Some(draws) = self.draws.get_mut(start..end) {
            for draw in draws {
                draw.cull_geometry = cull_geometry;
            }
        }
    }

    /// Refits cached spatial data for spaces whose dynamic bounds changed.
    pub(super) fn refit_cached_spatial_for_spaces<I>(&mut self, space_ids: I) -> usize
    where
        I: IntoIterator<Item = RenderSpaceId>,
    {
        self.spatial
            .refit_spaces(&self.draws, &self.runs, space_ids)
    }

    /// Finalizes a retained snapshot rebuild by refreshing runs, chunks, and material keys.
    pub(super) fn finish_cached_rebuild(&mut self, scene: &SceneCoordinator) {
        self.refresh_runs_material_keys_and_chunks(Some(scene));
    }

    /// Rebuilds the cached-space snapshot directly from supplied draw slices for tests.
    #[cfg(test)]
    fn rebuild_from_cached_spaces<'a, I>(&mut self, render_context: RenderingContext, spaces: I)
    where
        I: IntoIterator<Item = (RenderSpaceId, &'a [FramePreparedDraw])>,
    {
        self.begin_cached_rebuild(render_context);
        for (space_id, draws) in spaces {
            self.push_cached_space(space_id);
            self.extend_cached_draws(draws);
        }
        self.refresh_runs_material_keys_and_chunks(None);
    }
}

impl FramePreparedRenderables {
    /// Rebuilds pre-resolved LOD groups from the active scene spaces and current prepared draws.
    fn rebuild_lod_groups(&mut self, scene: Option<&SceneCoordinator>) {
        self.lod_groups.clear();
        let Some(scene) = scene else {
            return;
        };
        profiling::scope!("mesh::prepared_renderables::rebuild_lod_groups");
        let renderer_lookup = build_lod_renderer_lookup(&self.draws, &self.runs);
        for &space_id in &self.active_space_ids {
            let Some(space) = scene.space(space_id) else {
                continue;
            };
            for (scene_group_index, group) in space.lod_groups().iter().enumerate() {
                let mut view_dependent_bounds = false;
                let mut prepared_group = FramePreparedLodGroup {
                    space_id,
                    scene_group_index,
                    any_overlay: false,
                    world_aabb: None,
                    lods: Vec::new(),
                };
                for lod in &group.lods {
                    let mut prepared_lod = FramePreparedLodEntry {
                        screen_relative_transition_height: lod.screen_relative_transition_height,
                        renderers: Vec::with_capacity(lod.renderers.len()),
                    };
                    for renderer_ref in &lod.renderers {
                        let key = (space_id, renderer_ref.instance_id);
                        let Some(renderer) = renderer_lookup.get(&key).copied() else {
                            continue;
                        };
                        prepared_group.any_overlay |= renderer.is_overlay;
                        if let Some(bounds) = renderer.world_aabb {
                            if !view_dependent_bounds {
                                union_prepared_lod_aabb(&mut prepared_group.world_aabb, bounds);
                            }
                        } else {
                            view_dependent_bounds = true;
                            prepared_group.world_aabb = None;
                        }
                        prepared_lod.renderers.push(FramePreparedLodRenderer {
                            renderer_ordinal: renderer.renderer_ordinal,
                        });
                    }
                    prepared_group.lods.push(prepared_lod);
                }
                if prepared_group
                    .lods
                    .iter()
                    .any(|lod| !lod.renderers.is_empty())
                {
                    self.lod_groups.push(prepared_group);
                }
            }
        }
    }
}

/// Renderer metadata used while rebuilding prepared LOD groups.
#[derive(Clone, Copy)]
struct PreparedLodRendererLookup {
    /// Stable renderer ordinal.
    renderer_ordinal: usize,
    /// Whether the renderer is in the overlay layer.
    is_overlay: bool,
    /// View-invariant renderer AABB when available.
    world_aabb: Option<(glam::Vec3, glam::Vec3)>,
}

/// Builds a lookup from stable renderer identity to prepared LOD metadata.
fn build_lod_renderer_lookup(
    draws: &[FramePreparedDraw],
    runs: &[FramePreparedRun],
) -> HashMap<(RenderSpaceId, MeshRendererInstanceId), PreparedLodRendererLookup> {
    let mut lookup = HashMap::with_capacity(runs.len());
    for run in runs {
        let Some(first) = draws.get(run.start as usize) else {
            continue;
        };
        lookup.insert(
            (first.space_id, first.instance_id),
            PreparedLodRendererLookup {
                renderer_ordinal: first.renderer_ordinal,
                is_overlay: first.is_overlay,
                world_aabb: first.cull_geometry.and_then(|geometry| geometry.world_aabb),
            },
        );
    }
    lookup
}

/// Expands `dst` to include a prepared renderer AABB.
fn union_prepared_lod_aabb(
    dst: &mut Option<(glam::Vec3, glam::Vec3)>,
    bounds: (glam::Vec3, glam::Vec3),
) {
    match dst {
        Some((min, max)) => {
            *min = min.min(bounds.0);
            *max = max.max(bounds.1);
        }
        None => *dst = Some(bounds),
    }
}

/// Assigns stable scene-table renderer ordinals to every prepared draw row.
fn populate_renderer_ordinals_from_scene(
    draws: &mut [FramePreparedDraw],
    scene: &SceneCoordinator,
) {
    for draw in draws {
        let static_count = scene
            .space(draw.space_id)
            .map_or(0, |space| space.static_mesh_renderers().len());
        draw.renderer_ordinal = if draw.skinned {
            static_count.saturating_add(draw.renderable_index)
        } else {
            draw.renderable_index
        };
    }
}

/// Assigns dense renderer ordinals per render space when no scene table is available.
fn populate_renderer_ordinals_from_runs(
    draws: &mut [FramePreparedDraw],
    runs: &[FramePreparedRun],
) {
    let mut next_by_space: HashMap<RenderSpaceId, usize> = HashMap::new();
    for run in runs {
        let start = run.start as usize;
        let end = run.end as usize;
        let Some(first) = draws.get(start) else {
            continue;
        };
        let ordinal = *next_by_space
            .entry(first.space_id)
            .and_modify(|next| *next += 1)
            .or_insert(0);
        for draw in &mut draws[start..end] {
            draw.renderer_ordinal = ordinal;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::expand::populate_runs_and_material_keys;
    use super::*;
    use crate::camera::HostCameraFrame;
    use crate::gpu_pools::MeshPool;
    use crate::scene::{RenderSpaceId, SceneCoordinator, SkinnedMeshRenderer, StaticMeshRenderer};
    use crate::shared::{RenderTransform, ShadowCastMode};
    use crate::world_mesh::culling::{
        MeshCullGeometry, WorldMeshCullInput, WorldMeshCullProjParams,
    };
    use glam::{Mat4, Vec3};

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
            renderer_ordinal: 0,
            node_id: renderable_index as i32,
            mesh_asset_id: 10,
            is_overlay: false,
            is_hidden: false,
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
            rigid_world_matrix_override: None,
            particle_draw: ParticleDrawParams::default(),
        }
    }

    fn prepared_draw_with_bounds(
        renderable_index: usize,
        min: Vec3,
        max: Vec3,
    ) -> FramePreparedDraw {
        let mut draw = prepared_draw(renderable_index, 1, None);
        draw.cull_geometry = Some(MeshCullGeometry {
            world_aabb: Some((min, max)),
            rigid_world_matrix: Some(Mat4::IDENTITY),
            front_face_world_matrix: Some(Mat4::IDENTITY),
        });
        draw
    }

    fn prepared_overlay_draw_with_bounds(
        renderable_index: usize,
        min: Vec3,
        max: Vec3,
    ) -> FramePreparedDraw {
        let mut draw = prepared_draw_with_bounds(renderable_index, min, max);
        draw.is_overlay = true;
        draw
    }

    fn spatial_scene_and_cull(
        space_id: RenderSpaceId,
    ) -> (SceneCoordinator, HostCameraFrame, WorldMeshCullProjParams) {
        let mut scene = SceneCoordinator::new();
        scene.test_seed_space_identity_worlds(space_id, vec![RenderTransform::default()], vec![-1]);
        (
            scene,
            HostCameraFrame::default(),
            WorldMeshCullProjParams {
                world_proj: Mat4::IDENTITY,
                overlay_proj: Mat4::IDENTITY,
                vr_stereo: None,
            },
        )
    }

    fn prepared_from_space_draws(
        space_id: RenderSpaceId,
        draws: &[FramePreparedDraw],
    ) -> FramePreparedRenderables {
        let adjusted = draws
            .iter()
            .cloned()
            .map(|mut draw| {
                draw.space_id = space_id;
                draw
            })
            .collect::<Vec<_>>();
        let mut prepared = FramePreparedRenderables::empty(RenderingContext::UserView);
        prepared.rebuild_from_cached_spaces(
            RenderingContext::UserView,
            [(space_id, adjusted.as_slice())],
        );
        prepared
    }

    #[test]
    fn cached_rebuild_can_reuse_previous_space_ranges() {
        let mut prepared = FramePreparedRenderables::empty(RenderingContext::UserView);
        let draws = [prepared_draw(0, 10, None), prepared_draw(1, 11, None)];
        prepared.rebuild_from_cached_spaces(
            RenderingContext::UserView,
            [(RenderSpaceId(1), draws.as_slice())],
        );

        prepared.begin_cached_rebuild(RenderingContext::Camera);
        assert!(prepared.has_previous_cached_draws_for_space(RenderSpaceId(1)));
        prepared.push_cached_space(RenderSpaceId(1));
        assert!(prepared.extend_previous_cached_draws_for_space(RenderSpaceId(1)));
        prepared.finish_cached_rebuild(&empty_scene());

        assert_eq!(prepared.draws.len(), 2);
        assert_eq!(prepared.draws[0].material_asset_id, 10);
        assert_eq!(prepared.draws[1].material_asset_id, 11);
        assert!(
            prepared
                .cached_space_draw_ranges
                .contains_key(&RenderSpaceId(1))
        );
    }

    #[test]
    fn cull_geometry_update_uses_renderer_run_lookup() {
        let mut prepared = FramePreparedRenderables::empty(RenderingContext::UserView);
        let instance = MeshRendererInstanceId(42);
        let mut first_slot = prepared_draw(0, 10, None);
        first_slot.instance_id = instance;
        first_slot.slot_index = 0;
        let mut second_slot = first_slot.clone();
        second_slot.slot_index = 1;
        second_slot.material_asset_id = 11;
        let other = prepared_draw(1, 12, None);
        let draws = vec![first_slot, second_slot, other];
        prepared.rebuild_from_cached_spaces(
            RenderingContext::UserView,
            [(RenderSpaceId(1), draws.as_slice())],
        );

        let bounds = (Vec3::splat(-1.0), Vec3::splat(1.0));
        let geometry = MeshCullGeometry {
            world_aabb: Some(bounds),
            rigid_world_matrix: Some(Mat4::IDENTITY),
            front_face_world_matrix: Some(Mat4::IDENTITY),
        };
        prepared.update_cached_renderer_cull_geometry(
            RenderSpaceId(1),
            false,
            0,
            instance,
            Some(geometry),
        );

        assert_eq!(
            prepared.draws[0].cull_geometry.and_then(|g| g.world_aabb),
            Some(bounds)
        );
        assert_eq!(
            prepared.draws[1].cull_geometry.and_then(|g| g.world_aabb),
            Some(bounds)
        );
        assert!(prepared.draws[2].cull_geometry.is_none());
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
    fn renderer_ordinals_follow_static_scene_table_even_when_rows_emit_no_draws() {
        let space_id = RenderSpaceId(9);
        let mut scene = empty_scene();
        scene.test_insert_static_mesh_renderers(
            space_id,
            vec![
                StaticMeshRenderer::default(),
                StaticMeshRenderer::default(),
                StaticMeshRenderer::default(),
            ],
        );
        let mut static_draw = prepared_draw(1, 7, None);
        static_draw.space_id = space_id;
        static_draw.renderable_index = 1;
        static_draw.skinned = false;
        let mut draws = vec![static_draw];

        populate_renderer_ordinals_from_scene(&mut draws, &scene);

        assert_eq!(draws[0].renderer_ordinal, 1);
    }

    #[test]
    fn spatial_query_uses_bvh_for_large_spaces_and_filters_frustum() {
        let space_id = RenderSpaceId(1);
        let (scene, host_camera, proj) = spatial_scene_and_cull(space_id);
        let culling = WorldMeshCullInput {
            proj,
            host_camera: &host_camera,
            hi_z: None,
            hi_z_temporal: None,
        };
        let mut draws = Vec::new();
        for idx in 0..80 {
            let (min, max) = if idx < 40 {
                (Vec3::new(-0.5, -0.5, -0.5), Vec3::new(0.5, 0.5, 0.5))
            } else {
                (Vec3::new(2.0, -0.5, -0.5), Vec3::new(3.0, 0.5, 0.5))
            };
            draws.push(prepared_draw_with_bounds(idx, min, max));
        }
        let prepared = prepared_from_space_draws(space_id, &draws);

        let candidates = prepared.spatial_run_candidates(&[space_id], &scene, Some(&culling));

        assert!(prepared.space_uses_bvh_for_tests(space_id));
        assert_eq!(candidates.runs.len(), 40);
        assert_eq!(candidates.cull_stats, (40, 40, 0));
        assert_eq!(candidates.visibility.indexed_runs, 80);
        assert_eq!(candidates.visibility.fallback_runs, 0);
        assert_eq!(candidates.visibility.candidate_runs, 40);
        assert_eq!(candidates.visibility.broadphase_culled_runs, 40);
        assert_eq!(candidates.visibility.broadphase_culled_draws, 40);
    }

    #[test]
    fn spatial_query_keeps_small_spaces_on_linear_path() {
        let space_id = RenderSpaceId(2);
        let (scene, host_camera, proj) = spatial_scene_and_cull(space_id);
        let culling = WorldMeshCullInput {
            proj,
            host_camera: &host_camera,
            hi_z: None,
            hi_z_temporal: None,
        };
        let draws = (0..8)
            .map(|idx| {
                prepared_draw_with_bounds(
                    idx,
                    Vec3::new(-0.25, -0.25, -0.25),
                    Vec3::new(0.25, 0.25, 0.25),
                )
            })
            .collect::<Vec<_>>();
        let prepared = prepared_from_space_draws(space_id, &draws);

        let candidates = prepared.spatial_run_candidates(&[space_id], &scene, Some(&culling));

        assert!(!prepared.space_uses_bvh_for_tests(space_id));
        assert_eq!(candidates.runs.len(), 8);
        assert_eq!(candidates.cull_stats, (0, 0, 0));
        assert_eq!(candidates.visibility.indexed_runs, 8);
        assert_eq!(candidates.visibility.linear_fallback_runs, 8);
        assert_eq!(candidates.visibility.candidate_runs, 8);
    }

    #[test]
    fn spatial_query_counts_rejected_material_slots() {
        let space_id = RenderSpaceId(3);
        let (scene, host_camera, proj) = spatial_scene_and_cull(space_id);
        let culling = WorldMeshCullInput {
            proj,
            host_camera: &host_camera,
            hi_z: None,
            hi_z_temporal: None,
        };
        let outside_slot0 =
            prepared_draw_with_bounds(0, Vec3::new(2.0, -0.5, -0.5), Vec3::new(3.0, 0.5, 0.5));
        let mut outside_slot1 = outside_slot0.clone();
        outside_slot1.slot_index = 1;
        outside_slot1.material_asset_id = 2;
        let inside = prepared_draw_with_bounds(
            1,
            Vec3::new(-0.25, -0.25, -0.25),
            Vec3::new(0.25, 0.25, 0.25),
        );
        let prepared = prepared_from_space_draws(space_id, &[outside_slot0, outside_slot1, inside]);

        let candidates = prepared.spatial_run_candidates(&[space_id], &scene, Some(&culling));

        assert_eq!(candidates.runs.len(), 1);
        assert_eq!(candidates.cull_stats, (2, 2, 0));
        assert_eq!(candidates.visibility.indexed_runs, 2);
        assert_eq!(candidates.visibility.candidate_runs, 1);
        assert_eq!(candidates.visibility.broadphase_culled_runs, 1);
        assert_eq!(candidates.visibility.broadphase_culled_draws, 2);
    }

    #[test]
    fn spatial_query_keeps_overlay_runs_conservative() {
        let space_id = RenderSpaceId(4);
        let (scene, host_camera, proj) = spatial_scene_and_cull(space_id);
        let culling = WorldMeshCullInput {
            proj,
            host_camera: &host_camera,
            hi_z: None,
            hi_z_temporal: None,
        };
        let draws = (0..80)
            .map(|idx| {
                prepared_overlay_draw_with_bounds(
                    idx,
                    Vec3::new(2.0, -0.5, -0.5),
                    Vec3::new(3.0, 0.5, 0.5),
                )
            })
            .collect::<Vec<_>>();
        let prepared = prepared_from_space_draws(space_id, &draws);

        let candidates = prepared.spatial_run_candidates(&[space_id], &scene, Some(&culling));

        assert!(!prepared.space_uses_bvh_for_tests(space_id));
        assert_eq!(candidates.runs.len(), 80);
        assert_eq!(candidates.cull_stats, (0, 0, 0));
        assert_eq!(candidates.visibility.indexed_runs, 0);
        assert_eq!(candidates.visibility.fallback_runs, 80);
        assert_eq!(candidates.visibility.linear_fallback_runs, 80);
        assert_eq!(candidates.visibility.candidate_runs, 80);
    }

    #[test]
    fn spatial_query_preserves_run_order_across_multiple_spaces() {
        let first_space = RenderSpaceId(5);
        let second_space = RenderSpaceId(6);
        let mut scene = SceneCoordinator::new();
        scene.test_seed_space_identity_worlds(
            first_space,
            vec![RenderTransform::default()],
            vec![-1],
        );
        scene.test_seed_space_identity_worlds(
            second_space,
            vec![RenderTransform::default()],
            vec![-1],
        );
        let host_camera = HostCameraFrame::default();
        let culling = WorldMeshCullInput {
            proj: WorldMeshCullProjParams {
                world_proj: Mat4::IDENTITY,
                overlay_proj: Mat4::IDENTITY,
                vr_stereo: None,
            },
            host_camera: &host_camera,
            hi_z: None,
            hi_z_temporal: None,
        };
        let mut prepared = FramePreparedRenderables::empty(RenderingContext::UserView);
        let mut first_draw = prepared_draw_with_bounds(
            0,
            Vec3::new(-0.25, -0.25, -0.25),
            Vec3::new(0.25, 0.25, 0.25),
        );
        first_draw.space_id = first_space;
        let first = [first_draw];
        let mut second_draw = prepared_draw_with_bounds(
            1,
            Vec3::new(-0.25, -0.25, -0.25),
            Vec3::new(0.25, 0.25, 0.25),
        );
        second_draw.space_id = second_space;
        let second = [second_draw];
        prepared.rebuild_from_cached_spaces(
            RenderingContext::UserView,
            [
                (first_space, first.as_slice()),
                (second_space, second.as_slice()),
            ],
        );

        let candidates =
            prepared.spatial_run_candidates(&[second_space, first_space], &scene, Some(&culling));

        assert_eq!(
            candidates.runs,
            vec![
                FramePreparedRun { start: 0, end: 1 },
                FramePreparedRun { start: 1, end: 2 },
            ]
        );
        assert_eq!(candidates.visibility.candidate_runs, 2);
    }

    #[test]
    fn spatial_query_dedups_duplicate_space_queries_in_prepared_order() {
        let space_id = RenderSpaceId(7);
        let (scene, host_camera, proj) = spatial_scene_and_cull(space_id);
        let culling = WorldMeshCullInput {
            proj,
            host_camera: &host_camera,
            hi_z: None,
            hi_z_temporal: None,
        };
        let first = prepared_draw_with_bounds(
            0,
            Vec3::new(-0.25, -0.25, -0.25),
            Vec3::new(0.25, 0.25, 0.25),
        );
        let second = prepared_draw_with_bounds(
            1,
            Vec3::new(-0.25, -0.25, -0.25),
            Vec3::new(0.25, 0.25, 0.25),
        );
        let prepared = prepared_from_space_draws(space_id, &[first, second]);

        let candidates =
            prepared.spatial_run_candidates(&[space_id, space_id], &scene, Some(&culling));

        assert_eq!(
            candidates.runs,
            vec![
                FramePreparedRun { start: 0, end: 1 },
                FramePreparedRun { start: 1, end: 2 },
            ]
        );
        assert_eq!(candidates.visibility.raw_candidate_marks, 4);
        assert_eq!(candidates.visibility.candidate_runs, 2);
        assert_eq!(candidates.visibility.duplicate_candidate_marks, 2);
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
