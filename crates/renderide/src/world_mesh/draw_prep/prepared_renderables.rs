//! Frame-scope dense expansion of scene mesh renderables into one entry per
//! `(renderer, material slot)` pair.
//!
//! Resident meshes, material slots, submesh ranges, and render-context overrides are resolved
//! once per frame and shared across views.
//!
//! Culling and [`super::item::WorldMeshDrawItem`] construction remain per-view because they
//! depend on the view's camera, filter, and Hi-Z snapshot.

mod expand;
mod lod;
mod spatial;

use hashbrown::{HashMap, HashSet};
#[cfg(test)]
use rayon::prelude::*;
use std::ops::Range;

use crate::cpu_parallelism::RENDER_COMMAND_CHUNK_DRAWS;
use crate::gpu_pools::MeshPool;
use crate::particles::ParticleDrawParams;
use crate::render_contract::ParticleDrawKind;
use crate::scene::{
    MeshRendererInstanceId, RenderSpaceId, RenderWorldParticleRendererDirty,
    RenderWorldParticleRendererKind, RenderWorldRendererDirty, RenderWorldRendererKind,
    SceneCoordinator, SceneMeshRendererRead, WorldMeshSceneRead,
};
use crate::shared::{RenderingContext, ShadowCastMode};
use crate::world_mesh::culling::{MeshCullGeometry, WorldMeshCullInput};

use expand::{empty_material_key_signature, populate_runs_and_material_keys};
pub(super) use lod::{FramePreparedLodEntry, FramePreparedLodGroup};
use spatial::{PreparedSpatialIndex, PreparedSpatialRunCandidates};

#[cfg(test)]
pub(in crate::world_mesh::draw_prep) use expand::estimated_draw_count;
#[cfg(test)]
pub(in crate::world_mesh::draw_prep) use expand::expand_space_into;
#[cfg(test)]
pub(in crate::world_mesh::draw_prep) use expand::expand_space_into_aggressive;
pub(in crate::world_mesh::draw_prep) use expand::{
    expand_render_buffer_renderer_into, expand_render_buffer_renderers_into,
    expand_skinned_renderer_into, expand_static_renderer_into,
};

/// Target draw count for one prepared renderer-run chunk.
///
/// Collection is branch-light compared with command recording, so using the 64-draw command
/// packet directly creates many sub-20-us worker calls in draw-heavy scenes. Keep renderer runs
/// intact while amortizing collection setup over a coarser packet.
pub(super) const PREPARED_RUN_CHUNK_DRAW_TARGET: usize = RENDER_COMMAND_CHUNK_DRAWS * 4;
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
#[derive(Clone, Debug, PartialEq)]
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
    /// Host shadow-caster mode for this renderer.
    pub shadow_cast_mode: ShadowCastMode,
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
    /// Material-stack ordering marker when this slot reuses the final submesh.
    pub material_stack_order: Option<super::item::MaterialStackOrder>,
    /// First index in the mesh index buffer for the selected submesh range.
    pub first_index: u32,
    /// Number of indices for this submesh draw (always `> 0`).
    pub index_count: u32,
    /// Material id after [`SceneCoordinator::overridden_material_asset_id`] resolution.
    ///
    /// `-1` is retained as the host missing-material sentinel and routes to the Null fallback.
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

/// Stable source identity for all prepared rows emitted by one PhotonDust renderer.
///
/// Mesh-particle renderers produce one ordinary prepared run per point instance, so the
/// [`FramePreparedRunLookupKey`] cannot represent the complete renderer by itself. This coarser
/// key retains the same stable space/table/index identity while grouping that contiguous run set
/// into one patchable range.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct FramePreparedParticleRendererLookupKey {
    /// Host render space that owns the renderer.
    space_id: RenderSpaceId,
    /// PhotonDust renderer family.
    kind: RenderWorldParticleRendererKind,
    /// Dense renderer index in its source table.
    renderable_index: usize,
}

impl From<RenderWorldParticleRendererDirty> for FramePreparedParticleRendererLookupKey {
    fn from(dirty: RenderWorldParticleRendererDirty) -> Self {
        Self {
            space_id: dirty.space_id,
            kind: dirty.kind,
            renderable_index: dirty.renderable_index,
        }
    }
}

/// Result of patching generated particle rows in an existing prepared snapshot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct PreparedParticlePatchStats {
    /// Particle renderer sources whose previous or replacement range was non-empty.
    pub(super) renderer_count: usize,
    /// Prepared draw rows copied into the snapshot.
    pub(super) draw_count: usize,
    /// Whether any prepared rows changed.
    pub(super) changed: bool,
    /// Whether variable-length/run/material changes required metadata reconstruction.
    pub(super) structural_rebuild: bool,
    /// Spatial spaces refit on the stable-run fast path.
    pub(super) spatial_refit_count: usize,
}

/// Result of patching exact static/skinned renderer ranges in an existing prepared snapshot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct PreparedMeshPatchStats {
    /// Unique renderer rows requested by the caller.
    pub(super) candidate_count: usize,
    /// Non-empty old or replacement ranges that actually changed.
    pub(super) range_count: usize,
    /// Fresh prepared draw rows copied by changed replacements.
    pub(super) draw_count: usize,
    /// Candidate ranges proven semantically identical after expansion.
    pub(super) noop_count: usize,
    /// Whether any prepared row changed.
    pub(super) changed: bool,
    /// Whether draw/run/material shape changes required metadata reconstruction.
    pub(super) structural_rebuild: bool,
    /// Spatial spaces refit on the stable-range fast path.
    pub(super) spatial_refit_count: usize,
}

/// Summary of lightweight context-overlay synchronization and exact renderer patching.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct PreparedContextPatchStats {
    /// Static/skinned renderer ranges expanded for context-local values.
    pub(super) mesh_renderer_count: usize,
    /// Particle renderer ranges expanded for context-local values.
    pub(super) particle_renderer_count: usize,
    /// Fresh prepared rows copied by the context patch.
    pub(super) draw_count: usize,
    /// Whether any prepared row changed.
    pub(super) changed: bool,
    /// Number of metadata reconstructions caused by shape/material changes.
    pub(super) structural_rebuild_count: usize,
    /// Spatial spaces refit without metadata reconstruction.
    pub(super) spatial_refit_count: usize,
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

/// Rebuilds the direct particle-renderer range lookup from prepared draw order.
fn populate_particle_renderer_draw_lookup(
    draws: &[FramePreparedDraw],
    lookup: &mut HashMap<FramePreparedParticleRendererLookupKey, Range<usize>>,
) {
    lookup.clear();
    for (draw_index, draw) in draws.iter().enumerate() {
        let Some(kind) = particle_renderer_kind(draw.particle_draw.kind) else {
            continue;
        };
        let key = FramePreparedParticleRendererLookupKey {
            space_id: draw.space_id,
            kind,
            renderable_index: draw.renderable_index,
        };
        match lookup.entry(key) {
            hashbrown::hash_map::Entry::Occupied(mut entry) => {
                let range = entry.get_mut();
                debug_assert_eq!(
                    range.end, draw_index,
                    "particle renderer rows must remain contiguous"
                );
                range.end = draw_index + 1;
            }
            hashbrown::hash_map::Entry::Vacant(entry) => {
                entry.insert(draw_index..draw_index + 1);
            }
        }
    }
}

fn particle_renderer_kind(kind: ParticleDrawKind) -> Option<RenderWorldParticleRendererKind> {
    match kind {
        ParticleDrawKind::None => None,
        ParticleDrawKind::Billboard => Some(RenderWorldParticleRendererKind::Billboard),
        ParticleDrawKind::Mesh => Some(RenderWorldParticleRendererKind::Mesh),
        ParticleDrawKind::Trail => Some(RenderWorldParticleRendererKind::Trail),
    }
}

/// Frame-scope dense list of [`FramePreparedDraw`] entries across every active render space.
///
/// Build once per frame via [`FramePreparedRenderables::build_for_frame`] and hand as a borrow to
/// every per-view [`super::collect::DrawCollectionInputs`]. Per-view collection walks this list,
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
    /// Direct lookup from one PhotonDust source renderer to all contiguous prepared rows it emits.
    particle_renderer_draw_lookup: HashMap<FramePreparedParticleRendererLookupKey, Range<usize>>,
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
            particle_renderer_draw_lookup: HashMap::new(),
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
        self.particle_renderer_draw_lookup.clear();
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

        // Retain per-space scratch capacity across frame rebuilds.
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
    fn refresh_runs_material_keys_and_chunks<S>(&mut self, scene: Option<&S>)
    where
        S: WorldMeshSceneRead + ?Sized,
    {
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
        populate_particle_renderer_draw_lookup(
            &self.draws,
            &mut self.particle_renderer_draw_lookup,
        );
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

    /// Clones a finalized base snapshot as a context-specialized prepared overlay.
    pub(super) fn clone_for_context_overlay(&self, render_context: RenderingContext) -> Self {
        Self {
            active_space_ids: self.active_space_ids.clone(),
            cached_space_draw_ranges: self.cached_space_draw_ranges.clone(),
            draws: self.draws.clone(),
            runs: self.runs.clone(),
            run_chunks: self.run_chunks.clone(),
            renderer_run_lookup: self.renderer_run_lookup.clone(),
            particle_renderer_draw_lookup: self.particle_renderer_draw_lookup.clone(),
            material_property_keys: self.material_property_keys.clone(),
            material_property_key_signature: self.material_property_key_signature,
            spatial: self.spatial.clone(),
            lod_groups: self.lod_groups.clone(),
            render_context,
            context_invariant: false,
            previous_draws: Vec::new(),
            previous_cached_space_draw_ranges: HashMap::new(),
            #[cfg(test)]
            space_scratch: Vec::new(),
            material_property_seen_scratch: HashSet::with_capacity(
                self.material_property_seen_scratch.capacity(),
            ),
        }
    }

    /// Whether per-view camera state can change renderer selection through an LOD group.
    #[inline]
    pub(crate) fn has_lod_groups(&self) -> bool {
        !self.lod_groups.is_empty()
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
    /// per-view [`super::collect::DrawCollectionViewInputs::render_context`] so material-override
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
        self.particle_renderer_draw_lookup.clear();
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

    /// Appends only the retained non-particle rows for `id` from the previous snapshot.
    ///
    /// Generated particle rows form a suffix in prepared-space order, so the stable prefix can be
    /// copied as one slice.
    pub(super) fn extend_previous_cached_non_particle_draws_for_space(
        &mut self,
        id: RenderSpaceId,
    ) -> bool {
        let Some(range) = self.previous_cached_space_draw_ranges.get(&id).cloned() else {
            return false;
        };
        let Some(draws) = self.previous_draws.get(range) else {
            return false;
        };
        let stable_end = draws
            .iter()
            .position(|draw| draw.particle_draw.kind != ParticleDrawKind::None)
            .unwrap_or(draws.len());
        debug_assert!(
            draws[stable_end..]
                .iter()
                .all(|draw| draw.particle_draw.kind != ParticleDrawKind::None),
            "prepared space order must remain static/skinned rows followed by generated particles"
        );
        self.draws.extend_from_slice(&draws[..stable_end]);
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

    /// Refits cached spatial data and rebuilds LOD metadata after dynamic bounds changed.
    ///
    /// Prepared LOD groups cache the union of their renderer AABBs, so updating draw-row cull
    /// geometry without rebuilding them leaves LOD selection on stale bounds even when the
    /// spatial index itself was refit.
    pub(super) fn refit_cached_spatial_and_lods_for_spaces<S, I>(
        &mut self,
        scene: &S,
        space_ids: I,
    ) -> usize
    where
        S: WorldMeshSceneRead + ?Sized,
        I: IntoIterator<Item = RenderSpaceId>,
    {
        let spatial_refit_count = self
            .spatial
            .refit_spaces(&self.draws, &self.runs, space_ids);
        self.rebuild_lod_groups(Some(scene));
        spatial_refit_count
    }

    /// Re-expands and patches exact static/skinned renderer ranges.
    ///
    /// Equal-size replacements with stable renderer/material/spatial shape update in place and
    /// preserve every cached range and lookup offset. Visibility, residency, slot-count, material,
    /// or renderer-identity changes splice only the affected ranges and rebuild prepared metadata
    /// once after all replacements.
    pub(super) fn patch_mesh_renderers<S>(
        &mut self,
        scene: &S,
        mesh_pool: &MeshPool,
        render_context: RenderingContext,
        dirty_renderers: &HashSet<RenderWorldRendererDirty>,
    ) -> PreparedMeshPatchStats
    where
        S: WorldMeshSceneRead + ?Sized,
    {
        profiling::scope!("mesh::prepared_renderables::patch_mesh_renderers");
        let mut dirties = dirty_renderers.iter().copied().collect::<Vec<_>>();
        dirties.sort_unstable_by_key(|dirty| {
            (
                dirty.space_id.0,
                matches!(dirty.kind, RenderWorldRendererKind::Skinned),
                dirty.renderable_index,
            )
        });

        let candidate_count = dirties.len();
        let mut replacements = Vec::with_capacity(candidate_count);
        for dirty in dirties {
            let skinned = matches!(dirty.kind, RenderWorldRendererKind::Skinned);
            let instance_id = if skinned {
                scene
                    .skinned_mesh_renderers(dirty.space_id)
                    .and_then(|renderers| renderers.get(dirty.renderable_index))
                    .map(|renderer| renderer.base.instance_id)
            } else {
                scene
                    .static_mesh_renderers(dirty.space_id)
                    .and_then(|renderers| renderers.get(dirty.renderable_index))
                    .map(|renderer| renderer.instance_id)
            };
            let old_range = instance_id
                .and_then(|instance_id| {
                    self.renderer_run_lookup
                        .get(&FramePreparedRunLookupKey {
                            space_id: dirty.space_id,
                            skinned,
                            renderable_index: dirty.renderable_index,
                            instance_id,
                        })
                        .copied()
                })
                .map(|run| run.start as usize..run.end as usize)
                .or_else(|| self.cached_mesh_renderer_draw_range(dirty));

            let mut fresh = Vec::new();
            if skinned {
                expand_skinned_renderer_into(
                    &mut fresh,
                    scene,
                    mesh_pool,
                    render_context,
                    dirty.space_id,
                    dirty.renderable_index,
                );
            } else {
                expand_static_renderer_into(
                    &mut fresh,
                    scene,
                    mesh_pool,
                    render_context,
                    dirty.space_id,
                    dirty.renderable_index,
                );
            }
            let old_range = old_range.unwrap_or_else(|| {
                let insertion = self.mesh_renderer_insertion_index(
                    dirty.space_id,
                    skinned,
                    dirty.renderable_index,
                );
                insertion..insertion
            });
            replacements.push(PreparedRangeReplacement {
                space_id: dirty.space_id,
                old_range,
                fresh,
            });
        }

        let applied = self.apply_range_replacements(scene, replacements);
        PreparedMeshPatchStats {
            candidate_count,
            range_count: applied.range_count,
            draw_count: applied.draw_count,
            noop_count: applied.noop_count,
            changed: applied.changed,
            structural_rebuild: applied.structural_rebuild,
            spatial_refit_count: applied.spatial_refit_count,
        }
    }

    /// Patches generated particle rows while retaining static/skinned prepared runs in place.
    ///
    /// Stable renderer shapes (the usual billboard/trail and fixed-count mesh-particle frame)
    /// replace their existing ranges directly, preserve run lookup offsets, and refit only touched
    /// spatial spaces. Membership, run-identity, material-key, or indexed-bounds-shape changes
    /// splice the affected particle ranges and rebuild prepared metadata once after all patches.
    pub(super) fn patch_particle_renderers<S>(
        &mut self,
        scene: &S,
        mesh_pool: &MeshPool,
        point_render_buffers: &HashMap<i32, crate::particles::PointRenderBufferAsset>,
        render_context: RenderingContext,
        dirty_spaces: &HashSet<RenderSpaceId>,
        dirty_renderers: &HashSet<RenderWorldParticleRendererDirty>,
    ) -> PreparedParticlePatchStats
    where
        S: WorldMeshSceneRead + ?Sized,
    {
        profiling::scope!("mesh::prepared_renderables::patch_particle_renderers");
        let mut full_spaces = dirty_spaces.clone();
        let mut replacements = Vec::new();

        for &dirty in dirty_renderers {
            if full_spaces.contains(&dirty.space_id) {
                continue;
            }
            let key = FramePreparedParticleRendererLookupKey::from(dirty);
            let old_range = self.particle_renderer_draw_lookup.get(&key).cloned();
            let mut fresh = Vec::new();
            expand_render_buffer_renderer_into(
                &mut fresh,
                scene,
                mesh_pool,
                point_render_buffers,
                render_context,
                dirty,
            );
            if old_range.is_none() && !fresh.is_empty() {
                // A previously filtered/non-resident row becoming drawable needs its deterministic
                // position relative to sibling particle tables restored.
                full_spaces.insert(dirty.space_id);
                continue;
            }
            if let Some(old_range) = old_range {
                replacements.push(PreparedRangeReplacement {
                    space_id: dirty.space_id,
                    old_range,
                    fresh,
                });
            }
        }

        replacements.retain(|replacement| !full_spaces.contains(&replacement.space_id));
        for &space_id in &full_spaces {
            let Some(old_range) = self.cached_particle_draw_range_for_space(space_id) else {
                continue;
            };
            let mut fresh = Vec::new();
            expand_render_buffer_renderers_into(
                &mut fresh,
                scene,
                mesh_pool,
                point_render_buffers,
                render_context,
                space_id,
            );
            replacements.push(PreparedRangeReplacement {
                space_id,
                old_range,
                fresh,
            });
        }

        let applied = self.apply_range_replacements(scene, replacements);
        PreparedParticlePatchStats {
            renderer_count: applied.range_count,
            draw_count: applied.draw_count,
            changed: applied.changed,
            structural_rebuild: applied.structural_rebuild,
            spatial_refit_count: applied.spatial_refit_count,
        }
    }

    /// Synchronizes only generated particle suffixes from a context-invariant base snapshot.
    pub(super) fn sync_particle_rows_from<S>(
        &mut self,
        source: &Self,
        scene: &S,
    ) -> PreparedParticlePatchStats
    where
        S: WorldMeshSceneRead + ?Sized,
    {
        let mut replacements = Vec::new();
        for &space_id in &self.active_space_ids {
            let Some(old_range) = self.cached_particle_draw_range_for_space(space_id) else {
                continue;
            };
            let fresh = source
                .cached_particle_draw_range_for_space(space_id)
                .and_then(|range| source.draws.get(range))
                .map_or_else(Vec::new, <[FramePreparedDraw]>::to_vec);
            replacements.push(PreparedRangeReplacement {
                space_id,
                old_range,
                fresh,
            });
        }
        let applied = self.apply_range_replacements(scene, replacements);
        PreparedParticlePatchStats {
            renderer_count: applied.range_count,
            draw_count: applied.draw_count,
            changed: applied.changed,
            structural_rebuild: applied.structural_rebuild,
            spatial_refit_count: applied.spatial_refit_count,
        }
    }

    /// Re-expands only the static/skinned and particle rows affected by one render context.
    pub(super) fn patch_context_override_renderers<S>(
        &mut self,
        scene: &S,
        mesh_pool: &MeshPool,
        point_render_buffers: &HashMap<i32, crate::particles::PointRenderBufferAsset>,
        render_context: RenderingContext,
        mesh_renderers: &HashSet<RenderWorldRendererDirty>,
        particle_renderers: &HashSet<RenderWorldParticleRendererDirty>,
    ) -> PreparedContextPatchStats
    where
        S: WorldMeshSceneRead + ?Sized,
    {
        let mesh_patch =
            self.patch_mesh_renderers(scene, mesh_pool, render_context, mesh_renderers);
        let particle_patch = self.patch_particle_renderers(
            scene,
            mesh_pool,
            point_render_buffers,
            render_context,
            &HashSet::new(),
            particle_renderers,
        );
        PreparedContextPatchStats {
            mesh_renderer_count: mesh_patch.range_count,
            particle_renderer_count: particle_patch.renderer_count,
            draw_count: mesh_patch.draw_count + particle_patch.draw_count,
            changed: mesh_patch.changed || particle_patch.changed,
            structural_rebuild_count: usize::from(mesh_patch.structural_rebuild)
                + usize::from(particle_patch.structural_rebuild),
            spatial_refit_count: mesh_patch.spatial_refit_count
                + particle_patch.spatial_refit_count,
        }
    }

    /// Finds an existing non-particle renderer run by dense table identity.
    ///
    /// The normal path performs an O(1) exact lookup with the live renderer instance id. This
    /// scan is the defensive fallback for a stale/missing scene row so a structural patch removes
    /// an old prepared run instead of inserting a duplicate beside it.
    fn cached_mesh_renderer_draw_range(
        &self,
        dirty: RenderWorldRendererDirty,
    ) -> Option<Range<usize>> {
        let skinned = matches!(dirty.kind, RenderWorldRendererKind::Skinned);
        self.renderer_run_lookup.iter().find_map(|(key, run)| {
            (key.space_id == dirty.space_id
                && key.skinned == skinned
                && key.renderable_index == dirty.renderable_index
                && self
                    .draws
                    .get(run.start as usize)
                    .is_some_and(|draw| draw.particle_draw.kind == ParticleDrawKind::None))
            .then_some(run.start as usize..run.end as usize)
        })
    }

    fn mesh_renderer_insertion_index(
        &self,
        space_id: RenderSpaceId,
        skinned: bool,
        renderable_index: usize,
    ) -> usize {
        let Some(range) = self.cached_space_draw_ranges.get(&space_id).cloned() else {
            return self.draws.len();
        };
        for draw_index in range.clone() {
            let draw = &self.draws[draw_index];
            if draw.particle_draw.kind != ParticleDrawKind::None {
                return draw_index;
            }
            let follows = if skinned {
                draw.skinned && draw.renderable_index > renderable_index
            } else {
                draw.skinned || (!draw.skinned && draw.renderable_index > renderable_index)
            };
            if follows {
                return draw_index;
            }
        }
        range.end
    }

    fn apply_range_replacements<S>(
        &mut self,
        scene: &S,
        mut replacements: Vec<PreparedRangeReplacement>,
    ) -> PreparedRangePatchStats
    where
        S: WorldMeshSceneRead + ?Sized,
    {
        if replacements.is_empty() {
            return PreparedRangePatchStats::default();
        }
        replacements.sort_by_key(|replacement| replacement.old_range.start);
        debug_assert!(
            replacements
                .windows(2)
                .all(|pair| pair[0].old_range.end <= pair[1].old_range.start),
            "prepared patch ranges must not overlap"
        );

        // Fresh expansion assigns ordinal zero. Preserve the finalized ordinal before comparing
        // payloads so an otherwise identical renderer at ordinal > 0 is still a true no-op.
        // Normalizing here also means the stable copy below does not transiently publish a stale
        // ordinal.
        let mut noop_count = 0usize;
        replacements.retain_mut(|replacement| {
            let Some(old) = self.draws.get(replacement.old_range.clone()) else {
                return true;
            };
            if old.len() == replacement.fresh.len() {
                for (previous, fresh) in old.iter().zip(&mut replacement.fresh) {
                    fresh.renderer_ordinal = previous.renderer_ordinal;
                }
            }
            if old == replacement.fresh.as_slice() {
                noop_count += 1;
                false
            } else {
                true
            }
        });
        if replacements.is_empty() {
            return PreparedRangePatchStats {
                noop_count,
                ..Default::default()
            };
        }

        let range_count = replacements
            .iter()
            .filter(|replacement| {
                !replacement.old_range.is_empty() || !replacement.fresh.is_empty()
            })
            .count();
        if range_count == 0 {
            return PreparedRangePatchStats {
                noop_count,
                ..Default::default()
            };
        }
        let draw_count = replacements
            .iter()
            .map(|replacement| replacement.fresh.len())
            .sum();
        let stable_in_place = replacements.iter().all(|replacement| {
            self.draws
                .get(replacement.old_range.clone())
                .is_some_and(|old| prepared_patch_shape_is_stable(old, &replacement.fresh))
        });
        let touched_spaces = replacements
            .iter()
            .map(|replacement| replacement.space_id)
            .collect::<HashSet<_>>();

        if stable_in_place {
            for replacement in replacements {
                let old = &mut self.draws[replacement.old_range];
                for (destination, fresh) in old.iter_mut().zip(replacement.fresh) {
                    *destination = fresh;
                }
            }
            let spatial_refit_count =
                self.refit_cached_spatial_and_lods_for_spaces(scene, touched_spaces);
            return PreparedRangePatchStats {
                range_count,
                draw_count,
                noop_count,
                changed: true,
                structural_rebuild: false,
                spatial_refit_count,
            };
        }

        for replacement in replacements.into_iter().rev() {
            self.draws
                .splice(replacement.old_range, replacement.fresh.into_iter());
        }
        self.refresh_runs_material_keys_and_chunks(Some(scene));
        PreparedRangePatchStats {
            range_count,
            draw_count,
            noop_count,
            changed: true,
            structural_rebuild: true,
            spatial_refit_count: 0,
        }
    }

    /// Returns the particle suffix range for one active prepared space.
    fn cached_particle_draw_range_for_space(
        &self,
        space_id: RenderSpaceId,
    ) -> Option<Range<usize>> {
        let space_range = self.cached_space_draw_ranges.get(&space_id)?.clone();
        let draws = self.draws.get(space_range.clone())?;
        let particle_offset = draws
            .iter()
            .position(|draw| draw.particle_draw.kind != ParticleDrawKind::None)
            .unwrap_or(draws.len());
        Some(space_range.start + particle_offset..space_range.end)
    }

    /// Finalizes a retained snapshot rebuild by refreshing runs, chunks, and material keys.
    pub(super) fn finish_cached_rebuild<S>(&mut self, scene: &S)
    where
        S: WorldMeshSceneRead + ?Sized,
    {
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
        self.refresh_runs_material_keys_and_chunks::<SceneCoordinator>(None);
    }
}

struct PreparedRangeReplacement {
    space_id: RenderSpaceId,
    old_range: Range<usize>,
    fresh: Vec<FramePreparedDraw>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PreparedRangePatchStats {
    range_count: usize,
    draw_count: usize,
    noop_count: usize,
    changed: bool,
    structural_rebuild: bool,
    spatial_refit_count: usize,
}

/// Returns whether a replacement can preserve every prepared-run/material/spatial shape.
fn prepared_patch_shape_is_stable(old: &[FramePreparedDraw], fresh: &[FramePreparedDraw]) -> bool {
    old.len() == fresh.len()
        && old.iter().zip(fresh).all(|(old, fresh)| {
            old.space_id == fresh.space_id
                && old.skinned == fresh.skinned
                && old.renderable_index == fresh.renderable_index
                && old.instance_id == fresh.instance_id
                && old.particle_draw.kind == fresh.particle_draw.kind
                && old.material_asset_id == fresh.material_asset_id
                && old.property_block_id == fresh.property_block_id
                && old
                    .cull_geometry
                    .and_then(|geometry| geometry.world_aabb)
                    .is_some()
                    == fresh
                        .cull_geometry
                        .and_then(|geometry| geometry.world_aabb)
                        .is_some()
        })
}

/// Assigns stable scene-table renderer ordinals to every prepared draw row.
fn populate_renderer_ordinals_from_scene(
    draws: &mut [FramePreparedDraw],
    scene: &(impl SceneMeshRendererRead + ?Sized),
) {
    for draw in draws {
        let static_count = scene
            .static_mesh_renderers(draw.space_id)
            .map_or(0, |renderers| renderers.len());
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
mod tests;
