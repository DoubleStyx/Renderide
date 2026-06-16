//! Persistent cache of material-derived batch key fields, keyed by
//! `(material_asset_id, property_block_id)`.
//!
//! All values in [`ResolvedMaterialBatch`] are pure functions of
//! `(material_asset_id, property_block_id, shader_perm)` plus the current router state and
//! material/property-block property-store state. Caching them amortises repeated dictionary and
//! router lookups across all draws that share the same material: in a typical scene, hundreds of
//! draws share a few dozen materials.
//!
//! Unlike the previous per-frame rebuild, this cache lives across frames on [`RenderBackend`] and
//! invalidates individual entries via monotonic generation counters maintained by
//! [`crate::materials::host_data::MaterialPropertyStore`] and [`crate::materials::MaterialRouter`].
//! A frame where nothing has changed touches each live entry with one HashMap probe and four
//! `u64` comparisons -- no dictionary or router lookups required.

use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;

use crate::cpu_parallelism::{
    ParallelAdmission, RELEVANCE_PACKET_MIN_ITEMS, admit_relevance_items,
    current_reference_worker_count, record_parallel_admission,
};
use crate::materials::ShaderPermutation;
use crate::materials::host_data::MaterialDictionary;
use crate::materials::{MaterialPipelinePropertyIds, MaterialRouter};
use crate::scene::{RenderSpaceId, SceneCoordinator};
use crate::world_mesh::FramePreparedRenderables;

use super::keys::{collect_material_keys_for_space, collect_material_keys_into};
use super::resolve::{MaterialResolveCtx, ResolvedMaterialBatch, resolve_material_batch};
use super::slot::normalized_material_slot;

/// Active render spaces assigned to one material-key collection worker.
const MATERIAL_KEY_PARALLEL_CHUNK_SPACES: usize = 1;
/// Material keys assigned to one parallel material-resolution worker.
const MATERIAL_RESOLVE_PARALLEL_CHUNK_KEYS: usize = RELEVANCE_PACKET_MIN_ITEMS;
/// Material-key count required before stale/missing prepared keys resolve on Rayon workers.
const MATERIAL_RESOLVE_PARALLEL_MIN_KEYS: usize = MATERIAL_RESOLVE_PARALLEL_CHUNK_KEYS * 2;
/// Material keys assigned to one prepared-cache classification worker.
const MATERIAL_CLASSIFY_PARALLEL_CHUNK_KEYS: usize = RELEVANCE_PACKET_MIN_ITEMS;
/// Material-key count required before prepared cache classification uses Rayon.
const MATERIAL_CLASSIFY_PARALLEL_MIN_KEYS: usize = MATERIAL_CLASSIFY_PARALLEL_CHUNK_KEYS * 2;

/// Shared immutable inputs for scene-driven material cache refresh.
#[derive(Clone, Copy)]
struct SceneMaterialRefreshInputs<'a, 'b> {
    /// Scene containing active render spaces to walk.
    scene: &'a SceneCoordinator,
    /// Material resolution context used for stale or missing keys.
    ctx: MaterialResolveCtx<'b>,
    /// Material router generation captured at the start of refresh.
    router_gen: u64,
    /// Cache frame stamp assigned to entries touched during refresh.
    current_frame: u64,
}

/// Returns the material-key collection admission decision for a known worker count.
#[inline]
fn material_key_collection_admission(
    space_count: usize,
    work_units: usize,
    worker_count: usize,
) -> ParallelAdmission {
    let work_admission = admit_relevance_items(work_units, worker_count);
    if space_count >= MATERIAL_KEY_PARALLEL_CHUNK_SPACES * 2 && work_admission.is_parallel() {
        ParallelAdmission::Parallel {
            chunk_size: MATERIAL_KEY_PARALLEL_CHUNK_SPACES,
        }
    } else {
        ParallelAdmission::Serial
    }
}

/// Estimates material-slot rows scanned during frame material-key collection.
fn estimate_material_key_collection_work(
    scene: &SceneCoordinator,
    space_ids: &[RenderSpaceId],
) -> usize {
    space_ids
        .iter()
        .filter_map(|space_id| scene.space(*space_id))
        .map(|space| {
            let static_keys = space
                .static_mesh_renderers()
                .iter()
                .filter(|renderer| {
                    renderer.mesh_asset_id >= 0 && renderer.emits_visible_color_draws()
                })
                .map(estimate_renderer_material_keys)
                .sum::<usize>();
            let skinned_keys = space
                .skinned_mesh_renderers()
                .iter()
                .filter(|renderer| {
                    renderer.base.mesh_asset_id >= 0 && renderer.base.emits_visible_color_draws()
                })
                .map(|renderer| estimate_renderer_material_keys(&renderer.base))
                .sum::<usize>();
            static_keys.saturating_add(skinned_keys)
        })
        .sum()
}

/// Estimates visible material-key rows referenced by one mesh renderer.
fn estimate_renderer_material_keys(renderer: &crate::scene::StaticMeshRenderer) -> usize {
    if renderer.material_slots.is_empty() {
        usize::from(
            renderer
                .primary_material_asset_id
                .and_then(|id| normalized_material_slot(id, renderer.primary_property_block_id))
                .is_some(),
        )
    } else {
        renderer
            .material_slots
            .iter()
            .filter(|slot| {
                normalized_material_slot(slot.material_asset_id, slot.property_block_id).is_some()
            })
            .count()
    }
}

/// Cached resolution plus the validation keys captured at resolve time.
#[derive(Clone)]
struct CacheEntry {
    batch: ResolvedMaterialBatch,
    /// Material-side mutation generation at resolve time
    /// (see [`crate::materials::host_data::MaterialPropertyStore::material_generation`]).
    material_gen: u64,
    /// Property-block mutation generation at resolve time, or `0` when `property_block_id` is `None`.
    property_block_gen: u64,
    /// Router generation at resolve time (see [`MaterialRouter::generation`]).
    router_gen: u64,
    /// Shader permutation the entry was resolved for.
    shader_perm: ShaderPermutation,
    /// Cache's frame counter at the most recent touch; used to evict entries no longer referenced.
    last_used_frame: u64,
}

impl CacheEntry {
    /// Replaces the resolved batch and validation keys while preserving the hash-map allocation.
    fn refresh(
        &mut self,
        batch: ResolvedMaterialBatch,
        material_gen: u64,
        property_block_gen: u64,
        router_gen: u64,
        shader_perm: ShaderPermutation,
        last_used_frame: u64,
    ) {
        self.batch = batch;
        self.material_gen = material_gen;
        self.property_block_gen = property_block_gen;
        self.router_gen = router_gen;
        self.shader_perm = shader_perm;
        self.last_used_frame = last_used_frame;
    }
}

/// Per-refresh material batch cache touch counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct MaterialBatchCacheTouchStats {
    /// Entries that matched all generations and only needed their frame stamp advanced.
    hits: usize,
    /// Entries that existed but were re-resolved because a validation key changed.
    stale: usize,
    /// Entries inserted because the key was not present.
    misses: usize,
    /// Entries evicted because they were not touched by the current refresh.
    evicted: usize,
    /// Whole-refresh skips taken by a matching dependency snapshot and prepared live-set signature.
    fast_path_skips: usize,
}

impl MaterialBatchCacheTouchStats {
    /// Records one per-key touch outcome.
    fn note(&mut self, outcome: TouchOutcome) {
        match outcome {
            TouchOutcome::Hit => self.hits += 1,
            TouchOutcome::Stale => self.stale += 1,
            TouchOutcome::Miss => self.misses += 1,
        }
    }
}

/// Result of touching one material/property-block cache key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TouchOutcome {
    /// Existing cache entry was valid.
    Hit,
    /// Existing cache entry was present but stale and had to be refreshed.
    Stale,
    /// No cache entry existed for the key.
    Miss,
}

/// Cache key requiring a fresh material-batch resolve after serial classification.
#[derive(Clone, Copy)]
struct PendingMaterialResolve {
    /// Material asset id from the prepared draw live set.
    material_asset_id: i32,
    /// Optional mesh property-block id paired with the material.
    property_block_id: Option<i32>,
    /// Material-side mutation generation captured during classification.
    material_gen: u64,
    /// Property-block mutation generation captured during classification.
    property_block_gen: u64,
}

/// Immutable classification result for one prepared material key.
struct PreparedMaterialClassification {
    /// Material asset id from the prepared draw live set.
    material_asset_id: i32,
    /// Optional mesh property-block id paired with the material.
    property_block_id: Option<i32>,
    /// Material-side mutation generation captured during classification.
    material_gen: u64,
    /// Property-block mutation generation captured during classification.
    property_block_gen: u64,
    /// Cache touch outcome that should be applied serially.
    outcome: TouchOutcome,
}

impl PendingMaterialResolve {
    /// Resolves this stale or missing key using immutable material state.
    fn resolve(
        self,
        ctx: MaterialResolveCtx<'_>,
        router_gen: u64,
        current_frame: u64,
    ) -> ResolvedMaterialCacheUpdate {
        let batch = resolve_material_batch(
            self.material_asset_id,
            self.property_block_id,
            ctx.dict,
            ctx.router,
            ctx.pipeline_property_ids,
            ctx.shader_perm,
        );
        ResolvedMaterialCacheUpdate {
            material_asset_id: self.material_asset_id,
            property_block_id: self.property_block_id,
            batch,
            material_gen: self.material_gen,
            property_block_gen: self.property_block_gen,
            router_gen,
            shader_perm: ctx.shader_perm,
            last_used_frame: current_frame,
        }
    }
}

/// Fully resolved cache entry staged for a serial apply phase.
struct ResolvedMaterialCacheUpdate {
    /// Material asset id from the prepared draw live set.
    material_asset_id: i32,
    /// Optional mesh property-block id paired with the material.
    property_block_id: Option<i32>,
    /// Resolved material batch fields.
    batch: ResolvedMaterialBatch,
    /// Material-side mutation generation at resolve time.
    material_gen: u64,
    /// Property-block mutation generation at resolve time.
    property_block_gen: u64,
    /// Router generation at resolve time.
    router_gen: u64,
    /// Shader permutation the entry was resolved for.
    shader_perm: ShaderPermutation,
    /// Cache frame stamp to assign when applying the update.
    last_used_frame: u64,
}

/// Persistent `(material_asset_id, property_block_id)` -> [`ResolvedMaterialBatch`] lookup table.
///
/// Owned by the renderer host and passed through per-view collection as an immutable reference.
/// Call [`Self::refresh_for_frame`] once per frame before per-view draw
/// collection: it walks every active render space, ensures every referenced key has an up-to-date
/// entry (re-resolving on generation mismatch), and evicts entries not referenced this frame.
///
/// In steady state (no material/router mutations, same shader permutation, same scene keys), this
/// pass performs one HashMap probe and four `u64` compares per unique material -- no dictionary or
/// router lookups, no allocations.
pub struct FrameMaterialBatchCache {
    entries: HashMap<(i32, Option<i32>), CacheEntry>,
    /// Monotonically advanced once per [`Self::refresh_for_frame`] call. Used as a "stamp" to mark
    /// entries touched this frame; entries whose stamp does not match the current counter at the
    /// end of `refresh_for_frame` are evicted.
    frame_counter: u64,
    /// Reused per-frame deduplication set for `(material_asset_id, property_block_id)` keys
    /// observed during [`Self::refresh_for_frame`]; cleared at the top of every refresh and
    /// repopulated.
    seen_scratch: HashSet<(i32, Option<i32>)>,
    /// Reused active-space-id list for the multi-space refresh path; cleared at the top of every
    /// [`Self::refresh_for_frame`] that needs it.
    active_scratch: Vec<RenderSpaceId>,
    /// Reused outer/inner key buffers for the multi-space refresh path. The outer [`Vec`] is
    /// cleared and resized to the active-space count; each inner [`Vec`] is cleared inside the
    /// rayon worker before [`collect_material_keys_into`] re-fills it. Capacities persist.
    keys_per_space_scratch: Vec<Vec<(i32, Option<i32>)>>,
    /// Snapshot of the inputs that determine whether a refresh would re-resolve any entry. When
    /// the next refresh sees the same triple, no host-side material state has changed since the
    /// last walk: the cache fast-paths by stamping every existing entry's `last_used_frame` so
    /// eviction preserves them. Newly referenced materials (none in steady state) fall through to
    /// the slow path in [`super::resolve::batch_key_for_slot_cached`], which resolves directly via
    /// [`super::resolve::batch_key_for_slot`].
    last_refresh_router_gen: Option<u64>,
    /// Snapshot of [`crate::materials::host_data::MaterialPropertyStore::global_generation`] at
    /// the most recent refresh, paired with [`Self::last_refresh_router_gen`].
    last_refresh_dict_global_gen: Option<u64>,
    /// Snapshot of the [`ShaderPermutation`] the cache was last refreshed for; the gate skips the
    /// walk only when the next refresh targets the same permutation.
    last_refresh_shader_perm: Option<ShaderPermutation>,
    /// Signature of the prepared material live set used by the most recent prepared refresh.
    ///
    /// `None` means the most recent refresh did not come from a prepared snapshot, so prepared
    /// refreshes must walk keys before they can trust membership.
    last_refresh_prepared_material_signature: Option<u64>,
    /// Counters from the most recent refresh.
    last_touch_stats: MaterialBatchCacheTouchStats,
}

impl Default for FrameMaterialBatchCache {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameMaterialBatchCache {
    /// Creates an empty cache.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            frame_counter: 0,
            seen_scratch: HashSet::new(),
            active_scratch: Vec::new(),
            keys_per_space_scratch: Vec::new(),
            last_refresh_router_gen: None,
            last_refresh_dict_global_gen: None,
            last_refresh_shader_perm: None,
            last_refresh_prepared_material_signature: None,
            last_touch_stats: MaterialBatchCacheTouchStats::default(),
        }
    }

    /// Returns `true` and stamps every entry's `last_used_frame` to `current_frame` when the
    /// inputs that determine cache-entry resolution are unchanged since the last refresh.
    ///
    /// Callers use the result to skip the per-pair walk: any draw that references a still-cached
    /// material reads the existing entry, while a draw referencing a freshly added material falls
    /// through to the slow path in
    /// [`crate::world_mesh::materials::resolve::batch_key_for_slot_cached`]. New materials show
    /// up exclusively after a host mutation (which bumps the global generation and disqualifies
    /// the gate), so the slow-path fall-through is rare in practice.
    fn try_fast_path_skip(
        &mut self,
        router_gen: u64,
        dict_global_gen: u64,
        shader_perm: ShaderPermutation,
        current_frame: u64,
    ) -> bool {
        if self.last_refresh_router_gen == Some(router_gen)
            && self.last_refresh_dict_global_gen == Some(dict_global_gen)
            && self.last_refresh_shader_perm == Some(shader_perm)
        {
            for entry in self.entries.values_mut() {
                entry.last_used_frame = current_frame;
            }
            true
        } else {
            false
        }
    }

    /// Records the snapshot of `(router_gen, dict_global_gen, shader_perm)` that the most recent
    /// refresh resolved against. Read by [`Self::try_fast_path_skip`] on the next refresh.
    fn record_refresh_snapshot(
        &mut self,
        router_gen: u64,
        dict_global_gen: u64,
        shader_perm: ShaderPermutation,
        prepared_material_signature: Option<u64>,
    ) {
        self.last_refresh_router_gen = Some(router_gen);
        self.last_refresh_dict_global_gen = Some(dict_global_gen);
        self.last_refresh_shader_perm = Some(shader_perm);
        self.last_refresh_prepared_material_signature = prepared_material_signature;
    }

    /// Returns `true` when both the material dependency generations and prepared live-set
    /// signature match the most recent prepared refresh.
    fn try_prepared_fast_path_skip(
        &self,
        router_gen: u64,
        dict_global_gen: u64,
        shader_perm: ShaderPermutation,
        prepared_material_signature: u64,
    ) -> bool {
        self.last_refresh_router_gen == Some(router_gen)
            && self.last_refresh_dict_global_gen == Some(dict_global_gen)
            && self.last_refresh_shader_perm == Some(shader_perm)
            && self.last_refresh_prepared_material_signature == Some(prepared_material_signature)
    }

    /// Number of cached entries (debug / diagnostics).
    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns a cached entry without inserting.
    ///
    /// Restricted to `pub(super)` because [`ResolvedMaterialBatch`] is internal to
    /// the world-mesh material resolution module.
    pub(super) fn get(
        &self,
        material_asset_id: i32,
        property_block_id: Option<i32>,
    ) -> Option<&ResolvedMaterialBatch> {
        self.entries
            .get(&(material_asset_id, property_block_id))
            .map(|e| &e.batch)
    }

    /// Walks active scene renderers and refreshes every referenced material key.
    fn refresh_scene_material_keys(
        &mut self,
        inputs: SceneMaterialRefreshInputs<'_, '_>,
    ) -> MaterialBatchCacheTouchStats {
        let mut active_space_ids = inputs
            .scene
            .render_space_ids()
            .filter(|id| inputs.scene.space(*id).is_some_and(|s| s.is_active()));
        let first = active_space_ids.next();
        let second = active_space_ids.next();
        let mut seen = std::mem::take(&mut self.seen_scratch);
        seen.clear();
        let mut touch_stats = MaterialBatchCacheTouchStats::default();

        match (first, second) {
            (None, _) => {}
            (Some(only), None) => {
                self.refresh_single_space_material_keys(inputs, only, &mut seen, &mut touch_stats);
            }
            (Some(first), Some(second)) => {
                self.refresh_multi_space_material_keys(
                    inputs,
                    [first, second].into_iter().chain(active_space_ids),
                    &mut seen,
                    &mut touch_stats,
                );
            }
        }

        self.seen_scratch = seen;
        touch_stats
    }

    /// Refreshes cache entries referenced by one active render space.
    fn refresh_single_space_material_keys(
        &mut self,
        inputs: SceneMaterialRefreshInputs<'_, '_>,
        space_id: RenderSpaceId,
        seen: &mut HashSet<(i32, Option<i32>)>,
        touch_stats: &mut MaterialBatchCacheTouchStats,
    ) {
        for key in collect_material_keys_for_space(inputs.scene, space_id) {
            if seen.insert(key) {
                touch_stats.note(self.touch_or_refresh(
                    key.0,
                    key.1,
                    inputs.ctx,
                    inputs.router_gen,
                    inputs.current_frame,
                ));
            }
        }
    }

    /// Refreshes cache entries referenced by multiple active render spaces.
    fn refresh_multi_space_material_keys(
        &mut self,
        inputs: SceneMaterialRefreshInputs<'_, '_>,
        active_space_ids: impl Iterator<Item = RenderSpaceId>,
        seen: &mut HashSet<(i32, Option<i32>)>,
        touch_stats: &mut MaterialBatchCacheTouchStats,
    ) {
        let mut active = std::mem::take(&mut self.active_scratch);
        active.clear();
        active.extend(active_space_ids);
        let work_units = estimate_material_key_collection_work(inputs.scene, &active);
        let admission = material_key_collection_admission(
            active.len(),
            work_units,
            current_reference_worker_count(),
        );
        record_parallel_admission(
            "material_key_collection",
            work_units,
            active.len(),
            admission,
        );

        let mut keys_per_space = std::mem::take(&mut self.keys_per_space_scratch);
        keys_per_space.resize_with(active.len(), Vec::new);
        if admission.is_parallel() {
            keys_per_space
                .par_iter_mut()
                .with_min_len(MATERIAL_KEY_PARALLEL_CHUNK_SPACES)
                .zip(
                    active
                        .par_iter()
                        .with_min_len(MATERIAL_KEY_PARALLEL_CHUNK_SPACES),
                )
                .for_each(|(out, &space_id)| {
                    out.clear();
                    collect_material_keys_into(inputs.scene, space_id, out);
                });
        } else {
            for (out, &space_id) in keys_per_space.iter_mut().zip(active.iter()) {
                out.clear();
                collect_material_keys_into(inputs.scene, space_id, out);
            }
        }

        for keys in &keys_per_space {
            for &key in keys {
                if seen.insert(key) {
                    touch_stats.note(self.touch_or_refresh(
                        key.0,
                        key.1,
                        inputs.ctx,
                        inputs.router_gen,
                        inputs.current_frame,
                    ));
                }
            }
        }

        self.active_scratch = active;
        self.keys_per_space_scratch = keys_per_space;
    }

    /// Refreshes the cache against the current scene and dependency state.
    ///
    /// Walks every active render space once, for each referenced
    /// `(material_asset_id, property_block_id)` key:
    ///
    /// - If an entry exists and all stored generations / shader permutation match the current
    ///   values -> stamp `last_used_frame` and keep.
    /// - Otherwise -> re-resolve via [`resolve_material_batch`] and overwrite.
    ///
    /// After the walk, entries not touched this frame are evicted so the cache size tracks the
    /// live working set. Call once per frame before any per-view draw collection that reads the
    /// cache.
    pub fn refresh_for_frame(
        &mut self,
        scene: &SceneCoordinator,
        dict: &MaterialDictionary<'_>,
        router: &MaterialRouter,
        pipeline_property_ids: &MaterialPipelinePropertyIds,
        shader_perm: ShaderPermutation,
    ) {
        profiling::scope!("mesh::material_batch_cache_refresh_for_frame");
        self.frame_counter = self.frame_counter.wrapping_add(1);
        let current_frame = self.frame_counter;
        let router_gen = router.generation();
        let dict_global_gen = dict.global_generation();
        let fast_path_skip = {
            profiling::scope!("mesh::material_batch_cache::prepared_fast_path");
            self.try_fast_path_skip(router_gen, dict_global_gen, shader_perm, current_frame)
        };
        if fast_path_skip {
            self.last_touch_stats = MaterialBatchCacheTouchStats {
                fast_path_skips: 1,
                ..Default::default()
            };
            return;
        }
        let ctx = MaterialResolveCtx {
            dict,
            router,
            pipeline_property_ids,
            shader_perm,
        };
        let mut touch_stats = self.refresh_scene_material_keys(SceneMaterialRefreshInputs {
            scene,
            ctx,
            router_gen,
            current_frame,
        });

        // Evict entries not referenced this frame so the cache tracks the live working set.
        // Cheap -- the cache typically holds a few dozen entries, and this touches them all once.
        let entry_count_before_evict = self.entries.len();
        self.entries
            .retain(|_, entry| entry.last_used_frame == current_frame);
        touch_stats.evicted = entry_count_before_evict.saturating_sub(self.entries.len());
        self.last_touch_stats = touch_stats;
        self.record_refresh_snapshot(router_gen, dict_global_gen, shader_perm, None);
    }

    /// Refreshes the cache from a pre-expanded draw list instead of walking scene renderers.
    ///
    /// `FramePreparedRenderables` already resolves render-context material overrides and
    /// per-slot property blocks once for the frame. Reusing those keys avoids a second
    /// O(renderers x material slots) scene walk in `render::build_frame_material_cache`.
    ///
    /// The prepared snapshot exposes first-seen unique keys, so this path touches each material
    /// once per shader permutation and does not allocate or run a second per-draw dedup pass.
    pub fn refresh_for_prepared(
        &mut self,
        prepared: &FramePreparedRenderables,
        dict: &MaterialDictionary<'_>,
        router: &MaterialRouter,
        pipeline_property_ids: &MaterialPipelinePropertyIds,
        shader_perm: ShaderPermutation,
    ) {
        profiling::scope!("mesh::material_batch_cache_refresh_for_prepared");
        self.frame_counter = self.frame_counter.wrapping_add(1);
        let current_frame = self.frame_counter;
        let router_gen = router.generation();
        let dict_global_gen = dict.global_generation();
        let prepared_material_signature = prepared.material_property_key_signature();
        let fast_path_skip = {
            profiling::scope!("mesh::material_batch_cache::prepared_fast_path_check");
            self.try_prepared_fast_path_skip(
                router_gen,
                dict_global_gen,
                shader_perm,
                prepared_material_signature,
            )
        };
        if fast_path_skip {
            profiling::scope!("mesh::material_batch_cache::prepared_fast_path_skip");
            self.last_touch_stats = MaterialBatchCacheTouchStats {
                fast_path_skips: 1,
                ..Default::default()
            };
            return;
        }
        let ctx = MaterialResolveCtx {
            dict,
            router,
            pipeline_property_ids,
            shader_perm,
        };
        let (mut touch_stats, pending_resolves) = {
            profiling::scope!("mesh::material_batch_cache::prepared_classify");
            self.classify_prepared_keys(
                prepared.unique_material_property_pairs(),
                ctx,
                router_gen,
                current_frame,
            )
        };

        let resolved_updates = {
            profiling::scope!("mesh::material_batch_cache::prepared_resolve");
            resolve_pending_material_batches(pending_resolves, ctx, router_gen, current_frame)
        };
        {
            profiling::scope!("mesh::material_batch_cache::prepared_apply_resolved");
            self.apply_resolved_material_updates(resolved_updates);
        }

        {
            profiling::scope!("mesh::material_batch_cache::prepared_evict_unused");
            let entry_count_before_evict = self.entries.len();
            self.entries
                .retain(|_, entry| entry.last_used_frame == current_frame);
            touch_stats.evicted = entry_count_before_evict.saturating_sub(self.entries.len());
        }
        {
            profiling::scope!("mesh::material_batch_cache::prepared_record_snapshot");
            self.record_refresh_snapshot(
                router_gen,
                dict_global_gen,
                shader_perm,
                Some(prepared_material_signature),
            );
        }
        self.last_touch_stats = touch_stats;
    }

    /// Classifies one prepared key, stamping valid hits and staging stale/missing keys for
    /// immutable parallel resolution.
    fn classify_prepared_key(
        &mut self,
        material_asset_id: i32,
        property_block_id: Option<i32>,
        ctx: MaterialResolveCtx<'_>,
        router_gen: u64,
        current_frame: u64,
        pending_resolves: &mut Vec<PendingMaterialResolve>,
    ) -> TouchOutcome {
        profiling::scope!("mesh::material_batch_cache::classify_prepared_key");
        let material_gen = ctx.dict.material_generation(material_asset_id);
        let property_block_gen =
            property_block_id.map_or(0, |b| ctx.dict.property_block_generation(b));
        let key = (material_asset_id, property_block_id);
        match self.entries.get_mut(&key) {
            Some(entry)
                if entry.material_gen == material_gen
                    && entry.property_block_gen == property_block_gen
                    && entry.router_gen == router_gen
                    && entry.shader_perm == ctx.shader_perm =>
            {
                entry.last_used_frame = current_frame;
                TouchOutcome::Hit
            }
            Some(_) => {
                pending_resolves.push(PendingMaterialResolve {
                    material_asset_id,
                    property_block_id,
                    material_gen,
                    property_block_gen,
                });
                TouchOutcome::Stale
            }
            None => {
                pending_resolves.push(PendingMaterialResolve {
                    material_asset_id,
                    property_block_id,
                    material_gen,
                    property_block_gen,
                });
                TouchOutcome::Miss
            }
        }
    }

    /// Classifies all prepared keys, using Rayon once the prepared live set has two useful chunks.
    fn classify_prepared_keys(
        &mut self,
        keys: &[(i32, Option<i32>)],
        ctx: MaterialResolveCtx<'_>,
        router_gen: u64,
        current_frame: u64,
    ) -> (MaterialBatchCacheTouchStats, Vec<PendingMaterialResolve>) {
        let mut touch_stats = MaterialBatchCacheTouchStats::default();
        let mut pending_resolves = Vec::new();
        let admission = admit_relevance_items(keys.len(), current_reference_worker_count());
        record_parallel_admission("material_classify", keys.len(), keys.len(), admission);
        if keys.len() >= MATERIAL_CLASSIFY_PARALLEL_MIN_KEYS && admission.is_parallel() {
            profiling::scope!("mesh::material_batch_cache::prepared_classify_parallel");
            let chunk_size = admission
                .chunk_size()
                .unwrap_or(MATERIAL_CLASSIFY_PARALLEL_CHUNK_KEYS);
            let classified = keys
                .par_iter()
                .with_min_len(chunk_size)
                .map(|&(material_asset_id, property_block_id)| {
                    self.classify_prepared_key_immutable(
                        material_asset_id,
                        property_block_id,
                        ctx,
                        router_gen,
                    )
                })
                .collect::<Vec<_>>();
            for classified in classified {
                self.apply_prepared_key_classification(
                    classified,
                    current_frame,
                    &mut touch_stats,
                    &mut pending_resolves,
                );
            }
            return (touch_stats, pending_resolves);
        }

        profiling::scope!("mesh::material_batch_cache::prepared_classify_serial");
        for &(material_asset_id, property_block_id) in keys {
            let outcome = self.classify_prepared_key(
                material_asset_id,
                property_block_id,
                ctx,
                router_gen,
                current_frame,
                &mut pending_resolves,
            );
            touch_stats.note(outcome);
        }
        (touch_stats, pending_resolves)
    }

    /// Classifies one prepared key without mutating the cache.
    fn classify_prepared_key_immutable(
        &self,
        material_asset_id: i32,
        property_block_id: Option<i32>,
        ctx: MaterialResolveCtx<'_>,
        router_gen: u64,
    ) -> PreparedMaterialClassification {
        let material_gen = ctx.dict.material_generation(material_asset_id);
        let property_block_gen =
            property_block_id.map_or(0, |b| ctx.dict.property_block_generation(b));
        let key = (material_asset_id, property_block_id);
        let outcome = match self.entries.get(&key) {
            Some(entry)
                if entry.material_gen == material_gen
                    && entry.property_block_gen == property_block_gen
                    && entry.router_gen == router_gen
                    && entry.shader_perm == ctx.shader_perm =>
            {
                TouchOutcome::Hit
            }
            Some(_) => TouchOutcome::Stale,
            None => TouchOutcome::Miss,
        };
        PreparedMaterialClassification {
            material_asset_id,
            property_block_id,
            material_gen,
            property_block_gen,
            outcome,
        }
    }

    /// Applies one immutable classification result in prepared-key order.
    fn apply_prepared_key_classification(
        &mut self,
        classified: PreparedMaterialClassification,
        current_frame: u64,
        touch_stats: &mut MaterialBatchCacheTouchStats,
        pending_resolves: &mut Vec<PendingMaterialResolve>,
    ) {
        touch_stats.note(classified.outcome);
        match classified.outcome {
            TouchOutcome::Hit => {
                if let Some(entry) = self
                    .entries
                    .get_mut(&(classified.material_asset_id, classified.property_block_id))
                {
                    entry.last_used_frame = current_frame;
                }
            }
            TouchOutcome::Stale | TouchOutcome::Miss => {
                pending_resolves.push(PendingMaterialResolve {
                    material_asset_id: classified.material_asset_id,
                    property_block_id: classified.property_block_id,
                    material_gen: classified.material_gen,
                    property_block_gen: classified.property_block_gen,
                });
            }
        }
    }

    /// Applies resolved prepared-cache updates after the parallel resolution phase has completed.
    fn apply_resolved_material_updates(&mut self, updates: Vec<ResolvedMaterialCacheUpdate>) {
        for update in updates {
            let key = (update.material_asset_id, update.property_block_id);
            match self.entries.get_mut(&key) {
                Some(entry) => entry.refresh(
                    update.batch,
                    update.material_gen,
                    update.property_block_gen,
                    update.router_gen,
                    update.shader_perm,
                    update.last_used_frame,
                ),
                None => {
                    self.entries.insert(
                        key,
                        CacheEntry {
                            batch: update.batch,
                            material_gen: update.material_gen,
                            property_block_gen: update.property_block_gen,
                            router_gen: update.router_gen,
                            shader_perm: update.shader_perm,
                            last_used_frame: update.last_used_frame,
                        },
                    );
                }
            }
        }
    }

    /// Ensures the cache has a valid entry for `(material_asset_id, property_block_id)` and
    /// stamps it as used this frame. Resolves / re-resolves on miss or generation mismatch.
    fn touch_or_refresh(
        &mut self,
        material_asset_id: i32,
        property_block_id: Option<i32>,
        ctx: MaterialResolveCtx<'_>,
        router_gen: u64,
        current_frame: u64,
    ) -> TouchOutcome {
        profiling::scope!("mesh::material_batch_cache::touch_or_refresh");
        let material_gen = ctx.dict.material_generation(material_asset_id);
        let property_block_gen =
            property_block_id.map_or(0, |b| ctx.dict.property_block_generation(b));

        let key = (material_asset_id, property_block_id);
        match self.entries.get_mut(&key) {
            Some(entry)
                if entry.material_gen == material_gen
                    && entry.property_block_gen == property_block_gen
                    && entry.router_gen == router_gen
                    && entry.shader_perm == ctx.shader_perm =>
            {
                profiling::scope!("mesh::material_batch_cache::touch_hit");
                entry.last_used_frame = current_frame;
                TouchOutcome::Hit
            }
            Some(entry) => {
                profiling::scope!("mesh::material_batch_cache::touch_stale");
                let batch = resolve_material_batch(
                    material_asset_id,
                    property_block_id,
                    ctx.dict,
                    ctx.router,
                    ctx.pipeline_property_ids,
                    ctx.shader_perm,
                );
                entry.refresh(
                    batch,
                    material_gen,
                    property_block_gen,
                    router_gen,
                    ctx.shader_perm,
                    current_frame,
                );
                TouchOutcome::Stale
            }
            None => {
                profiling::scope!("mesh::material_batch_cache::touch_miss");
                let batch = resolve_material_batch(
                    material_asset_id,
                    property_block_id,
                    ctx.dict,
                    ctx.router,
                    ctx.pipeline_property_ids,
                    ctx.shader_perm,
                );
                self.entries.insert(
                    key,
                    CacheEntry {
                        batch,
                        material_gen,
                        property_block_gen,
                        router_gen,
                        shader_perm: ctx.shader_perm,
                        last_used_frame: current_frame,
                    },
                );
                TouchOutcome::Miss
            }
        }
    }
}

/// Resolves all stale or missing prepared cache keys using immutable material state.
fn resolve_pending_material_batches(
    pending: Vec<PendingMaterialResolve>,
    ctx: MaterialResolveCtx<'_>,
    router_gen: u64,
    current_frame: u64,
) -> Vec<ResolvedMaterialCacheUpdate> {
    if pending.is_empty() {
        return Vec::new();
    }
    let admission = admit_relevance_items(pending.len(), current_reference_worker_count());
    record_parallel_admission("material_resolve", pending.len(), pending.len(), admission);
    if pending.len() >= MATERIAL_RESOLVE_PARALLEL_MIN_KEYS && admission.is_parallel() {
        profiling::scope!("mesh::material_batch_cache::prepared_resolve_parallel");
        let chunk_size = admission
            .chunk_size()
            .unwrap_or(MATERIAL_RESOLVE_PARALLEL_CHUNK_KEYS);
        pending
            .par_iter()
            .with_min_len(chunk_size)
            .map(|pending| pending.resolve(ctx, router_gen, current_frame))
            .collect()
    } else {
        profiling::scope!("mesh::material_batch_cache::prepared_resolve_serial");
        pending
            .into_iter()
            .map(|pending| pending.resolve(ctx, router_gen, current_frame))
            .collect()
    }
}

#[cfg(test)]
mod tests;
