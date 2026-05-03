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

use hashbrown::HashMap;

use crate::materials::ShaderPermutation;
use crate::materials::host_data::MaterialDictionary;
use crate::materials::{MaterialPipelinePropertyIds, MaterialRouter};
use crate::scene::{RenderSpaceId, SceneCoordinator};
use crate::world_mesh::FramePreparedRenderables;

use super::keys::{collect_material_keys_for_space, collect_material_keys_into};
use super::resolve::{MaterialResolveCtx, ResolvedMaterialBatch, resolve_material_batch};

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
    seen_scratch: hashbrown::HashSet<(i32, Option<i32>)>,
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
            seen_scratch: hashbrown::HashSet::new(),
            active_scratch: Vec::new(),
            keys_per_space_scratch: Vec::new(),
            last_refresh_router_gen: None,
            last_refresh_dict_global_gen: None,
            last_refresh_shader_perm: None,
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
    ) {
        self.last_refresh_router_gen = Some(router_gen);
        self.last_refresh_dict_global_gen = Some(dict_global_gen);
        self.last_refresh_shader_perm = Some(shader_perm);
    }

    /// Clears all entries while retaining allocated capacity.
    pub fn clear(&mut self) {
        self.entries.clear();
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
            return;
        }
        let ctx = MaterialResolveCtx {
            dict,
            router,
            pipeline_property_ids,
            shader_perm,
        };

        // Walk active spaces lazily so the single-space steady state skips the
        // `Vec<RenderSpaceId>` allocation entirely.
        let mut active_space_ids = scene
            .render_space_ids()
            .filter(|id| scene.space(*id).is_some_and(|s| s.is_active));
        let first = active_space_ids.next();
        let second = active_space_ids.next();

        // Pull cross-frame scratch out so it can be passed around independently of `&mut self`.
        // `mem::take` leaves a default (empty, allocation-less) container behind; we restore the
        // populated containers (with their grown capacities) before returning.
        let mut seen = std::mem::take(&mut self.seen_scratch);
        seen.clear();

        match (first, second) {
            (None, _) => {}
            (Some(only), None) => {
                // Single-space fast path: probe directly without intermediate Vec allocations.
                for key in collect_material_keys_for_space(scene, only) {
                    if seen.insert(key) {
                        self.touch_or_refresh(key.0, key.1, ctx, router_gen, current_frame);
                    }
                }
            }
            (Some(first), Some(second)) => {
                let mut active = std::mem::take(&mut self.active_scratch);
                active.clear();
                active.reserve(2 + active_space_ids.size_hint().0);
                active.push(first);
                active.push(second);
                active.extend(active_space_ids);

                // Phase A: collect `(material_asset_id, property_block_id)` keys per space in
                // parallel. The walk is O(renderers x slots); parallelising it across spaces
                // keeps the serial Phase B work bounded by unique materials rather than per-draw
                // references. Inner `Vec`s are reused across frames and cleared in place so the
                // collect routine appends without reallocating in steady state.
                let mut keys_per_space = std::mem::take(&mut self.keys_per_space_scratch);
                keys_per_space.resize_with(active.len(), Vec::new);
                use rayon::prelude::*;
                keys_per_space
                    .par_iter_mut()
                    .zip(active.par_iter())
                    .for_each(|(out, &space_id)| {
                        out.clear();
                        collect_material_keys_into(scene, space_id, out);
                    });

                // Phase B: serial dedup + cache probe/insert. Each unique key is touched once;
                // the cache entry's `last_used_frame` stamp makes the visit count-invariant.
                for keys in &keys_per_space {
                    for &key in keys {
                        if seen.insert(key) {
                            self.touch_or_refresh(key.0, key.1, ctx, router_gen, current_frame);
                        }
                    }
                }

                // Restore scratch (capacities retained).
                self.active_scratch = active;
                self.keys_per_space_scratch = keys_per_space;
            }
        }

        // Restore the dedup scratch (capacity retained).
        self.seen_scratch = seen;

        // Evict entries not referenced this frame so the cache tracks the live working set.
        // Cheap -- the cache typically holds a few dozen entries, and this touches them all once.
        self.entries
            .retain(|_, entry| entry.last_used_frame == current_frame);
        self.record_refresh_snapshot(router_gen, dict_global_gen, shader_perm);
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
        if self.try_fast_path_skip(router_gen, dict_global_gen, shader_perm, current_frame) {
            return;
        }
        let ctx = MaterialResolveCtx {
            dict,
            router,
            pipeline_property_ids,
            shader_perm,
        };

        {
            profiling::scope!("mesh::material_batch_cache::prepared_serial_dedup_touch");
            for &(material_asset_id, property_block_id) in prepared.unique_material_property_pairs()
            {
                self.touch_or_refresh(
                    material_asset_id,
                    property_block_id,
                    ctx,
                    router_gen,
                    current_frame,
                );
            }
        }

        {
            profiling::scope!("mesh::material_batch_cache::prepared_evict_unused");
            self.entries
                .retain(|_, entry| entry.last_used_frame == current_frame);
        }
        {
            profiling::scope!("mesh::material_batch_cache::prepared_record_snapshot");
            self.record_refresh_snapshot(router_gen, dict_global_gen, shader_perm);
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
    ) {
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
            }
            _ => {
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
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::materials::ShaderPermutation;
    use crate::materials::host_data::{
        MaterialDictionary, MaterialPropertyStore, MaterialPropertyValue, PropertyIdRegistry,
    };
    use crate::materials::{MaterialPipelinePropertyIds, MaterialRouter, RasterPipelineKind};

    use super::FrameMaterialBatchCache;
    use crate::world_mesh::materials::MaterialResolveCtx;

    fn make_test_deps() -> (MaterialPropertyStore, MaterialRouter, PropertyIdRegistry) {
        let store = MaterialPropertyStore::new();
        let router = MaterialRouter::new(RasterPipelineKind::Null);
        let reg = PropertyIdRegistry::new();
        (store, router, reg)
    }

    /// Directly exercise the private `touch_or_refresh` path so we can unit-test generation
    /// invalidation without setting up a `SceneCoordinator`. `refresh_for_frame` is the
    /// production entry; it wraps the same per-key logic over a scene walk.
    fn touch(
        cache: &mut FrameMaterialBatchCache,
        mat: i32,
        pb: Option<i32>,
        ctx: MaterialResolveCtx<'_>,
        frame: u64,
    ) {
        cache.frame_counter = frame;
        let rgen = ctx.router.generation();
        cache.touch_or_refresh(mat, pb, ctx, rgen, frame);
    }

    /// Helper that bundles the four handles into a [`MaterialResolveCtx`] for a test call site.
    fn make_ctx<'a>(
        dict: &'a MaterialDictionary<'a>,
        router: &'a MaterialRouter,
        ids: &'a MaterialPipelinePropertyIds,
        perm: ShaderPermutation,
    ) -> MaterialResolveCtx<'a> {
        MaterialResolveCtx {
            dict,
            router,
            pipeline_property_ids: ids,
            shader_perm: perm,
        }
    }

    #[test]
    fn first_touch_resolves_and_inserts_entry() {
        let (store, router, reg) = make_test_deps();
        let dict = MaterialDictionary::new(&store);
        let ids = MaterialPipelinePropertyIds::new(&reg);
        let mut cache = FrameMaterialBatchCache::new();
        touch(
            &mut cache,
            42,
            None,
            make_ctx(&dict, &router, &ids, ShaderPermutation(0)),
            1,
        );
        assert!(cache.get(42, None).is_some());
        // Unknown material id -> shader id -1.
        assert_eq!(cache.get(42, None).unwrap().shader_asset_id, -1);
    }

    #[test]
    fn unchanged_entry_is_reused_without_reresolve() {
        let (store, router, reg) = make_test_deps();
        let dict = MaterialDictionary::new(&store);
        let ids = MaterialPipelinePropertyIds::new(&reg);
        let mut cache = FrameMaterialBatchCache::new();
        touch(
            &mut cache,
            1,
            None,
            make_ctx(&dict, &router, &ids, ShaderPermutation(0)),
            1,
        );
        let before = cache.entries.get(&(1, None)).unwrap().clone();
        touch(
            &mut cache,
            1,
            None,
            make_ctx(&dict, &router, &ids, ShaderPermutation(0)),
            2,
        );
        let after = cache.entries.get(&(1, None)).unwrap();
        assert_eq!(before.material_gen, after.material_gen);
        assert_eq!(before.router_gen, after.router_gen);
        // last_used_frame advanced but generations did not -- confirms no re-resolve.
        assert_eq!(after.last_used_frame, 2);
    }

    #[test]
    fn material_mutation_invalidates_entry() {
        let (mut store, router, reg) = make_test_deps();
        let ids = MaterialPipelinePropertyIds::new(&reg);
        let mut cache = FrameMaterialBatchCache::new();
        {
            let dict = MaterialDictionary::new(&store);
            touch(
                &mut cache,
                1,
                None,
                make_ctx(&dict, &router, &ids, ShaderPermutation(0)),
                1,
            );
        };
        let gen_before = cache.entries.get(&(1, None)).unwrap().material_gen;
        store.set_material(1, 7, MaterialPropertyValue::Float(0.25));
        {
            let dict = MaterialDictionary::new(&store);
            touch(
                &mut cache,
                1,
                None,
                make_ctx(&dict, &router, &ids, ShaderPermutation(0)),
                2,
            );
        };
        let gen_after = cache.entries.get(&(1, None)).unwrap().material_gen;
        assert_ne!(gen_before, gen_after);
    }

    #[test]
    fn router_mutation_invalidates_entry() {
        let (store, mut router, reg) = make_test_deps();
        let dict = MaterialDictionary::new(&store);
        let ids = MaterialPipelinePropertyIds::new(&reg);
        let mut cache = FrameMaterialBatchCache::new();
        touch(
            &mut cache,
            1,
            None,
            make_ctx(&dict, &router, &ids, ShaderPermutation(0)),
            1,
        );
        let rgen_before = cache.entries.get(&(1, None)).unwrap().router_gen;
        router.set_shader_pipeline(
            7,
            RasterPipelineKind::EmbeddedStem(std::sync::Arc::from("x_default")),
        );
        touch(
            &mut cache,
            1,
            None,
            make_ctx(&dict, &router, &ids, ShaderPermutation(0)),
            2,
        );
        let rgen_after = cache.entries.get(&(1, None)).unwrap().router_gen;
        assert_ne!(rgen_before, rgen_after);
    }

    #[test]
    fn shader_perm_mismatch_triggers_reresolve() {
        let (store, router, reg) = make_test_deps();
        let dict = MaterialDictionary::new(&store);
        let ids = MaterialPipelinePropertyIds::new(&reg);
        let mut cache = FrameMaterialBatchCache::new();
        touch(
            &mut cache,
            1,
            None,
            make_ctx(&dict, &router, &ids, ShaderPermutation(0)),
            1,
        );
        touch(
            &mut cache,
            1,
            None,
            make_ctx(&dict, &router, &ids, ShaderPermutation(1)),
            2,
        );
        assert_eq!(
            cache.entries.get(&(1, None)).unwrap().shader_perm,
            ShaderPermutation(1)
        );
    }

    #[test]
    fn property_block_id_produces_separate_entry() {
        let (store, router, reg) = make_test_deps();
        let dict = MaterialDictionary::new(&store);
        let ids = MaterialPipelinePropertyIds::new(&reg);
        let mut cache = FrameMaterialBatchCache::new();
        touch(
            &mut cache,
            10,
            None,
            make_ctx(&dict, &router, &ids, ShaderPermutation(0)),
            1,
        );
        touch(
            &mut cache,
            10,
            Some(99),
            make_ctx(&dict, &router, &ids, ShaderPermutation(0)),
            1,
        );
        assert_eq!(cache.len(), 2);
    }
}
