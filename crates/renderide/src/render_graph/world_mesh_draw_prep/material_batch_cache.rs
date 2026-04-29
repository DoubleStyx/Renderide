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
//! [`crate::assets::material::MaterialPropertyStore`] and [`crate::materials::MaterialRouter`].
//! A frame where nothing has changed touches each live entry with one HashMap probe and four
//! `u64` comparisons — no dictionary or router lookups required.

use hashbrown::HashMap;

use crate::assets::material::{MaterialDictionary, MaterialPropertyLookupIds};
use crate::materials::{
    embedded_stem_needs_color_stream, embedded_stem_needs_extended_vertex_streams,
    embedded_stem_needs_uv0_stream, embedded_stem_requires_grab_pass,
    embedded_stem_requires_intersection_pass, embedded_stem_uses_alpha_blending,
    embedded_stem_uses_scene_color_snapshot, embedded_stem_uses_scene_depth_snapshot,
    material_blend_mode_from_maps, material_render_state_from_maps, resolve_raster_pipeline,
    MaterialBlendMode, MaterialPipelinePropertyIds, MaterialRenderState, MaterialRouter,
    RasterPipelineKind,
};
use crate::pipelines::ShaderPermutation;
use crate::scene::{MeshMaterialSlot, RenderSpaceId, SceneCoordinator, StaticMeshRenderer};

/// Read-only material-resolution context threaded through the cache refresh walker and the cached
/// batch-key lookup.
///
/// Bundles the four handles that every batch-key computation needs: the material dictionary for
/// shader assignment and property-block lookups, the router for pipeline selection, the pipeline
/// property ids for render-state decoding, and the active shader permutation.
#[derive(Copy, Clone)]
pub(super) struct MaterialResolveCtx<'a> {
    /// Material property dictionary for batch keys.
    pub dict: &'a MaterialDictionary<'a>,
    /// Shader stem / pipeline routing.
    pub router: &'a MaterialRouter,
    /// Interned material property ids that affect pipeline state.
    pub pipeline_property_ids: &'a MaterialPipelinePropertyIds,
    /// Default vs multiview permutation for embedded materials.
    pub shader_perm: ShaderPermutation,
}

/// Batch key fields derived from one `(material_asset_id, property_block_id)` pair.
///
/// All fields mirror what `batch_key_for_slot` computes on every draw; caching here avoids
/// repeating those dictionary and router lookups for every draw that uses the same material.
#[derive(Clone)]
pub(super) struct ResolvedMaterialBatch {
    /// Host shader asset id from material `set_shader` (`-1` when unknown).
    pub shader_asset_id: i32,
    /// Resolved raster pipeline kind for this material's shader.
    pub pipeline: RasterPipelineKind,
    /// Whether the active shader permutation requires a UV0 vertex stream.
    pub embedded_needs_uv0: bool,
    /// Whether the active shader permutation requires a color vertex stream.
    pub embedded_needs_color: bool,
    /// Whether the active shader permutation requires extended vertex streams (tangent, UV1-3).
    pub embedded_needs_extended_vertex_streams: bool,
    /// Whether the material requires a second forward subpass with a depth snapshot.
    pub embedded_requires_intersection_pass: bool,
    /// Whether the material requires the grab-pass transparent subpass with a scene-color snapshot.
    pub embedded_requires_grab_pass: bool,
    /// Whether the active shader permutation declares a scene-depth snapshot binding.
    pub embedded_uses_scene_depth_snapshot: bool,
    /// Whether the active shader permutation declares a scene-color snapshot binding.
    pub embedded_uses_scene_color_snapshot: bool,
    /// Resolved material blend mode.
    pub blend_mode: MaterialBlendMode,
    /// Runtime color, stencil, and depth state for this material/property-block pair.
    pub render_state: MaterialRenderState,
    /// Whether draws using this material should be sorted back-to-front.
    pub alpha_blended: bool,
}

/// Cached resolution plus the validation keys captured at resolve time.
#[derive(Clone)]
struct CacheEntry {
    batch: ResolvedMaterialBatch,
    /// Material-side mutation generation at resolve time
    /// (see [`crate::assets::material::MaterialPropertyStore::material_generation`]).
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

/// Persistent `(material_asset_id, property_block_id)` → [`ResolvedMaterialBatch`] lookup table.
///
/// Owned by [`crate::backend::RenderBackend`] and passed through per-view collection as an
/// immutable reference. Call [`Self::refresh_for_frame`] once per frame before per-view draw
/// collection: it walks every active render space, ensures every referenced key has an up-to-date
/// entry (re-resolving on generation mismatch), and evicts entries not referenced this frame.
///
/// In steady state (no material/router mutations, same shader permutation, same scene keys), this
/// pass performs one HashMap probe and four `u64` compares per unique material — no dictionary or
/// router lookups, no allocations.
pub struct FrameMaterialBatchCache {
    entries: HashMap<(i32, Option<i32>), CacheEntry>,
    /// Monotonically advanced once per [`Self::refresh_for_frame`] call. Used as a "stamp" to mark
    /// entries touched this frame; entries whose stamp does not match the current counter at the
    /// end of `refresh_for_frame` are evicted.
    frame_counter: u64,
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
        }
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
    /// `world_mesh_draw_prep`.
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
    ///   values → stamp `last_used_frame` and keep.
    /// - Otherwise → re-resolve via [`resolve_material_batch`] and overwrite.
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
        let ctx = MaterialResolveCtx {
            dict,
            router,
            pipeline_property_ids,
            shader_perm,
        };

        let active_space_ids: Vec<RenderSpaceId> = scene
            .render_space_ids()
            .filter(|id| scene.space(*id).map(|s| s.is_active).unwrap_or(false))
            .collect();

        // Phase A: collect `(material_asset_id, property_block_id)` keys per space. This is the
        // O(renderers × slots) walk; parallelising it across spaces keeps the serial Phase B work
        // bounded by unique materials rather than per-draw references.
        let keys_per_space: Vec<Vec<(i32, Option<i32>)>> = if active_space_ids.len() >= 2 {
            use rayon::prelude::*;
            active_space_ids
                .par_iter()
                .map(|&space_id| collect_material_keys_for_space(scene, space_id))
                .collect()
        } else {
            active_space_ids
                .iter()
                .map(|&space_id| collect_material_keys_for_space(scene, space_id))
                .collect()
        };

        // Phase B: serial dedup + cache probe/insert. Each unique key is touched once; the cache
        // entry's `last_used_frame` stamp makes the visit count-invariant.
        let mut seen: hashbrown::HashSet<(i32, Option<i32>)> = hashbrown::HashSet::new();
        for keys in &keys_per_space {
            for &key in keys {
                if seen.insert(key) {
                    self.touch_or_refresh(key.0, key.1, ctx, router_gen, current_frame);
                }
            }
        }

        // Evict entries not referenced this frame so the cache tracks the live working set.
        // Cheap — the cache typically holds a few dozen entries, and this touches them all once.
        self.entries
            .retain(|_, entry| entry.last_used_frame == current_frame);
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
        let property_block_gen = property_block_id
            .map(|b| ctx.dict.property_block_generation(b))
            .unwrap_or(0);

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

/// Walks one render space's renderer lists and collects every referenced
/// `(material_asset_id, property_block_id)` key. Pure — no cache mutation — so it runs in
/// parallel across spaces.
fn collect_material_keys_for_space(
    scene: &SceneCoordinator,
    space_id: RenderSpaceId,
) -> Vec<(i32, Option<i32>)> {
    let mut out = Vec::new();
    let Some(space) = scene.space(space_id) else {
        return out;
    };
    for r in &space.static_mesh_renderers {
        if r.mesh_asset_id >= 0 {
            append_renderer_material_keys(r, &mut out);
        }
    }
    for sk in &space.skinned_mesh_renderers {
        if sk.base.mesh_asset_id >= 0 {
            append_renderer_material_keys(&sk.base, &mut out);
        }
    }
    out
}

/// Appends one renderer's `(material_asset_id, property_block_id)` slot keys to `out`.
fn append_renderer_material_keys(r: &StaticMeshRenderer, out: &mut Vec<(i32, Option<i32>)>) {
    let fallback_slot;
    let slots: &[MeshMaterialSlot] = if !r.material_slots.is_empty() {
        &r.material_slots
    } else if let Some(mat_id) = r.primary_material_asset_id {
        fallback_slot = MeshMaterialSlot {
            material_asset_id: mat_id,
            property_block_id: r.primary_property_block_id,
        };
        std::slice::from_ref(&fallback_slot)
    } else {
        return;
    };
    for slot in slots {
        if slot.material_asset_id < 0 {
            continue;
        }
        out.push((slot.material_asset_id, slot.property_block_id));
    }
}

/// Computes all batch key fields for one `(material_asset_id, property_block_id)` pair.
///
/// The pipeline-kind match is collapsed into a single expression so we extract the embedded
/// stem at most once per resolve (it was re-matched five times in the previous implementation).
fn resolve_material_batch(
    material_asset_id: i32,
    property_block_id: Option<i32>,
    dict: &MaterialDictionary<'_>,
    router: &MaterialRouter,
    pipeline_property_ids: &MaterialPipelinePropertyIds,
    shader_perm: ShaderPermutation,
) -> ResolvedMaterialBatch {
    let shader_asset_id = dict
        .shader_asset_for_material(material_asset_id)
        .unwrap_or(-1);
    let pipeline = resolve_raster_pipeline(shader_asset_id, router);
    let (
        embedded_needs_uv0,
        embedded_needs_color,
        embedded_needs_extended_vertex_streams,
        embedded_requires_intersection_pass,
        embedded_requires_grab_pass,
        embedded_uses_scene_depth_snapshot,
        embedded_uses_scene_color_snapshot,
        embedded_uses_alpha_blending,
    ) = match &pipeline {
        RasterPipelineKind::EmbeddedStem(stem) => {
            let s = stem.as_ref();
            (
                embedded_stem_needs_uv0_stream(s, shader_perm),
                embedded_stem_needs_color_stream(s, shader_perm),
                embedded_stem_needs_extended_vertex_streams(s, shader_perm),
                embedded_stem_requires_intersection_pass(s, shader_perm),
                embedded_stem_requires_grab_pass(s, shader_perm),
                embedded_stem_uses_scene_depth_snapshot(s, shader_perm),
                embedded_stem_uses_scene_color_snapshot(s, shader_perm),
                embedded_stem_uses_alpha_blending(s),
            )
        }
        RasterPipelineKind::Null => (false, false, false, false, false, false, false, false),
    };
    let lookup_ids = MaterialPropertyLookupIds {
        material_asset_id,
        mesh_property_block_slot0: property_block_id,
    };
    // Fetch the two inner property maps once and reuse for both blend-mode and render-state
    // resolution: ~30 `get_merged` calls per resolve collapse to one outer-map probe per side
    // plus per-id inner-map lookups.
    let (mat_map, pb_map) = dict.fetch_property_maps(lookup_ids);
    let blend_mode = material_blend_mode_from_maps(mat_map, pb_map, pipeline_property_ids);
    let render_state = material_render_state_from_maps(mat_map, pb_map, pipeline_property_ids);
    let alpha_blended =
        embedded_uses_alpha_blending || blend_mode.is_transparent() || embedded_requires_grab_pass;
    ResolvedMaterialBatch {
        shader_asset_id,
        pipeline,
        embedded_needs_uv0,
        embedded_needs_color,
        embedded_needs_extended_vertex_streams,
        embedded_requires_intersection_pass,
        embedded_requires_grab_pass,
        embedded_uses_scene_depth_snapshot,
        embedded_uses_scene_color_snapshot,
        blend_mode,
        render_state,
        alpha_blended,
    }
}

#[cfg(test)]
mod tests {
    use crate::assets::material::{
        MaterialDictionary, MaterialPropertyStore, MaterialPropertyValue, PropertyIdRegistry,
    };
    use crate::materials::{MaterialPipelinePropertyIds, MaterialRouter, RasterPipelineKind};
    use crate::pipelines::ShaderPermutation;

    use super::{FrameMaterialBatchCache, MaterialResolveCtx};

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
        // Unknown material id → shader id -1.
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
        // last_used_frame advanced but generations did not — confirms no re-resolve.
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
        }
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
        }
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
