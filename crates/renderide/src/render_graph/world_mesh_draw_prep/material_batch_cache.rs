//! Per-frame cache of material-derived batch key fields, keyed by `(material_asset_id, property_block_id)`.
//!
//! All values in [`ResolvedMaterialBatch`] are functions of
//! `(material_asset_id, property_block_id, shader_perm)` and are constant for the lifetime of a
//! single collection call. Caching them amortises repeated dictionary and router lookups across
//! all draws that share the same material: in a typical scene, hundreds of draws share a few dozen
//! materials.

use hashbrown::HashMap;
use rayon::prelude::*;

use crate::assets::material::{MaterialDictionary, MaterialPropertyLookupIds};
use crate::materials::{
    embedded_stem_needs_color_stream, embedded_stem_needs_extended_vertex_streams,
    embedded_stem_needs_uv0_stream, embedded_stem_requires_intersection_pass,
    embedded_stem_uses_alpha_blending, material_blend_mode_for_lookup,
    material_render_state_for_lookup, resolve_raster_pipeline, MaterialBlendMode,
    MaterialPipelinePropertyIds, MaterialRenderState, MaterialRouter, RasterPipelineKind,
};
use crate::pipelines::ShaderPermutation;
use crate::scene::{MeshMaterialSlot, RenderSpaceId, SceneCoordinator, StaticMeshRenderer};

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
    /// Resolved material blend mode.
    pub blend_mode: MaterialBlendMode,
    /// Runtime color, stencil, and depth state for this material/property-block pair.
    pub render_state: MaterialRenderState,
    /// Whether draws using this material should be sorted back-to-front.
    pub alpha_blended: bool,
}

/// Per-frame lookup table mapping `(material_asset_id, property_block_id)` →
/// [`ResolvedMaterialBatch`].
///
/// Built lazily via [`Self::get_or_insert`] on first miss, or eagerly by calling
/// [`Self::build_for_frame`] to walk all active render spaces once before per-view draw
/// collection. Once built, the table can be shared as an immutable reference across rayon
/// workers — the per-view collection path only reads from it.
///
/// Hoist the build out of per-view collection when rendering multiple views in one frame
/// (secondary render-texture cameras + main swapchain): without hoisting, the cache is rebuilt
/// N+1 times and every `(material_asset_id, property_block_id)` lookup pays the dictionary /
/// router resolution cost repeatedly.
pub struct FrameMaterialBatchCache {
    entries: HashMap<(i32, Option<i32>), ResolvedMaterialBatch>,
    shader_perm: ShaderPermutation,
}

impl FrameMaterialBatchCache {
    /// Creates an empty cache for the given shader permutation.
    pub fn new(shader_perm: ShaderPermutation) -> Self {
        Self {
            entries: HashMap::new(),
            shader_perm,
        }
    }

    /// Returns the cached [`ResolvedMaterialBatch`] for the given key, inserting on first miss.
    ///
    /// Restricted to `pub(super)` because [`ResolvedMaterialBatch`] is internal to
    /// `world_mesh_draw_prep`. External callers interact with the cache through
    /// [`Self::build_for_frame`] and pass the returned cache into [`DrawCollectionContext`] — the
    /// per-draw lookup happens inside this module.
    pub(super) fn get_or_insert(
        &mut self,
        material_asset_id: i32,
        property_block_id: Option<i32>,
        dict: &MaterialDictionary<'_>,
        router: &MaterialRouter,
        pipeline_property_ids: &MaterialPipelinePropertyIds,
    ) -> &ResolvedMaterialBatch {
        let shader_perm = self.shader_perm;
        self.entries
            .entry((material_asset_id, property_block_id))
            .or_insert_with(|| {
                resolve_material_batch(
                    material_asset_id,
                    property_block_id,
                    dict,
                    router,
                    pipeline_property_ids,
                    shader_perm,
                )
            })
    }

    /// Returns a cached entry without inserting.
    ///
    /// Returns `None` when the entry was never populated via [`Self::get_or_insert`].
    ///
    /// Restricted to `pub(super)` for the same reason as [`Self::get_or_insert`].
    pub(super) fn get(
        &self,
        material_asset_id: i32,
        property_block_id: Option<i32>,
    ) -> Option<&ResolvedMaterialBatch> {
        self.entries.get(&(material_asset_id, property_block_id))
    }

    /// Clears all entries while retaining allocated capacity for reuse.
    ///
    /// Reserved for a future pooled-cache optimization; currently a fresh cache is built each
    /// frame via [`Self::new`].
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Moves entries from `other` into this cache, keeping the first-seen value on key collision.
    ///
    /// Used by the parallel warm-up path: each rayon worker builds a per-space local cache, then
    /// the main thread folds them into a single shared cache before the collection phase begins.
    /// Both caches must carry the same [`ShaderPermutation`]; entries resolved from the same
    /// `(material_asset_id, property_block_id)` pair are functions of that pair plus the shader
    /// permutation, so duplicate keys between caches are equivalent and the retained value is
    /// semantically arbitrary.
    pub fn merge_from(&mut self, other: FrameMaterialBatchCache) {
        debug_assert_eq!(
            self.shader_perm, other.shader_perm,
            "merging caches with mismatched shader permutations"
        );
        self.entries.reserve(other.entries.len());
        for (k, v) in other.entries {
            self.entries.entry(k).or_insert(v);
        }
    }

    /// Builds a fully-warmed cache by walking every active render space's static and skinned
    /// renderer lists once.
    ///
    /// Per-space warm-up runs in parallel via [`rayon`] when more than one space is active; the
    /// main thread then folds the per-space caches into one shared cache. Entries are pure
    /// functions of their key plus `shader_perm`, so merge order does not matter.
    ///
    /// Call this **once per frame** before per-view draw collection in multi-view paths so each
    /// view's collection can share the same immutable cache. Single-view callers can keep calling
    /// the lazy [`Self::get_or_insert`] path instead.
    pub fn build_for_frame(
        scene: &SceneCoordinator,
        dict: &MaterialDictionary<'_>,
        router: &MaterialRouter,
        pipeline_property_ids: &MaterialPipelinePropertyIds,
        shader_perm: ShaderPermutation,
    ) -> Self {
        profiling::scope!("mesh::material_batch_cache_build_for_frame");
        let active_space_ids: Vec<RenderSpaceId> = scene
            .render_space_ids()
            .filter(|id| scene.space(*id).map(|s| s.is_active).unwrap_or(false))
            .collect();

        if active_space_ids.len() <= 1 {
            let mut cache = Self::new(shader_perm);
            for space_id in active_space_ids {
                warm_cache_for_space(
                    &mut cache,
                    scene,
                    space_id,
                    dict,
                    router,
                    pipeline_property_ids,
                );
            }
            return cache;
        }

        let per_space: Vec<Self> = active_space_ids
            .par_iter()
            .map(|&space_id| {
                let mut local = Self::new(shader_perm);
                warm_cache_for_space(
                    &mut local,
                    scene,
                    space_id,
                    dict,
                    router,
                    pipeline_property_ids,
                );
                local
            })
            .collect();

        let mut merged = Self::new(shader_perm);
        for local in per_space {
            merged.merge_from(local);
        }
        merged
    }
}

/// Warms `cache` with every `(material_asset_id, property_block_id)` pair used by one render space.
fn warm_cache_for_space(
    cache: &mut FrameMaterialBatchCache,
    scene: &SceneCoordinator,
    space_id: RenderSpaceId,
    dict: &MaterialDictionary<'_>,
    router: &MaterialRouter,
    pipeline_property_ids: &MaterialPipelinePropertyIds,
) {
    let Some(space) = scene.space(space_id) else {
        return;
    };
    for r in &space.static_mesh_renderers {
        if r.mesh_asset_id >= 0 {
            warm_cache_for_renderer(cache, r, dict, router, pipeline_property_ids);
        }
    }
    for sk in &space.skinned_mesh_renderers {
        if sk.base.mesh_asset_id >= 0 {
            warm_cache_for_renderer(cache, &sk.base, dict, router, pipeline_property_ids);
        }
    }
}

/// Warms `cache` with every material slot referenced by one static or skinned base renderer.
fn warm_cache_for_renderer(
    cache: &mut FrameMaterialBatchCache,
    r: &StaticMeshRenderer,
    dict: &MaterialDictionary<'_>,
    router: &MaterialRouter,
    pipeline_property_ids: &MaterialPipelinePropertyIds,
) {
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
        cache.get_or_insert(
            slot.material_asset_id,
            slot.property_block_id,
            dict,
            router,
            pipeline_property_ids,
        );
    }
}

/// Computes all batch key fields for one `(material_asset_id, property_block_id)` pair.
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
    let embedded_needs_uv0 = match &pipeline {
        RasterPipelineKind::EmbeddedStem(stem) => {
            embedded_stem_needs_uv0_stream(stem.as_ref(), shader_perm)
        }
        RasterPipelineKind::DebugWorldNormals => false,
    };
    let embedded_needs_color = match &pipeline {
        RasterPipelineKind::EmbeddedStem(stem) => {
            embedded_stem_needs_color_stream(stem.as_ref(), shader_perm)
        }
        RasterPipelineKind::DebugWorldNormals => false,
    };
    let embedded_needs_extended_vertex_streams = match &pipeline {
        RasterPipelineKind::EmbeddedStem(stem) => {
            embedded_stem_needs_extended_vertex_streams(stem.as_ref(), shader_perm)
        }
        RasterPipelineKind::DebugWorldNormals => false,
    };
    let embedded_requires_intersection_pass = match &pipeline {
        RasterPipelineKind::EmbeddedStem(stem) => {
            embedded_stem_requires_intersection_pass(stem.as_ref(), shader_perm)
        }
        RasterPipelineKind::DebugWorldNormals => false,
    };
    let lookup_ids = MaterialPropertyLookupIds {
        material_asset_id,
        mesh_property_block_slot0: property_block_id,
    };
    let blend_mode = material_blend_mode_for_lookup(dict, lookup_ids, pipeline_property_ids);
    let render_state = material_render_state_for_lookup(dict, lookup_ids, pipeline_property_ids);
    let alpha_blended = match &pipeline {
        RasterPipelineKind::EmbeddedStem(stem) => embedded_stem_uses_alpha_blending(stem.as_ref()),
        RasterPipelineKind::DebugWorldNormals => false,
    } || blend_mode.is_transparent();
    ResolvedMaterialBatch {
        shader_asset_id,
        pipeline,
        embedded_needs_uv0,
        embedded_needs_color,
        embedded_needs_extended_vertex_streams,
        embedded_requires_intersection_pass,
        blend_mode,
        render_state,
        alpha_blended,
    }
}

#[cfg(test)]
mod tests {
    use crate::assets::material::{MaterialDictionary, MaterialPropertyStore, PropertyIdRegistry};
    use crate::materials::{MaterialPipelinePropertyIds, MaterialRouter, RasterPipelineKind};
    use crate::pipelines::ShaderPermutation;

    use super::FrameMaterialBatchCache;

    fn make_test_deps() -> (MaterialPropertyStore, MaterialRouter, PropertyIdRegistry) {
        let store = MaterialPropertyStore::new();
        let router = MaterialRouter::new(RasterPipelineKind::DebugWorldNormals);
        let reg = PropertyIdRegistry::new();
        (store, router, reg)
    }

    #[test]
    fn get_or_insert_caches_by_material_id() {
        let (store, router, reg) = make_test_deps();
        let dict = MaterialDictionary::new(&store);
        let ids = MaterialPipelinePropertyIds::new(&reg);
        let mut cache = FrameMaterialBatchCache::new(ShaderPermutation(0));

        let a = cache
            .get_or_insert(42, None, &dict, &router, &ids)
            .shader_asset_id;
        let b = cache
            .get_or_insert(42, None, &dict, &router, &ids)
            .shader_asset_id;
        assert_eq!(a, b);
        // Unknown material → shader id -1.
        assert_eq!(a, -1);
    }

    #[test]
    fn distinct_material_ids_produce_separate_entries() {
        let (store, router, reg) = make_test_deps();
        let dict = MaterialDictionary::new(&store);
        let ids = MaterialPipelinePropertyIds::new(&reg);
        let mut cache = FrameMaterialBatchCache::new(ShaderPermutation(0));

        cache.get_or_insert(1, None, &dict, &router, &ids);
        cache.get_or_insert(2, None, &dict, &router, &ids);
        assert_eq!(cache.entries.len(), 2);
        assert!(cache.get(1, None).is_some());
        assert!(cache.get(2, None).is_some());
        assert!(cache.get(99, None).is_none());
    }

    #[test]
    fn property_block_id_produces_separate_entry() {
        let (store, router, reg) = make_test_deps();
        let dict = MaterialDictionary::new(&store);
        let ids = MaterialPipelinePropertyIds::new(&reg);
        let mut cache = FrameMaterialBatchCache::new(ShaderPermutation(0));

        cache.get_or_insert(10, None, &dict, &router, &ids);
        cache.get_or_insert(10, Some(99), &dict, &router, &ids);
        assert_eq!(cache.entries.len(), 2);
    }

    #[test]
    fn clear_removes_entries_but_retains_capacity() {
        let (store, router, reg) = make_test_deps();
        let dict = MaterialDictionary::new(&store);
        let ids = MaterialPipelinePropertyIds::new(&reg);
        let mut cache = FrameMaterialBatchCache::new(ShaderPermutation(0));

        cache.get_or_insert(5, None, &dict, &router, &ids);
        cache.clear();
        assert_eq!(cache.entries.len(), 0);
        assert!(cache.get(5, None).is_none());
    }
}
