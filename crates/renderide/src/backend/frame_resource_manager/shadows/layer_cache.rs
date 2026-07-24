//! Persistent shadow atlas layer assignment with content-signature reuse.
//!
//! Each planned shadow view is keyed by a stable light identity derived from its packed light
//! fields. A cache entry pins the atlas array layer for that identity across frames and stores a
//! content signature of the last rendered layer. When the signature matches and the atlas texture
//! itself was not recreated, the layer's depth texels are still valid and the layer is skipped
//! during recording. Entries not used by the current plan release their layer back to a free
//! list, so the atlas layer count stays bounded by the peak simultaneous shadow-view count.

use hashbrown::{HashMap, HashSet};
use std::hash::{BuildHasher, Hasher};

use crate::gpu_pools::MeshPool;

/// Cached state for one persistent shadow atlas layer.
struct ShadowLayerCacheEntry {
    /// Atlas array layer pinned to this light identity.
    layer: u32,
    /// Light-parameter signature of the layer's last rendered depth contents.
    params_sig: u64,
    /// Caster content signature of the last render, when it was computed.
    ///
    /// [`None`] means the last render skipped the caster scan (its light parameters had
    /// changed, so the scan would have been wasted work) or the members were not reusable.
    /// A pending caster signature costs at most one extra re-render once the parameters
    /// stabilize again.
    caster_sig: Option<u64>,
    /// Whether the layer currently holds rendered contents matching the signatures.
    rendered: bool,
    /// Planning frame that last used this entry.
    last_used_frame: u64,
}

/// Result of acquiring a persistent layer for one planned shadow view.
pub(super) struct ShadowLayerLease {
    /// Atlas array layer assigned to the light identity.
    pub(super) layer: u32,
    /// Light-parameter signature of valid rendered contents already in the layer.
    pub(super) rendered_params_sig: Option<u64>,
    /// Caster signature of those contents, when the last render computed one.
    pub(super) rendered_caster_sig: Option<u64>,
}

/// Persistent shadow layer allocator and reuse tracker.
pub(in crate::backend::frame_resource_manager) struct ShadowLayerCache {
    entries: HashMap<u64, ShadowLayerCacheEntry>,
    free_layers: Vec<u32>,
    layer_count: u32,
    frame: u64,
    hasher: hashbrown::DefaultHashBuilder,
    mesh_mutation_generation: u64,
    mutated_meshes: HashSet<i32>,
    all_meshes_mutated: bool,
}

impl ShadowLayerCache {
    pub(in crate::backend::frame_resource_manager) fn new() -> Self {
        Self {
            entries: HashMap::new(),
            free_layers: Vec::new(),
            layer_count: 0,
            frame: 0,
            hasher: hashbrown::DefaultHashBuilder::default(),
            mesh_mutation_generation: 0,
            mutated_meshes: HashSet::new(),
            all_meshes_mutated: true,
        }
    }

    /// Starts a new planning frame and refreshes the mutated-mesh set from the pool delta.
    pub(super) fn begin_frame(&mut self, mesh_pool: Option<&MeshPool>) {
        self.frame = self.frame.wrapping_add(1);
        self.mutated_meshes.clear();
        let Some(pool) = mesh_pool else {
            self.all_meshes_mutated = true;
            return;
        };
        let delta = pool.mutation_delta_since(self.mesh_mutation_generation);
        self.all_meshes_mutated = delta.requires_full_rebuild;
        if !delta.requires_full_rebuild {
            self.mutated_meshes.extend(delta.changed_asset_ids);
        }
        self.mesh_mutation_generation = delta.current_generation;
    }

    /// Whether `mesh_asset_id` had resident mesh data replaced since the previous plan.
    pub(super) fn mesh_mutated(&self, mesh_asset_id: i32) -> bool {
        self.all_meshes_mutated || self.mutated_meshes.contains(&mesh_asset_id)
    }

    /// Returns a session-stable streaming hasher for composite signatures.
    pub(super) fn build_hasher(&self) -> impl Hasher {
        self.hasher.build_hasher()
    }

    /// Acquires the persistent layer for `key` and marks it used this frame.
    pub(super) fn acquire(&mut self, key: u64) -> ShadowLayerLease {
        let free_layers = &mut self.free_layers;
        let layer_count = &mut self.layer_count;
        let entry = self.entries.entry(key).or_insert_with(|| {
            let layer = free_layers.pop().unwrap_or_else(|| {
                let layer = *layer_count;
                *layer_count = layer_count.saturating_add(1);
                layer
            });
            ShadowLayerCacheEntry {
                layer,
                params_sig: 0,
                caster_sig: None,
                rendered: false,
                last_used_frame: 0,
            }
        });
        entry.last_used_frame = self.frame;
        ShadowLayerLease {
            layer: entry.layer,
            rendered_params_sig: entry.rendered.then_some(entry.params_sig),
            rendered_caster_sig: entry.rendered.then_some(entry.caster_sig).flatten(),
        }
    }

    /// Records that `key`'s layer will hold contents matching these signatures after this frame.
    pub(super) fn mark_rendered(&mut self, key: u64, params_sig: u64, caster_sig: Option<u64>) {
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.params_sig = params_sig;
            entry.caster_sig = caster_sig;
            entry.rendered = true;
        }
    }

    /// Whether `key` holds valid rendered contents matching both signatures.
    pub(super) fn rendered_contents_match(
        &self,
        key: u64,
        params_sig: u64,
        caster_sig: Option<u64>,
    ) -> bool {
        self.entries.get(&key).is_some_and(|entry| {
            entry.rendered
                && entry.params_sig == params_sig
                && caster_sig.is_some()
                && entry.caster_sig == caster_sig
        })
    }

    /// Releases layers whose identities were not used by the current plan.
    pub(super) fn evict_unused(&mut self) {
        let frame = self.frame;
        let free_layers = &mut self.free_layers;
        self.entries.retain(|_, entry| {
            let keep = entry.last_used_frame == frame;
            if !keep {
                free_layers.push(entry.layer);
            }
            keep
        });
    }

    /// Marks every cached layer as holding invalid contents (atlas texture recreated).
    pub(super) fn invalidate_rendered_contents(&mut self) {
        for entry in self.entries.values_mut() {
            entry.rendered = false;
        }
    }

    /// Atlas array layers needed to cover every pinned assignment.
    pub(super) fn layer_count(&self) -> u32 {
        self.layer_count
    }
}

#[cfg(test)]
mod tests {
    use super::ShadowLayerCache;

    #[test]
    fn acquire_pins_layer_across_frames() {
        let mut cache = ShadowLayerCache::new();
        cache.begin_frame(None);
        let first = cache.acquire(7).layer;
        cache.evict_unused();
        cache.begin_frame(None);
        let second = cache.acquire(7).layer;
        assert_eq!(first, second);
    }

    #[test]
    fn distinct_keys_get_distinct_layers() {
        let mut cache = ShadowLayerCache::new();
        cache.begin_frame(None);
        let a = cache.acquire(1).layer;
        let b = cache.acquire(2).layer;
        assert_ne!(a, b);
        assert_eq!(cache.layer_count(), 2);
    }

    #[test]
    fn unused_layers_are_recycled() {
        let mut cache = ShadowLayerCache::new();
        cache.begin_frame(None);
        let a = cache.acquire(1).layer;
        cache.evict_unused();
        cache.begin_frame(None);
        let b = cache.acquire(2).layer;
        cache.evict_unused();
        assert_ne!(a, b);
        cache.begin_frame(None);
        let c = cache.acquire(3).layer;
        cache.evict_unused();
        assert_eq!(a, c);
        assert_eq!(cache.layer_count(), 2);
    }

    #[test]
    fn rendered_signatures_survive_until_invalidated() {
        let mut cache = ShadowLayerCache::new();
        cache.begin_frame(None);
        let lease = cache.acquire(3);
        assert_eq!(lease.rendered_params_sig, None);
        cache.mark_rendered(3, 99, Some(7));
        cache.evict_unused();
        cache.begin_frame(None);
        let lease = cache.acquire(3);
        assert_eq!(lease.rendered_params_sig, Some(99));
        assert_eq!(lease.rendered_caster_sig, Some(7));
        assert!(cache.rendered_contents_match(3, 99, Some(7)));
        assert!(!cache.rendered_contents_match(3, 99, None));
        assert!(!cache.rendered_contents_match(3, 98, Some(7)));
        cache.invalidate_rendered_contents();
        cache.evict_unused();
        cache.begin_frame(None);
        assert_eq!(cache.acquire(3).rendered_params_sig, None);
    }

    #[test]
    fn pending_caster_signature_reports_no_match() {
        let mut cache = ShadowLayerCache::new();
        cache.begin_frame(None);
        let _ = cache.acquire(4);
        cache.mark_rendered(4, 50, None);
        cache.evict_unused();
        cache.begin_frame(None);
        let lease = cache.acquire(4);
        assert_eq!(lease.rendered_params_sig, Some(50));
        assert_eq!(lease.rendered_caster_sig, None);
        assert!(!cache.rendered_contents_match(4, 50, Some(1)));
    }

    #[test]
    fn missing_mesh_pool_marks_all_meshes_mutated() {
        let mut cache = ShadowLayerCache::new();
        cache.begin_frame(None);
        assert!(cache.mesh_mutated(42));
    }
}
