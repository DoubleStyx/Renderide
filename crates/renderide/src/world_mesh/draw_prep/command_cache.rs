//! Retained CPU command-list cache for arranged world-mesh draw collections.

use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use hashbrown::HashMap;
use parking_lot::Mutex;

use super::arrange::arrange_draw_chunks_by_phase_bins;
use super::item::{WorldMeshDrawArrangementStats, WorldMeshDrawItem};

const WORLD_MESH_COMMAND_CACHE_CAPACITY: usize = 256;

/// Runtime counters for the retained world-mesh command cache.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorldMeshCommandCacheStats {
    /// Retained arranged command lists currently resident in the cache.
    pub entries: usize,
    /// Cache lookups that reused an arranged draw list.
    pub hits: u64,
    /// Cache lookups that had to rebuild the arranged draw list.
    pub misses: u64,
    /// New arranged draw lists inserted into the cache.
    pub insertions: u64,
    /// Entries evicted to keep the cache bounded.
    pub evictions: u64,
}

/// Bounded renderer-level cache for pre-arranged world-mesh draw command lists.
#[derive(Debug, Default)]
pub(crate) struct WorldMeshCommandCache {
    inner: Mutex<WorldMeshCommandCacheInner>,
}

#[derive(Debug, Default)]
struct WorldMeshCommandCacheInner {
    entries: HashMap<WorldMeshCommandCacheKey, WorldMeshCommandCacheEntry>,
    recency: VecDeque<WorldMeshCommandCacheKey>,
    stats: WorldMeshCommandCacheStats,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct WorldMeshCommandCacheKey {
    fingerprint: u64,
    draw_count: usize,
    chunk_count: usize,
}

#[derive(Clone, Debug)]
struct WorldMeshCommandCacheEntry {
    items: Arc<[WorldMeshDrawItem]>,
    arrangement: WorldMeshDrawArrangementStats,
}

impl WorldMeshCommandCache {
    /// Returns a cached arrangement for `chunks` or builds and retains one when absent.
    pub(crate) fn arrange_draw_chunks(
        &self,
        chunks: Vec<Vec<WorldMeshDrawItem>>,
        allow_parallel_sort: bool,
    ) -> (Vec<WorldMeshDrawItem>, WorldMeshDrawArrangementStats) {
        profiling::scope!("mesh::arrange_command_cache");
        let key = WorldMeshCommandCacheKey::from_chunks(&chunks);
        if let Some(entry) = self.entry(&key) {
            return (entry.items.as_ref().to_vec(), entry.arrangement);
        }

        let (items, arrangement) = arrange_draw_chunks_by_phase_bins(chunks, allow_parallel_sort);
        self.insert(key, &items, arrangement);
        (items, arrangement)
    }

    /// Captures a point-in-time diagnostic snapshot of the command cache.
    pub(crate) fn stats(&self) -> WorldMeshCommandCacheStats {
        let inner = self.inner.lock();
        let mut stats = inner.stats;
        stats.entries = inner.entries.len();
        stats
    }

    fn entry(&self, key: &WorldMeshCommandCacheKey) -> Option<WorldMeshCommandCacheEntry> {
        let mut inner = self.inner.lock();
        let entry = inner.entries.get(key).cloned();
        if entry.is_some() {
            inner.stats.hits = inner.stats.hits.saturating_add(1);
            inner.recency.push_back(*key);
        } else {
            inner.stats.misses = inner.stats.misses.saturating_add(1);
        }
        entry
    }

    fn insert(
        &self,
        key: WorldMeshCommandCacheKey,
        items: &[WorldMeshDrawItem],
        arrangement: WorldMeshDrawArrangementStats,
    ) {
        let mut inner = self.inner.lock();
        if let Some(entry) = inner.entries.get_mut(&key) {
            entry.items = Arc::from(items.to_vec());
            entry.arrangement = arrangement;
            inner.recency.push_back(key);
            drop(inner);
            return;
        }
        inner.entries.insert(
            key,
            WorldMeshCommandCacheEntry {
                items: Arc::from(items.to_vec()),
                arrangement,
            },
        );
        inner.recency.push_back(key);
        inner.stats.insertions = inner.stats.insertions.saturating_add(1);
        while inner.entries.len() > WORLD_MESH_COMMAND_CACHE_CAPACITY {
            let Some(candidate) = inner.recency.pop_front() else {
                break;
            };
            if inner.entries.remove(&candidate).is_some() {
                inner.stats.evictions = inner.stats.evictions.saturating_add(1);
            }
        }
        drop(inner);
    }
}

impl WorldMeshCommandCacheKey {
    fn from_chunks(chunks: &[Vec<WorldMeshDrawItem>]) -> Self {
        let draw_count = chunks.iter().map(Vec::len).sum::<usize>();
        Self {
            fingerprint: fingerprint_world_mesh_draw_chunks(chunks),
            draw_count,
            chunk_count: chunks.len(),
        }
    }
}

fn fingerprint_world_mesh_draw_chunks(chunks: &[Vec<WorldMeshDrawItem>]) -> u64 {
    let mut hasher = ahash::AHasher::default();
    chunks.len().hash(&mut hasher);
    for chunk in chunks {
        chunk.len().hash(&mut hasher);
        hash_world_mesh_draw_items(chunk, &mut hasher);
    }
    hasher.finish()
}

/// Computes a deterministic structural fingerprint for an already ordered draw slice.
pub(crate) fn fingerprint_world_mesh_draws(draws: &[WorldMeshDrawItem]) -> u64 {
    let mut hasher = ahash::AHasher::default();
    draws.len().hash(&mut hasher);
    hash_world_mesh_draw_items(draws, &mut hasher);
    hasher.finish()
}

fn hash_world_mesh_draw_items<H: Hasher>(draws: &[WorldMeshDrawItem], hasher: &mut H) {
    for item in draws {
        hash_world_mesh_draw_item(item, hasher);
    }
}

fn hash_world_mesh_draw_item<H: Hasher>(item: &WorldMeshDrawItem, hasher: &mut H) {
    item.space_id.hash(hasher);
    item.node_id.hash(hasher);
    item.renderable_index.hash(hasher);
    item.instance_id.hash(hasher);
    item.mesh_asset_id.hash(hasher);
    item.slot_index.hash(hasher);
    item.first_index.hash(hasher);
    item.index_count.hash(hasher);
    item.is_overlay.hash(hasher);
    item.sorting_order.hash(hasher);
    item.skinned.hash(hasher);
    item.world_space_deformed.hash(hasher);
    item.blendshape_deformed.hash(hasher);
    hash_camera_distance_if_ordered(item, hasher);
    item.lookup_ids.material_asset_id.hash(hasher);
    item.lookup_ids.mesh_property_block_slot0.hash(hasher);
    item.lookup_ids.mesh_renderer_property_block_id.hash(hasher);
    item.batch_key.hash(hasher);
    item.batch_key_hash.hash(hasher);
    item._opaque_depth_bucket.hash(hasher);
    item.sort_prefix.hash(hasher);
    hash_mat4_option(item.rigid_world_matrix, hasher);
    hash_reflection_probe_selection(item, hasher);
    hash_vec4_option(item.ui_rect_clip_local, hasher);
}

fn hash_camera_distance_if_ordered<H: Hasher>(item: &WorldMeshDrawItem, hasher: &mut H) {
    if item.batch_key.requires_strict_order() {
        item.camera_distance_sq.to_bits().hash(hasher);
    }
}

fn hash_mat4_option<H: Hasher>(value: Option<glam::Mat4>, hasher: &mut H) {
    value.is_some().hash(hasher);
    if let Some(value) = value {
        for component in value.to_cols_array() {
            component.to_bits().hash(hasher);
        }
    }
}

fn hash_reflection_probe_selection<H: Hasher>(item: &WorldMeshDrawItem, hasher: &mut H) {
    item.reflection_probes.atlas_indices.hash(hasher);
    item.reflection_probes.importance_mask.hash(hasher);
}

fn hash_vec4_option<H: Hasher>(value: Option<glam::Vec4>, hasher: &mut H) {
    value.is_some().hash(hasher);
    if let Some(value) = value {
        for component in value.to_array() {
            component.to_bits().hash(hasher);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_mesh::test_fixtures::{DummyDrawItemSpec, dummy_world_mesh_draw_item};

    fn draw(node_id: i32) -> WorldMeshDrawItem {
        dummy_world_mesh_draw_item(DummyDrawItemSpec {
            material_asset_id: 1,
            property_block: None,
            skinned: false,
            sorting_order: 0,
            mesh_asset_id: 1,
            node_id,
            slot_index: 0,
            collect_order: 0,
            alpha_blended: false,
        })
    }

    #[test]
    fn command_cache_reuses_equivalent_arrangements() {
        let cache = WorldMeshCommandCache::default();
        let chunks = vec![vec![draw(2), draw(1)]];

        let (first, _) = cache.arrange_draw_chunks(chunks.clone(), false);
        let (second, _) = cache.arrange_draw_chunks(chunks, false);

        assert_eq!(first.len(), second.len());
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().entries, 1);
    }

    #[test]
    fn command_cache_invalidates_on_structural_change() {
        let cache = WorldMeshCommandCache::default();

        let _ = cache.arrange_draw_chunks(vec![vec![draw(1)]], false);
        let _ = cache.arrange_draw_chunks(vec![vec![draw(2)]], false);

        assert_eq!(cache.stats().hits, 0);
        assert_eq!(cache.stats().misses, 2);
        assert_eq!(cache.stats().entries, 2);
    }
}
