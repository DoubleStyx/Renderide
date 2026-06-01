//! Retained CPU command-list cache for arranged world-mesh draw collections.

use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use hashbrown::HashMap;
use parking_lot::Mutex;

use crate::cpu_parallelism::RENDER_COMMAND_CHUNK_DRAWS;

use super::arrange::arrange_draw_chunks_by_phase_bins;
use super::item::{WorldMeshDrawArrangementStats, WorldMeshDrawItem};

const WORLD_MESH_COMMAND_CACHE_CAPACITY: usize = 256;
const WORLD_MESH_COMMAND_CACHE_MIN_DRAWS: usize = RENDER_COMMAND_CHUNK_DRAWS * 2;
const WORLD_MESH_COMMAND_CACHE_THRASH_WINDOW_LOOKUPS: u32 = 16;
const WORLD_MESH_COMMAND_CACHE_THRASH_MIN_HIT_RATE_PER_MILLE: u32 = 250;
const WORLD_MESH_COMMAND_CACHE_THRASH_BYPASS_LOOKUPS: u32 = 16;

/// Runtime counters for the retained world-mesh command cache.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorldMeshCommandCacheStats {
    /// Retained arranged command lists currently resident in the cache.
    pub entries: usize,
    /// Cache lookups that reused an arranged draw list.
    pub hits: u64,
    /// Cache lookups that had to rebuild the arranged draw list.
    pub misses: u64,
    /// Eligible cache attempts skipped because the draw count was too small to justify hashing.
    pub skipped_small: u64,
    /// Eligible cache attempts skipped while recent probes were missing too often.
    pub skipped_thrash: u64,
    /// Hit rate for cache probes, in hits per 1000 lookups.
    pub hit_rate_per_mille: u16,
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
    thrash: CacheThrashWindow,
}

#[derive(Debug, Default)]
struct CacheThrashWindow {
    lookups: u32,
    hits: u32,
    bypass_remaining: u32,
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
        let draw_count = command_cache_draw_count(&chunks);
        if !Self::admits_draw_count(draw_count) {
            self.record_skipped_small();
            return arrange_draw_chunks_by_phase_bins(chunks, allow_parallel_sort);
        }
        if !self.should_probe_cache() {
            return arrange_draw_chunks_by_phase_bins(chunks, allow_parallel_sort);
        }
        let key = {
            profiling::scope!("mesh::arrange_command_cache::fingerprint");
            WorldMeshCommandCacheKey::from_chunks(&chunks, draw_count)
        };
        if let Some(entry) = self.entry(&key) {
            profiling::scope!("mesh::arrange_command_cache::clone_hit");
            return (entry.items.as_ref().to_vec(), entry.arrangement);
        }

        let (items, arrangement) = {
            profiling::scope!("mesh::arrange_command_cache::miss_arrange");
            arrange_draw_chunks_by_phase_bins(chunks, allow_parallel_sort)
        };
        self.insert(key, &items, arrangement);
        (items, arrangement)
    }

    /// Captures a point-in-time diagnostic snapshot of the command cache.
    pub(crate) fn stats(&self) -> WorldMeshCommandCacheStats {
        let inner = self.inner.lock();
        let mut stats = inner.stats;
        stats.entries = inner.entries.len();
        drop(inner);
        stats.hit_rate_per_mille = cache_hit_rate_per_mille(stats.hits, stats.misses);
        stats
    }

    fn admits_draw_count(draw_count: usize) -> bool {
        draw_count >= WORLD_MESH_COMMAND_CACHE_MIN_DRAWS
    }

    fn record_skipped_small(&self) {
        let mut inner = self.inner.lock();
        inner.stats.skipped_small = inner.stats.skipped_small.saturating_add(1);
    }

    fn should_probe_cache(&self) -> bool {
        profiling::scope!("mesh::arrange_command_cache::admit");
        let mut inner = self.inner.lock();
        if inner.thrash.bypass_remaining == 0 {
            return true;
        }
        inner.thrash.bypass_remaining = inner.thrash.bypass_remaining.saturating_sub(1);
        inner.stats.skipped_thrash = inner.stats.skipped_thrash.saturating_add(1);
        false
    }

    fn entry(&self, key: &WorldMeshCommandCacheKey) -> Option<WorldMeshCommandCacheEntry> {
        let mut inner = self.inner.lock();
        let entry = inner.entries.get(key).cloned();
        if entry.is_some() {
            inner.stats.hits = inner.stats.hits.saturating_add(1);
            inner.recency.push_back(*key);
            inner.thrash.record_hit();
        } else {
            inner.stats.misses = inner.stats.misses.saturating_add(1);
            inner.thrash.record_miss();
        }
        if inner.thrash.should_enter_bypass() {
            inner.thrash.bypass_remaining = WORLD_MESH_COMMAND_CACHE_THRASH_BYPASS_LOOKUPS;
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
    fn from_chunks(chunks: &[Vec<WorldMeshDrawItem>], draw_count: usize) -> Self {
        Self {
            fingerprint: fingerprint_world_mesh_draw_chunks(chunks),
            draw_count,
            chunk_count: chunks.len(),
        }
    }
}

impl CacheThrashWindow {
    fn record_hit(&mut self) {
        self.lookups = self.lookups.saturating_add(1);
        self.hits = self.hits.saturating_add(1);
    }

    fn record_miss(&mut self) {
        self.lookups = self.lookups.saturating_add(1);
    }

    fn should_enter_bypass(&mut self) -> bool {
        if self.lookups < WORLD_MESH_COMMAND_CACHE_THRASH_WINDOW_LOOKUPS {
            return false;
        }
        let hits = self.hits as u64;
        let misses = self.lookups.saturating_sub(self.hits) as u64;
        let should_bypass = cache_hit_rate_per_mille(hits, misses)
            < WORLD_MESH_COMMAND_CACHE_THRASH_MIN_HIT_RATE_PER_MILLE as u16;
        self.lookups = 0;
        self.hits = 0;
        should_bypass
    }
}

fn command_cache_draw_count(chunks: &[Vec<WorldMeshDrawItem>]) -> usize {
    chunks.iter().map(Vec::len).sum()
}

fn cache_hit_rate_per_mille(hits: u64, misses: u64) -> u16 {
    let lookups = hits.saturating_add(misses);
    if lookups == 0 {
        return 0;
    }
    ((hits.saturating_mul(1000)) / lookups).min(1000) as u16
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

    fn large_chunk(seed: i32) -> Vec<Vec<WorldMeshDrawItem>> {
        let draws = (0..WORLD_MESH_COMMAND_CACHE_MIN_DRAWS)
            .map(|index| draw(seed + index as i32))
            .collect();
        vec![draws]
    }

    #[test]
    fn command_cache_reuses_equivalent_arrangements() {
        let cache = WorldMeshCommandCache::default();
        let chunks = large_chunk(1);

        let (first, _) = cache.arrange_draw_chunks(chunks.clone(), false);
        let (second, _) = cache.arrange_draw_chunks(chunks, false);

        assert_eq!(first.len(), second.len());
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().entries, 1);
        assert_eq!(cache.stats().hit_rate_per_mille, 500);
    }

    #[test]
    fn command_cache_invalidates_on_structural_change() {
        let cache = WorldMeshCommandCache::default();

        let _ = cache.arrange_draw_chunks(large_chunk(1), false);
        let _ = cache.arrange_draw_chunks(large_chunk(10_000), false);

        assert_eq!(cache.stats().hits, 0);
        assert_eq!(cache.stats().misses, 2);
        assert_eq!(cache.stats().entries, 2);
    }

    #[test]
    fn command_cache_bypasses_small_draw_lists() {
        let cache = WorldMeshCommandCache::default();

        let _ = cache.arrange_draw_chunks(vec![vec![draw(1)]], false);

        let stats = cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.entries, 0);
        assert_eq!(stats.skipped_small, 1);
    }

    #[test]
    fn command_cache_temporarily_bypasses_after_repeated_misses() {
        let cache = WorldMeshCommandCache::default();

        for seed in 0..WORLD_MESH_COMMAND_CACHE_THRASH_WINDOW_LOOKUPS {
            let _ = cache.arrange_draw_chunks(large_chunk((seed as i32) * 10_000), false);
        }
        let _ = cache.arrange_draw_chunks(large_chunk(900_000), false);

        let stats = cache.stats();
        assert_eq!(
            stats.misses,
            WORLD_MESH_COMMAND_CACHE_THRASH_WINDOW_LOOKUPS as u64
        );
        assert_eq!(stats.skipped_thrash, 1);
    }
}
