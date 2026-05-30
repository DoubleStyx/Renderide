//! Hash-sharded LRU cache for parallel render-resource lookups.

use std::hash::Hash;
use std::num::NonZeroUsize;

use ahash::RandomState;
use lru::LruCache;
use parking_lot::Mutex;

/// Hash-sharded LRU cache.
///
/// Each shard is an independent `Mutex<LruCache<K, V>>`; lookups and inserts route by hash so
/// distinct keys usually avoid each other's locks. The total capacity is split evenly across
/// shards, with a minimum of one slot per shard so the LRU type's invariant is preserved.
pub(crate) struct ShardedLru<K, V> {
    shards: Box<[Mutex<LruCache<K, V>>]>,
    hasher: RandomState,
    mask: usize,
}

impl<K: Eq + Hash, V> ShardedLru<K, V> {
    /// Builds an `n_shards`-way sharded LRU with `total_cap` total capacity.
    ///
    /// `n_shards` must be a power of two so the modulo collapses to a bitmask. Total capacity
    /// rounds up so the sharded total is at least the requested cap.
    pub(crate) fn new(total_cap: NonZeroUsize, n_shards: usize) -> Self {
        debug_assert!(
            n_shards.is_power_of_two(),
            "n_shards must be a power of two for the bitmask routing"
        );
        let per_shard = total_cap.get().div_ceil(n_shards).max(1);
        let per_shard_nz = NonZeroUsize::new(per_shard).unwrap_or(NonZeroUsize::MIN);
        let shards: Box<[Mutex<LruCache<K, V>>]> = (0..n_shards)
            .map(|_| Mutex::new(LruCache::new(per_shard_nz)))
            .collect();
        Self {
            shards,
            hasher: RandomState::new(),
            mask: n_shards - 1,
        }
    }

    #[inline]
    fn shard_index(&self, key: &K) -> usize {
        (self.hasher.hash_one(key) as usize) & self.mask
    }

    /// LRU lookup that promotes the entry to most-recently-used and returns a clone of the value.
    pub(crate) fn get_cloned(&self, key: &K) -> Option<V>
    where
        V: Clone,
    {
        let mut shard = self.shards[self.shard_index(key)].lock();
        shard.get(key).cloned()
    }

    /// Inserts `value` for `key`, returning a replaced or capacity-evicted value.
    pub(crate) fn put(&self, key: K, value: V) -> Option<V> {
        let mut shard = self.shards[self.shard_index(&key)].lock();
        shard.push(key, value).map(|(_, evicted)| evicted)
    }

    /// Removes every entry from every shard and returns the number of dropped entries.
    pub(crate) fn clear(&self) -> usize {
        let mut dropped = 0usize;
        for shard in &self.shards {
            let mut shard = shard.lock();
            dropped = dropped.saturating_add(shard.len());
            shard.clear();
        }
        dropped
    }

    /// Returns the total number of entries currently retained across all shards.
    pub(crate) fn len(&self) -> usize {
        self.shards.iter().map(|shard| shard.lock().len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::ShardedLru;

    #[test]
    fn get_cloned_promotes_and_returns_value() {
        let cache = ShardedLru::<u32, u32>::new(NonZeroUsize::MIN, 1);

        assert_eq!(cache.get_cloned(&7), None);
        assert_eq!(cache.put(7, 70), None);
        assert_eq!(cache.get_cloned(&7), Some(70));
    }

    #[test]
    fn put_returns_lru_eviction() {
        let cache = ShardedLru::<u32, u32>::new(NonZeroUsize::new(2).unwrap(), 1);

        assert_eq!(cache.put(1, 10), None);
        assert_eq!(cache.put(2, 20), None);
        assert_eq!(cache.get_cloned(&1), Some(10));

        assert_eq!(cache.put(3, 30), Some(20));
        assert_eq!(cache.get_cloned(&1), Some(10));
        assert_eq!(cache.get_cloned(&2), None);
        assert_eq!(cache.get_cloned(&3), Some(30));
    }

    #[test]
    fn clear_drops_all_shards() {
        let cache = ShardedLru::<u32, u32>::new(NonZeroUsize::new(4).unwrap(), 2);

        assert_eq!(cache.put(1, 10), None);
        assert_eq!(cache.put(2, 20), None);
        assert_eq!(cache.clear(), 2);
        assert_eq!(cache.get_cloned(&1), None);
        assert_eq!(cache.get_cloned(&2), None);
        assert_eq!(cache.clear(), 0);
    }

    #[test]
    fn len_counts_all_shards() {
        let cache = ShardedLru::<u32, u32>::new(NonZeroUsize::new(4).unwrap(), 2);

        assert_eq!(cache.len(), 0);
        assert_eq!(cache.put(1, 10), None);
        assert_eq!(cache.put(2, 20), None);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.clear(), 2);
        assert_eq!(cache.len(), 0);
    }
}
