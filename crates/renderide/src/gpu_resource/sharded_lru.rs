//! Hash-sharded LRU cache for parallel render-resource lookups.

use std::hash::Hash;
use std::num::NonZeroUsize;

use ahash::RandomState;
use lru::LruCache;
use parking_lot::RwLock;

/// Hash-sharded LRU cache.
///
/// Each shard is an independent `RwLock<LruCache<K, V>>`; lookups and inserts route by hash so
/// distinct keys usually avoid each other's locks and cache hits can run concurrently. The total
/// capacity is split evenly across shards, with a minimum of one slot per shard.
pub(crate) struct ShardedLru<K, V> {
    shards: Box<[RwLock<LruCache<K, V>>]>,
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
        let shards: Box<[RwLock<LruCache<K, V>>]> = (0..n_shards)
            .map(|_| RwLock::new(LruCache::new(per_shard_nz)))
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

    /// Returns a cloned cache hit and promotes it when the shard is not write-contended.
    pub(crate) fn get_cloned(&self, key: &K) -> Option<V>
    where
        V: Clone,
    {
        let shard = &self.shards[self.shard_index(key)];
        let value = shard.read().peek(key).cloned()?;

        // Recency affects eviction quality, not lookup correctness. Promote when the shard is
        // uncontended without turning parallel cache hits back into exclusive-lock waiters.
        if let Some(mut writable) = shard.try_write() {
            let _ = writable.get(key);
        }
        Some(value)
    }

    /// Inserts `value` for `key`, returning a replaced or capacity-evicted value.
    pub(crate) fn put(&self, key: K, value: V) -> Option<V> {
        let mut shard = self.shards[self.shard_index(&key)].write();
        shard.push(key, value).map(|(_, evicted)| evicted)
    }

    /// Removes every entry from every shard and returns their values.
    pub(crate) fn drain_values(&self) -> Vec<V> {
        let mut values = Vec::new();
        for shard in &self.shards {
            let mut shard = shard.write();
            values.reserve(shard.len());
            while let Some((_key, value)) = shard.pop_lru() {
                values.push(value);
            }
        }
        values
    }

    /// Returns the total number of entries currently retained across all shards.
    pub(crate) fn len(&self) -> usize {
        self.shards.iter().map(|shard| shard.read().len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    use super::ShardedLru;

    #[test]
    fn get_cloned_promotes_and_returns_value() {
        let cache = ShardedLru::<u32, u32>::new(NonZeroUsize::MIN, 1);

        assert_eq!(cache.get_cloned(&7), None);
        assert_eq!(cache.put(7, 70), None);
        assert_eq!(cache.get_cloned(&7), Some(70));
    }

    #[test]
    fn cache_hit_does_not_wait_for_another_reader() {
        let cache = Arc::new(ShardedLru::<u32, u32>::new(NonZeroUsize::MIN, 1));
        assert_eq!(cache.put(7, 70), None);
        let held_reader = cache.shards[0].read();
        let worker_cache = Arc::clone(&cache);
        let (tx, rx) = mpsc::sync_channel(1);
        let worker = std::thread::spawn(move || {
            tx.send(worker_cache.get_cloned(&7)).ok();
        });

        assert_eq!(
            rx.recv_timeout(Duration::from_secs(1)),
            Ok(Some(70)),
            "a cache hit should complete while another reader holds the shard"
        );
        drop(held_reader);
        worker.join().expect("cache-hit worker should finish");
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
    fn drain_values_removes_all_entries() {
        let cache = ShardedLru::<u32, u32>::new(NonZeroUsize::new(4).unwrap(), 2);

        assert_eq!(cache.put(1, 10), None);
        assert_eq!(cache.put(2, 20), None);
        let mut values = cache.drain_values();
        values.sort_unstable();
        assert_eq!(values, [10, 20]);
        assert_eq!(cache.get_cloned(&1), None);
        assert_eq!(cache.get_cloned(&2), None);
        assert!(cache.drain_values().is_empty());
    }

    #[test]
    fn len_counts_all_shards() {
        let cache = ShardedLru::<u32, u32>::new(NonZeroUsize::new(4).unwrap(), 2);

        assert_eq!(cache.len(), 0);
        assert_eq!(cache.put(1, 10), None);
        assert_eq!(cache.put(2, 20), None);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.drain_values().len(), 2);
        assert_eq!(cache.len(), 0);
    }
}
