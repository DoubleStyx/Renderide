//! Shared cache counters and snapshots.

use std::sync::atomic::{AtomicU64, Ordering};

/// Point-in-time cache counter snapshot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CacheStats {
    /// Cache hits.
    pub(crate) hits: u64,
    /// Cache misses.
    pub(crate) misses: u64,
    /// Cache insertions.
    pub(crate) insertions: u64,
    /// Cache evictions.
    pub(crate) evictions: u64,
}

impl CacheStats {
    /// Returns the saturating delta from `previous` to `self`.
    #[inline]
    pub(crate) fn delta_since(self, previous: Self) -> CacheStatsDelta {
        CacheStatsDelta {
            hits: self.hits.saturating_sub(previous.hits),
            misses: self.misses.saturating_sub(previous.misses),
            insertions: self.insertions.saturating_sub(previous.insertions),
            evictions: self.evictions.saturating_sub(previous.evictions),
        }
    }
}

/// Saturating cache counter delta.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CacheStatsDelta {
    /// Hit delta.
    pub(crate) hits: u64,
    /// Miss delta.
    pub(crate) misses: u64,
    /// Insertion delta.
    pub(crate) insertions: u64,
    /// Eviction delta.
    pub(crate) evictions: u64,
}

/// Non-atomic cache counters for single-threaded or externally synchronized caches.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CacheCounters {
    stats: CacheStats,
}

impl CacheCounters {
    /// Records a cache hit.
    #[inline]
    pub(crate) fn note_hit(&mut self) {
        self.stats.hits = self.stats.hits.saturating_add(1);
    }

    /// Records a cache miss.
    #[inline]
    pub(crate) fn note_miss(&mut self) {
        self.stats.misses = self.stats.misses.saturating_add(1);
    }

    /// Records a cache insertion.
    #[inline]
    pub(crate) fn note_insertion(&mut self) {
        self.stats.insertions = self.stats.insertions.saturating_add(1);
    }

    /// Records a cache eviction.
    #[inline]
    pub(crate) fn note_eviction(&mut self) {
        self.stats.evictions = self.stats.evictions.saturating_add(1);
    }

    /// Returns a point-in-time snapshot.
    #[inline]
    pub(crate) fn snapshot(&self) -> CacheStats {
        self.stats
    }
}

/// Atomic cache counters for caches shared across worker threads.
#[derive(Debug, Default)]
pub(crate) struct AtomicCacheCounters {
    hits: AtomicU64,
    misses: AtomicU64,
    insertions: AtomicU64,
    evictions: AtomicU64,
}

impl AtomicCacheCounters {
    /// Records a cache hit.
    #[inline]
    pub(crate) fn note_hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a cache miss.
    #[inline]
    pub(crate) fn note_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a cache insertion.
    #[inline]
    pub(crate) fn note_insertion(&self) {
        self.insertions.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a cache eviction.
    #[inline]
    pub(crate) fn note_eviction(&self) {
        self.evictions.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns a point-in-time snapshot.
    #[inline]
    pub(crate) fn snapshot(&self) -> CacheStats {
        CacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            insertions: self.insertions.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{AtomicCacheCounters, CacheCounters, CacheStats};

    #[test]
    fn non_atomic_counters_track_snapshot() {
        let mut counters = CacheCounters::default();

        counters.note_hit();
        counters.note_miss();
        counters.note_insertion();
        counters.note_eviction();

        assert_eq!(
            counters.snapshot(),
            CacheStats {
                hits: 1,
                misses: 1,
                insertions: 1,
                evictions: 1,
            }
        );
    }

    #[test]
    fn atomic_counters_track_snapshot() {
        let counters = AtomicCacheCounters::default();

        counters.note_hit();
        counters.note_hit();
        counters.note_miss();
        counters.note_insertion();
        counters.note_eviction();

        assert_eq!(
            counters.snapshot(),
            CacheStats {
                hits: 2,
                misses: 1,
                insertions: 1,
                evictions: 1,
            }
        );
    }

    #[test]
    fn stats_delta_saturates() {
        let after = CacheStats {
            hits: 10,
            misses: 1,
            insertions: 5,
            evictions: 0,
        };
        let before = CacheStats {
            hits: 7,
            misses: 3,
            insertions: 2,
            evictions: 4,
        };

        let delta = after.delta_since(before);

        assert_eq!(delta.hits, 3);
        assert_eq!(delta.misses, 0);
        assert_eq!(delta.insertions, 3);
        assert_eq!(delta.evictions, 0);
    }
}
