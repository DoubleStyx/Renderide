//! Keyed once-only gates for repeated diagnostic logs.

use hashbrown::HashSet;
use parking_lot::Mutex;

/// Thread-safe first-occurrence gate keyed by a diagnostic identity.
pub(crate) struct KeyedLogOnce<K> {
    /// Set of identities that have already emitted their first log line.
    seen: Mutex<HashSet<K>>,
}

impl<K> KeyedLogOnce<K>
where
    K: Eq + std::hash::Hash,
{
    /// Creates a keyed gate with no recorded identities.
    pub(crate) fn new() -> Self {
        Self {
            seen: Mutex::new(HashSet::new()),
        }
    }

    /// Returns `true` only for the first observation of `key`.
    pub(crate) fn should_log(&self, key: K) -> bool {
        self.seen.lock().insert(key)
    }

    /// Number of distinct keys observed by this gate.
    #[cfg(test)]
    pub(crate) fn distinct_keys(&self) -> usize {
        self.seen.lock().len()
    }
}

impl<K> Default for KeyedLogOnce<K>
where
    K: Eq + std::hash::Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::KeyedLogOnce;

    #[test]
    fn logs_first_occurrence_per_key() {
        let gate = KeyedLogOnce::new();

        assert!(gate.should_log(7));
        assert!(!gate.should_log(7));
        assert!(gate.should_log(8));
        assert_eq!(gate.distinct_keys(), 2);
    }
}
