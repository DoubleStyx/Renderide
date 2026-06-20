//! Small retained GPU-resource cache primitives.
//!
//! These slots cover presentation-side paths where the resource identity is a compact handle set
//! and a full map would add more moving parts than it saves. Multi-entry render graph caches live
//! closer to the graph/pool code.

/// Single-entry cache keyed by the resource handles that a retained object references.
pub(crate) struct SingleResourceSlot<K, V> {
    entry: Option<(K, V)>,
}

impl<K, V> Default for SingleResourceSlot<K, V> {
    fn default() -> Self {
        Self { entry: None }
    }
}

impl<K, V> std::fmt::Debug for SingleResourceSlot<K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SingleResourceSlot")
            .field("entry_present", &self.entry.is_some())
            .finish()
    }
}

impl<K, V> SingleResourceSlot<K, V> {
    /// Creates an empty slot.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Clears the cached entry.
    pub(crate) fn clear(&mut self) {
        self.entry = None;
    }
}

impl<K, V> SingleResourceSlot<K, V>
where
    K: PartialEq,
    V: Clone,
{
    /// Returns the cached value when `key` matches, otherwise builds and stores a replacement.
    pub(crate) fn get_or_build(&mut self, key: K, build: impl FnOnce() -> V) -> V {
        if let Some((cached_key, value)) = self.entry.as_ref()
            && cached_key == &key
        {
            return value.clone();
        }

        let value = build();
        self.entry = Some((key, value.clone()));
        value
    }
}

#[cfg(test)]
mod tests {
    use super::SingleResourceSlot;

    #[test]
    fn matching_key_reuses_cached_value() {
        let mut slot = SingleResourceSlot::new();
        let first = slot.get_or_build(7, || String::from("first"));
        let second = slot.get_or_build(7, || String::from("second"));

        assert_eq!(first, "first");
        assert_eq!(second, "first");
    }

    #[test]
    fn changed_key_rebuilds_cached_value() {
        let mut slot = SingleResourceSlot::new();
        let first = slot.get_or_build(1, || String::from("first"));
        let second = slot.get_or_build(2, || String::from("second"));

        assert_eq!(first, "first");
        assert_eq!(second, "second");
    }
}
