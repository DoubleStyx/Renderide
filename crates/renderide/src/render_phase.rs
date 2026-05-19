//! Per-view render phase queues shared by mesh pass planners.

use std::marker::PhantomData;

/// Stable key used to index a [`RenderPhaseSet`].
pub(crate) trait RenderPhaseKey: Copy + Eq {
    /// Number of phases represented by this key type.
    const COUNT: usize;

    /// Dense zero-based index for this phase key.
    fn index(self) -> usize;
}

/// One ordered queue of render items for a view-local phase.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct RenderPhase<T> {
    /// Queued phase items in submission order.
    items: Vec<T>,
}

impl<T> RenderPhase<T> {
    /// Creates an empty render phase.
    pub(crate) fn new() -> Self {
        Self { items: Vec::new() }
    }

    /// Appends one item to the phase.
    pub(crate) fn push(&mut self, item: T) {
        self.items.push(item);
    }

    /// Returns the queued items.
    pub(crate) fn items(&self) -> &[T] {
        &self.items
    }

    /// Returns the queued items mutably.
    pub(crate) fn items_mut(&mut self) -> &mut Vec<T> {
        &mut self.items
    }

    /// Returns the number of queued items.
    pub(crate) fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns whether the phase has no queued items.
    pub(crate) fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

/// Dense set of keyed render phases for one view.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RenderPhaseSet<K, T>
where
    K: RenderPhaseKey,
{
    /// Phase queues indexed by [`RenderPhaseKey::index`].
    phases: Vec<RenderPhase<T>>,
    /// Retains the key type without storing values.
    key: PhantomData<K>,
}

impl<K, T> RenderPhaseSet<K, T>
where
    K: RenderPhaseKey,
{
    /// Creates an empty set containing one queue for every key value.
    pub(crate) fn new() -> Self {
        Self {
            phases: std::iter::repeat_with(RenderPhase::new)
                .take(K::COUNT)
                .collect(),
            key: PhantomData,
        }
    }

    /// Returns the queue for `key`.
    pub(crate) fn phase(&self, key: K) -> &RenderPhase<T> {
        &self.phases[key.index()]
    }

    /// Returns the queue for `key` mutably.
    pub(crate) fn phase_mut(&mut self, key: K) -> &mut RenderPhase<T> {
        &mut self.phases[key.index()]
    }
}

impl<K, T> Default for RenderPhaseSet<K, T>
where
    K: RenderPhaseKey,
{
    fn default() -> Self {
        Self::new()
    }
}
