//! Default implementation of `MemoryPackerEntityPool` that creates new instances via `Default`
//! and discards returned values. Use when object pooling is not needed.

use super::memory_packable::MemoryPackable;
use super::memory_packer_entity_pool::MemoryPackerEntityPool;

/// A pool that creates new instances via `Default` and ignores returns.
/// Use when object pooling is not needed (e.g. renderer-side unpacking).
#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultEntityPool;

impl MemoryPackerEntityPool for DefaultEntityPool {
    fn borrow<T: MemoryPackable + Default>(&mut self) -> T {
        T::default()
    }

    fn r#return<T: MemoryPackable + Default>(&mut self, _value: &mut T) {
        // Discard; no pooling
    }
}
