//! Borrow/return abstraction for unpacking lists of heap-backed structs on the host.

use super::memory_packable::MemoryPackable;

/// Entity pool used while unpacking optional objects and object lists.
///
/// The `FrooxEngine` host may reuse instances from a pool; the renderer typically uses [`DefaultEntityPool`](super::default_entity_pool::DefaultEntityPool).
pub trait MemoryPackerEntityPool {
    /// Obtains a value to deserialize into (new or from a pool).
    fn borrow<T: MemoryPackable + Default>(&mut self) -> T;

    /// Returns `value` to the pool when a shorter list replaces a longer one (host-side pattern).
    fn r#return<T: MemoryPackable + Default>(&mut self, value: &mut T);
}
