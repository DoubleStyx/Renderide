//! Trivial pool that always constructs with [`Default`] and drops on “return”.

use super::memory_packable::MemoryPackable;
use super::memory_packer_entity_pool::MemoryPackerEntityPool;

/// [`MemoryPackerEntityPool`] implementation that allocates fresh values and ignores returns.
///
/// Suitable for renderer-side decoding where reuse is unnecessary.
#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultEntityPool;

impl MemoryPackerEntityPool for DefaultEntityPool {
    fn borrow<T: MemoryPackable + Default>(&mut self) -> T {
        T::default()
    }

    fn r#return<T: MemoryPackable + Default>(&mut self, _value: &mut T) {}
}
