use super::memory_packable::MemoryPackable;

/// Trait for pools that borrow and return `MemoryPackable` instances during unpacking.
/// Mirrors C# `Renderite.Shared.IMemoryPackerEntityPool`.
pub trait MemoryPackerEntityPool {
    /// Borrows a new or pooled instance of `T`.
    fn borrow<T: MemoryPackable + Default>(&mut self) -> T;

    /// Returns an instance to the pool for reuse.
    fn r#return<T: MemoryPackable + Default>(&mut self, value: &mut T);
}
