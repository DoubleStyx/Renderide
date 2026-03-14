use super::memory_packer::MemoryPacker;
use super::memory_packer_entity_pool::MemoryPackerEntityPool;
use super::memory_unpacker::MemoryUnpacker;

/// Trait for types that can be packed and unpacked for IPC.
/// Mirrors C# `Renderite.Shared.IMemoryPackable`.
pub trait MemoryPackable {
    /// Packs this value into the packer.
    fn pack(&mut self, packer: &mut MemoryPacker<'_>);

    /// Unpacks this value from the unpacker using the given entity pool.
    fn unpack<P: MemoryPackerEntityPool>(&mut self, unpacker: &mut MemoryUnpacker<'_, '_, P>);
}
