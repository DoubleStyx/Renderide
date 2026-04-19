//! Trait for application types that serialize through [`MemoryPacker`](super::memory_packer::MemoryPacker).

use super::memory_packer::MemoryPacker;
use super::memory_packer_entity_pool::MemoryPackerEntityPool;
use super::memory_unpacker::MemoryUnpacker;
use super::wire_decode_error::WireDecodeError;

/// Type that can be written to and read from IPC byte buffers using the shared wire format.
pub trait MemoryPackable {
    /// Serializes `self` into `packer`.
    fn pack(&mut self, packer: &mut MemoryPacker<'_>);

    /// Deserializes into `self` from `unpacker`, using `pool` when new owned structs are needed.
    fn unpack<P: MemoryPackerEntityPool>(
        &mut self,
        unpacker: &mut MemoryUnpacker<'_, '_, P>,
    ) -> Result<(), WireDecodeError>;
}
