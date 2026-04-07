//! Polymorphic variant encoding: type index followed by concrete payload.

use super::memory_packer::MemoryPacker;

/// Encode side of a sum type: writes the discriminant then the concrete fields.
///
/// Decoding is generated per base type (see generated `decode_*` functions and [`MemoryUnpacker::read_polymorphic_list`](super::memory_unpacker::MemoryUnpacker::read_polymorphic_list)).
pub trait PolymorphicEncode {
    /// Writes the type index and packed fields for this variant.
    fn encode(&mut self, packer: &mut MemoryPacker<'_>);
}
