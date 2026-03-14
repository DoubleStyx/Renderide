use super::memory_packer::MemoryPacker;

/// Trait for types that encode themselves with a type index for polymorphic serialization.
/// Equivalent to C# `PolymorphicMemoryPackableEntity.Encode`—writes a type discriminator
/// before packing so the decoder can dispatch to the correct concrete type.
pub trait PolymorphicEncode {
    /// Encodes this value into the packer, including any type discriminator.
    fn encode(&mut self, packer: &mut MemoryPacker<'_>);
}
