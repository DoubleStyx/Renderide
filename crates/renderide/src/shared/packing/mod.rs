//! Memory packing and unpacking for IPC serialization.
//!
//! Mirrors Renderite.Shared: MemoryPacker, MemoryUnpacker, IMemoryPackable,
//! IMemoryPackerEntityPool, and related types.

pub mod default_entity_pool;
pub mod enum_repr;
pub mod memory_packable;
pub mod memory_packer;
pub mod memory_packer_entity_pool;
pub mod memory_unpacker;
pub mod packed_bools;
pub mod polymorphic_memory_packable_entity;
