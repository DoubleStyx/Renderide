//! Memory-oriented serialization for host–renderer IPC.
//!
//! Primitives, strings, and collection encodings match the layout produced by the host’s
//! `MemoryPacker` / `MemoryUnpacker` types so a single binary contract works on all platforms.

pub mod default_entity_pool;
pub mod enum_repr;
pub mod memory_packable;
pub mod memory_packer;
pub mod memory_packer_entity_pool;
pub mod memory_unpack_error;
pub mod memory_unpacker;
pub mod packed_bools;
pub mod polymorphic_decode_error;
pub mod polymorphic_memory_packable_entity;
pub mod wire_decode_error;

#[cfg(test)]
mod wire_primitive_tests;
