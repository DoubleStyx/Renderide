//! Shared types and utilities for IPC between the host and renderer.
//!
//! Includes memory packing/unpacking (mirrors Renderite.Shared), shared memory buffer
//! descriptors, and the shared memory accessor for mmap-based IPC.

pub mod buffer;
pub mod packing;
pub mod shared;
pub mod shared_memory;

pub use packing::{
    default_entity_pool,
    enum_repr,
    memory_packable,
    memory_packer,
    memory_packer_entity_pool,
    memory_unpacker,
    packed_bools,
    polymorphic_memory_packable_entity,
};

/// Re-export shared types so consumers can use `crate::shared::Type` instead of `crate::shared::shared::Type`.
pub use shared::*;
