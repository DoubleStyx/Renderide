//! Shared types and utilities for IPC between the host and renderer.
//!
//! Includes memory packing/unpacking and shared memory buffer
//! descriptors. The shared memory accessor lives in ipc::shared_memory.

#![allow(clippy::module_inception)]

pub mod buffer;
pub mod packing;
pub mod shader_upload_extras;
pub mod shared;

pub use shader_upload_extras::unpack_appended_shader_logical_name;

pub use packing::{
    default_entity_pool, enum_repr, memory_packable, memory_packer, memory_packer_entity_pool,
    memory_unpacker, polymorphic_memory_packable_entity,
};

/// Re-export shared types so consumers can use `crate::shared::Type` instead of `crate::shared::shared::Type`.
pub use shared::*;
