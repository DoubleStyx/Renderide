//! Shared types and binary IPC helpers for communication between the FrooxEngine host and the renderer.
//!
//! The submodule [`shared`] holds generated Renderite shared structs and enums (emitted by the workspace `SharedTypeGenerator` tool).
//! [`packing`] implements the byte layout compatible with the host’s memory packer; [`buffer`] describes regions inside host-supplied shared memory; [`shader_upload_extras`] documents optional trailing payload fields.

#![allow(clippy::module_inception)]

pub mod buffer;
pub mod packing;
pub mod shader_upload_extras;

/// Automatically generated Renderite shared types and decode helpers.
pub mod shared;

pub use packing::{
    default_entity_pool, enum_repr, memory_packable, memory_packer, memory_packer_entity_pool,
    memory_unpacker, packed_bools, polymorphic_memory_packable_entity,
};
pub use shared::*;
