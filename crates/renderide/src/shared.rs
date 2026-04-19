//! Re-export of [`renderide_shared`], the workspace shared types and packing crate.
//!
//! This thin shim preserves `crate::shared::*` paths used throughout the renderer source. New
//! renderer code may import directly from [`renderide_shared`] instead.

pub use renderide_shared::buffer;
pub use renderide_shared::packing;
pub use renderide_shared::shader_upload_extras;

/// Generated Renderite shared types and decode helpers (re-exported from
/// [`renderide_shared::shared`]).
pub use renderide_shared::shared;

pub use renderide_shared::packing::polymorphic_decode_error::PolymorphicDecodeError;
pub use renderide_shared::packing::wire_decode_error::WireDecodeError;
pub use renderide_shared::packing::{
    default_entity_pool, enum_repr, memory_packable, memory_packer, memory_packer_entity_pool,
    memory_unpack_error, memory_unpacker, packed_bools, polymorphic_decode_error,
    polymorphic_memory_packable_entity, wire_decode_error,
};
pub use renderide_shared::shared::*;
