//! Shared IPC types, packing helpers, and queue + shared-memory primitives used by both the
//! Renderide renderer (`renderide`) and any host-side Rust tooling that talks to it (currently
//! `renderide-test`; potentially future debug CLIs and replay tools).
//!
//! This crate is the third "shared functionality" workspace member alongside [`interprocess`] and
//! [`logger`]: anything that crosses the host/renderer process boundary lives here so neither side
//! has to depend on the other's heavy dependencies (wgpu, naga, `OpenXR`, winit, imgui).
//!
//! # Module map
//!
//! - [`shared`] -- generated Renderite shared structs and enums (emitted by the workspace
//!   `SharedTypeGenerator` tool). Host and renderer agree on this byte layout.
//! - [`packing`] -- `MemoryPacker` / `MemoryUnpacker` and supporting traits implementing the
//!   binary contract, plus hand-rolled [`packing::extras`] impls for generated types whose layout
//!   the auto-classifier cannot derive.
//! - [`buffer`] -- [`buffer::SharedMemoryBufferDescriptor`] for shared-memory regions.
//! - [`ipc`] -- Cloudtoid queue helpers, the renderer-side dual-queue wrapper
//!   ([`ipc::DualQueueIpc`]), the host-side wrapper ([`ipc::HostDualQueueIpc`]), the read-only
//!   [`ipc::SharedMemoryAccessor`], and its host-side [`ipc::shared_memory::SharedMemoryWriter`]
//!   counterpart.

pub mod buffer;
pub mod ipc;
pub mod packing;
pub mod wire_writer;

/// Automatically generated Renderite shared types and decode helpers.
pub mod shared;

pub use ipc::shared_memory::{
    SharedMemoryWriter, SharedMemoryWriterConfig, SharedMemoryWriterError,
};

// The `polymorphic_decode_error` and `wire_decode_error` types AND their containing modules are
// re-exported at crate root. Both spellings are load-bearing: the renderer's
// `renderide/src/shared.rs` has a compile-time guard that resolves both
// `renderide_shared::PolymorphicDecodeError` (the type) and
// `renderide_shared::polymorphic_decode_error::PolymorphicDecodeError` (via the module path) so
// downstream consumers can pick either spelling. Keep both forms in sync.
pub use packing::polymorphic_decode_error::PolymorphicDecodeError;
pub use packing::wire_decode_error::WireDecodeError;
pub use packing::{
    bit_span, default_entity_pool, enum_repr, memory_packable, memory_packer,
    memory_packer_entity_pool, memory_unpack_error, memory_unpacker, packed_bools,
    polymorphic_decode_error, polymorphic_memory_packable_entity, wire_decode_error,
};
pub use shared::*;
