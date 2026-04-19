#![warn(missing_docs)]
#![allow(clippy::module_inception)]

//! Shared IPC types, packing helpers, and queue + shared-memory primitives used by both the
//! Renderide renderer (`renderide`) and any host-side Rust tooling that talks to it (currently
//! `renderide-test`; potentially future debug CLIs and replay tools).
//!
//! This crate is the third "shared functionality" workspace member alongside [`interprocess`] and
//! [`logger`]: anything that crosses the host/renderer process boundary lives here so neither side
//! has to depend on the other's heavy dependencies (wgpu, naga, OpenXR, winit, imgui).
//!
//! # Module map
//!
//! - [`shared`] — generated Renderite shared structs and enums (emitted by the workspace
//!   `SharedTypeGenerator` tool). Host and renderer agree on this byte layout.
//! - [`packing`] — `MemoryPacker` / `MemoryUnpacker` and supporting traits implementing the
//!   binary contract.
//! - [`buffer`] — [`buffer::SharedMemoryBufferDescriptor`] for shared-memory regions.
//! - [`shader_upload_extras`] — optional trailing payload fields on `ShaderUpload`.
//! - [`ipc`] — Cloudtoid queue helpers, the renderer-side dual-queue wrapper
//!   ([`ipc::DualQueueIpc`]), and the read-only [`ipc::SharedMemoryAccessor`].

pub mod buffer;
pub mod ipc;
pub mod packing;
pub mod shader_upload_extras;
pub mod shared_memory_writer;
pub mod wire_writer;

/// Automatically generated Renderite shared types and decode helpers.
pub mod shared;

pub use shared_memory_writer::{
    SharedMemoryWriter, SharedMemoryWriterConfig, SharedMemoryWriterError,
};

pub use packing::polymorphic_decode_error::PolymorphicDecodeError;
pub use packing::wire_decode_error::WireDecodeError;
pub use packing::{
    default_entity_pool, enum_repr, memory_packable, memory_packer, memory_packer_entity_pool,
    memory_unpack_error, memory_unpacker, packed_bools, polymorphic_decode_error,
    polymorphic_memory_packable_entity, wire_decode_error,
};
pub use shared::*;
