//! Memory-mapped access to host-owned shared buffers (mesh payloads, textures, probe results, etc.).
//!
//! Naming follows Renderite `Helper.ComposeMemoryViewName` (see [`compose_memory_view_name`]).
//!
//! # Layout (Cloudtoid / Renderite interop)
//!
//! - **Windows**: named section `CT_IP_{prefix}_{bufferId:X}` via
//!   [`windows_sys`] file mapping (same prefix as [`interprocess`] on Windows).
//! - **Unix**: file `{composed}.qu` in the MMF directory (see below).
//!
//! # Backing directory on Unix
//!
//! Uses the same directory resolution as the bootstrapper
//! ([`RENDERIDE_INTERPROCESS_DIR_ENV`]): if set to a non-empty path, that directory holds
//! `{name}.qu` files; otherwise [`interprocess::default_memory_dir`] — Linux:
//! `/dev/shm/.cloudtoid/interprocess/mmf`, **macOS and other Unix**: `std::env::temp_dir()` +
//! `.cloudtoid/interprocess/mmf`. This matches the workspace `interprocess` crate and avoids
//! assuming `/dev/shm` exists on non-Linux Unix.
//!
//! Managed Cloudtoid historically used `/dev/shm` for any `PlatformID.Unix`; portable Rust stacks
//! should set [`RENDERIDE_INTERPROCESS_DIR_ENV`] consistently on host and renderer when defaults
//! differ from the host implementation.

mod bounds;
mod naming;

#[cfg(unix)]
mod unix;

#[cfg(windows)]
mod windows;

mod accessor;

pub use accessor::SharedMemoryAccessor;

// Public surface for `crate::ipc::shared_memory::*`; also re-exported at [`crate::ipc`].
pub use naming::{compose_memory_view_name, RENDERIDE_INTERPROCESS_DIR_ENV};
