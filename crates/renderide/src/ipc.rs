//! Host-renderer IPC: re-exports of [`renderide_shared::ipc`] plus renderer-only headless config.
//!
//! The Cloudtoid queue layout, command encoding, and shared-memory accessor live in
//! [`renderide_shared::ipc`]; this module preserves the existing `crate::ipc::*` paths and adds
//! [`headless_config`] (renderer-process CLI parsing).

pub use renderide_shared::ipc::connection;

pub mod headless_config;

pub use renderide_shared::ipc::dual_queue::{DualQueueIpc, TimedRendererCommand};
pub use renderide_shared::ipc::shared_memory::SharedMemoryAccessor;

pub use headless_config::{HeadlessParams, get_headless_params, get_ignore_config};
