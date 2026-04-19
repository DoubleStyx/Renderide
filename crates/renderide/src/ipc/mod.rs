//! Host-renderer IPC: re-exports of [`renderide_shared::ipc`] plus renderer-only headless config.
//!
//! The Cloudtoid queue layout, command encoding, and shared-memory accessor live in
//! [`renderide_shared::ipc`]; this module preserves the existing `crate::ipc::*` paths and adds
//! [`headless_config`] (renderer-process CLI parsing).

pub use renderide_shared::ipc::connection;
pub use renderide_shared::ipc::dual_queue;
pub use renderide_shared::ipc::shared_memory;

pub mod headless_config;

pub use renderide_shared::ipc::dual_queue::DualQueueIpc;
pub use renderide_shared::ipc::shared_memory::{
    compose_memory_view_name, SharedMemoryAccessor, RENDERIDE_INTERPROCESS_DIR_ENV,
};

pub use headless_config::{get_headless_params, HeadlessParams};
