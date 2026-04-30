//! Host-renderer IPC: Cloudtoid ring-buffer queues and memory-mapped large payloads.
//!
//! Layout: [`connection`] (CLI `-QueueName` / `-QueueCapacity`, queue naming);
//! [`dual_queue`] ([`DualQueueIpc`]); [`shared_memory`] ([`SharedMemoryAccessor`], plus `bounds` /
//! `naming` helpers and platform `SharedMemoryView` modules).

pub mod connection;
pub mod dual_queue;
pub mod host_dual_queue;
pub mod shared_memory;

mod dual_queue_shared;

pub use dual_queue::DualQueueIpc;
pub use host_dual_queue::HostDualQueueIpc;
pub use shared_memory::{
    compose_memory_view_name, SharedMemoryAccessor, RENDERIDE_INTERPROCESS_DIR_ENV,
};
