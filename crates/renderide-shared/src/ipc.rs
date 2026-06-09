//! Host-renderer IPC: Cloudtoid ring-buffer queues and memory-mapped large payloads.
//!
//! Layout: [`connection`] (CLI `-QueueName` / `-QueueCapacity`, queue naming);
//! [`dual_queue`] ([`DualQueueIpc`], renderer side); [`host_dual_queue`] ([`HostDualQueueIpc`],
//! authority side); [`shared_memory`] ([`SharedMemoryAccessor`] plus `bounds`, `naming`,
//! diagnostics, and platform `SharedMemoryView` modules).
//!
//! Private siblings `dual_queue_shared` (encode/drain/queue-open primitives) and
//! `dual_queue_reliable_outbox` (renderer-side reliable-background FIFO) back both wrappers
//! without exposing any cross-crate API.

pub mod connection;
pub mod dual_queue;
pub mod host_dual_queue;
pub mod shared_memory;

mod dual_queue_reliable_outbox;
mod dual_queue_shared;

pub use dual_queue::{DualQueueIpc, TimedRendererCommand};
pub use host_dual_queue::HostDualQueueIpc;
pub use shared_memory::{
    InvalidSharedMemoryPrefix, RENDERIDE_INTERPROCESS_DIR_ENV, SharedMemoryAccessor,
    compose_memory_view_name,
};
