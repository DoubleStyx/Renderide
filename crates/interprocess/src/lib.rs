#![warn(missing_docs)]

//! Cloudtoid-compatible shared-memory queue for IPC between processes.
//!
//! ```text
//! Host process                Renderer / bootstrapper
//!     |  Publisher / Subscriber (same .qu + semaphore names)
//!     v
//! [ QueueHeader | ring buffer ]  <-- mmap / section
//! ```
//!
//! - **Unix**: file-backed read/write mapping under the configured directory. The portable default
//!   is [`default_memory_dir`]: `/dev/shm/.cloudtoid/...` on Linux (tmpfs), and a
//!   `.cloudtoid/interprocess/mmf` folder under [`std::env::temp_dir`] on macOS and other non-Linux Unix.
//! - **Windows**: named file mapping `CT_IP_{queue}` plus `Global\CT.IP.{queue}` semaphore; the default
//!   [`QueueOptions::path`] uses the same temp-dir subfolder as other platforms for consistency (the mapping does not read from disk).
//!
//! Binary layout details live in [`layout`]. Typical usage: build [`QueueOptions`], then
//! [`Publisher::new`] / [`Subscriber::new`] or [`QueueFactory`].

#[cfg(not(any(unix, windows)))]
compile_error!("The `interprocess` crate only supports `cfg(unix)` and `cfg(windows)` targets.");

mod error;
pub mod layout;
mod memory;
#[cfg(windows)]
mod naming;
mod options;
mod publisher;
mod queue;
mod queue_resources;
mod ring;
mod semaphore;
mod subscriber;

pub use error::OpenError;
pub use options::{
    default_memory_dir, QueueOptions, LINUX_SHM_MEMORY_DIR, RENDERIDE_INTERPROCESS_DIR_ENV,
};
pub use publisher::Publisher;
pub use queue::QueueFactory;
pub use subscriber::Subscriber;
