//! Cloudtoid.Interprocess-compatible queue for IPC with Resonite host.
//! Uses shared memory and POSIX semaphores on Linux.

mod backend;
mod circular_buffer;
mod queue;
mod publisher;
mod subscriber;

pub use queue::{QueueOptions, QueueFactory};
pub use subscriber::Subscriber;
pub use publisher::Publisher;
