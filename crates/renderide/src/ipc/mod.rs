//! Inter-process command queues (Primary + Background) compatible with the managed host.

mod dual_queue;
mod shared_memory;

pub use dual_queue::DualQueueIpc;
pub use shared_memory::SharedMemoryAccessor;
