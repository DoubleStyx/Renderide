#![warn(missing_docs)]

//! Cloudtoid-compatible shared-memory queue for IPC between processes.
//!
//! - **Unix**: file-backed read/write mapping under the configured directory. The portable default
//!   is [`default_memory_dir`]: `/dev/shm/.cloudtoid/...` on Linux (tmpfs), and a
//!   `.cloudtoid/interprocess/mmf` folder under [`std::env::temp_dir`] on macOS and other non-Linux Unix.
//! - **Windows**: named file mapping `CT_IP_{queue}` plus `Global\CT.IP.{queue}` semaphore; the default
//!   [`QueueOptions::path`] uses the same temp-dir subfolder as other platforms for consistency (the mapping does not read from disk).

#[cfg(not(any(unix, windows)))]
compile_error!("The `interprocess` crate only supports `cfg(unix)` and `cfg(windows)` targets.");

mod circular_buffer;
mod error;
mod layout;
mod memory;
#[cfg(windows)]
mod naming;
mod options;
pub mod queue;
mod semaphore;

mod publisher;
mod subscriber;

pub use error::{BackingError, OpenError};
pub use layout::MESSAGE_BODY_OFFSET;
pub use layout::{
    padded_message_length, MessageHeader, QueueHeader, STATE_LOCKED, STATE_READY, STATE_WRITING,
    TICKS_FOR_TEN_SECONDS, TICKS_PER_SECOND,
};
pub use options::{default_memory_dir, QueueOptions, LINUX_SHM_MEMORY_DIR};
#[allow(deprecated)]
pub use options::{DEFAULT_MEMORY_DIR, MEMORY_FILE_PATH};
pub use publisher::Publisher;
pub use queue::QueueFactory;
pub use subscriber::Subscriber;

#[cfg(test)]
mod ipc_tests {
    use crate::options::QueueOptions;
    use crate::{Publisher, Subscriber};

    #[test]
    fn enqueue_and_dequeue_on_shared_backing() {
        let dir =
            std::env::temp_dir().join(format!("interprocess_integration_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let opts = QueueOptions::with_path("integration_queue", &dir, 4096).expect("valid options");
        let mut publisher = Publisher::new(opts.clone()).expect("publisher");
        let mut subscriber = Subscriber::new(opts).expect("subscriber");

        assert!(publisher.try_enqueue(b"hello"));
        assert_eq!(
            subscriber.try_dequeue().as_deref(),
            Some(b"hello".as_slice())
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
