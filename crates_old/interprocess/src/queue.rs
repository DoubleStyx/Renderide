//! Queue structures matching Cloudtoid.Interprocess layout.

use std::path::PathBuf;

/// Path for shared memory files on Unix (Linux/Wine).
pub const MEMORY_FILE_PATH: &str = "/dev/shm/.cloudtoid/interprocess/mmf";

/// Queue header - 32 bytes, matches Cloudtoid QueueHeader.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct QueueHeader {
    pub read_offset: i64,
    pub write_offset: i64,
    pub read_lock_timestamp: i64,
    pub reserved: i64,
}

impl QueueHeader {
    /// Returns true if the queue has no messages (read and write offsets match).
    pub fn is_empty(&self) -> bool {
        self.read_offset == self.write_offset
    }
}

/// Message header - 8 bytes, matches Cloudtoid MessageHeader.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MessageHeader {
    pub state: i32,
    pub body_length: i32,
}

/// Message state: publisher is writing.
pub const STATE_WRITING: i32 = 0;
/// Message state: ready for subscriber to consume.
pub const STATE_READY: i32 = 2;

/// Padded message length (8-byte aligned). Includes MessageHeader + body.
pub fn padded_message_length(body_len: i64) -> i64 {
    let total = 8 + body_len; // MessageHeader + body
    ((total + 7) / 8) * 8
}

/// Offset of message body from message start (MessageHeader size).
pub const MESSAGE_BODY_OFFSET: i64 = 8;

/// Options for creating a Subscriber or Publisher.
pub struct QueueOptions {
    pub memory_view_name: String,
    pub path: PathBuf,
    pub capacity: i64,
    pub destroy_on_dispose: bool,
}

impl QueueOptions {
    /// Creates options with default path and destroy_on_dispose = false.
    pub fn new(queue_name: &str, capacity: i64) -> Self {
        Self {
            memory_view_name: queue_name.to_string(),
            path: PathBuf::from(MEMORY_FILE_PATH),
            capacity,
            destroy_on_dispose: false,
        }
    }

    /// Creates options with explicit destroy_on_dispose.
    pub fn with_destroy(queue_name: &str, capacity: i64, destroy_on_dispose: bool) -> Self {
        Self {
            memory_view_name: queue_name.to_string(),
            path: PathBuf::from(MEMORY_FILE_PATH),
            capacity,
            destroy_on_dispose,
        }
    }

    /// Total storage size: QueueHeader (32) + buffer capacity.
    pub fn actual_storage_size(&self) -> i64 {
        32 + self.capacity // QueueHeader + buffer
    }

    /// Path to the queue file (e.g. /dev/shm/.cloudtoid/interprocess/mmf/name.qu).
    pub fn file_path(&self) -> PathBuf {
        self.path.join(format!("{}.qu", self.memory_view_name))
    }

    /// POSIX semaphore name for blocking (e.g. /ct.ip.name).
    pub fn semaphore_name(&self) -> String {
        format!("/ct.ip.{}", self.memory_view_name)
    }
}

/// Factory for creating Subscriber and Publisher instances.
pub struct QueueFactory;

impl QueueFactory {
    /// Creates a new factory.
    pub fn new() -> Self {
        Self
    }

    /// Creates a Subscriber that reads from the queue.
    pub fn create_subscriber(
        &self,
        options: QueueOptions,
    ) -> Result<super::Subscriber, super::BackingError> {
        super::Subscriber::new(options)
    }

    /// Creates a Publisher that writes to the queue.
    pub fn create_publisher(
        &self,
        options: QueueOptions,
    ) -> Result<super::Publisher, super::BackingError> {
        super::Publisher::new(options)
    }
}

impl Default for QueueFactory {
    fn default() -> Self {
        Self::new()
    }
}
