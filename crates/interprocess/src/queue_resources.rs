//! Shared backing mapping, capacity, semaphore, and Unix `destroy_on_dispose` cleanup for both queue ends.

use std::fs;

use crate::error::OpenError;
use crate::layout::QueueHeader;
use crate::memory::SharedMapping;
use crate::options::QueueOptions;
use crate::ring::RingView;
use crate::semaphore::Semaphore;

/// Shared resources opened by both [`crate::Publisher::new`] and [`crate::Subscriber::new`].
pub(crate) struct QueueResources {
    /// Read/write mapping of the queue header plus byte ring.
    mapping: SharedMapping,
    /// Ring buffer capacity in bytes (user payload only; excludes the queue header).
    pub(crate) capacity: i64,
    /// Cross-process wakeup object signaled after each successful enqueue.
    sem: Semaphore,
    /// When `true`, best-effort unlink of the backing `.qu` path on drop (Unix file-backed queues only).
    destroy_on_dispose: bool,
}

impl QueueResources {
    /// Creates or opens the mapping and paired semaphore described by `options`.
    pub(crate) fn open(options: QueueOptions) -> Result<Self, OpenError> {
        let (mapping, sem) = SharedMapping::open_queue(&options)?;
        Ok(Self {
            mapping,
            capacity: options.capacity,
            sem,
            destroy_on_dispose: options.destroy_on_dispose,
        })
    }

    /// Shared queue header at the start of the mapping (atomics permit shared references).
    pub(crate) fn header(&self) -> &QueueHeader {
        // SAFETY: `open_queue` maps at least `BUFFER_BYTE_OFFSET + capacity` bytes; the header is
        // `repr(C)` at offset 0 and fits in `BUFFER_BYTE_OFFSET`.
        unsafe { &*(self.mapping.as_ptr() as *const QueueHeader) }
    }

    /// View over the byte ring after [`crate::layout::QueueHeader`].
    pub(crate) fn ring(&self) -> RingView {
        // SAFETY: Ring begins at `BUFFER_BYTE_OFFSET` within the mapping; length is `capacity`.
        unsafe {
            RingView::from_raw(
                self.mapping
                    .as_ptr()
                    .byte_add(crate::layout::BUFFER_BYTE_OFFSET)
                    .cast_mut(),
                self.capacity,
            )
        }
    }

    /// Signals waiters that new data may be available (after enqueue).
    pub(crate) fn post(&self) {
        self.sem.post();
    }

    /// Blocks up to `timeout` waiting for a post (used by blocking dequeue).
    pub(crate) fn wait_semaphore_timeout(&self, timeout: std::time::Duration) -> bool {
        self.sem.wait_timeout(timeout)
    }
}

impl Drop for QueueResources {
    fn drop(&mut self) {
        if self.destroy_on_dispose {
            if let Some(path) = self.mapping.backing_file_path() {
                let _ = fs::remove_file(path);
            }
        }
    }
}
