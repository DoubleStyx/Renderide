//! Shared backing mapping, capacity, semaphore, and Unix `destroy_on_dispose` cleanup for both queue ends.

use std::fs;
use std::io;

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
    /// Logical queue name (matches the mapping/semaphore name); used in diagnostic log lines.
    queue_name: String,
}

impl QueueResources {
    /// Creates or opens the mapping and paired semaphore described by `options`.
    pub(crate) fn open(options: QueueOptions) -> Result<Self, OpenError> {
        let queue_name = options.memory_view_name.clone();
        let capacity = options.capacity;
        let destroy_on_dispose = options.destroy_on_dispose;
        let (mapping, sem) = SharedMapping::open_queue(&options)?;
        logger::info!(
            "interprocess: opened queue '{}' (capacity {} B, destroy_on_dispose={})",
            queue_name,
            capacity,
            destroy_on_dispose
        );
        Ok(Self {
            mapping,
            capacity,
            sem,
            destroy_on_dispose,
            queue_name,
        })
    }

    /// Shared queue header at the start of the mapping (atomics permit shared references).
    pub(crate) fn header(&self) -> &QueueHeader {
        #[expect(
            clippy::cast_ptr_alignment,
            reason = "shared queue mappings are created at page alignment, which is stricter than QueueHeader alignment"
        )]
        let header_ptr = self.mapping.as_ptr().cast::<QueueHeader>();
        // SAFETY: `open_queue` maps at least `BUFFER_BYTE_OFFSET + capacity` bytes; the header is
        // `repr(C)` at offset 0 and fits in `BUFFER_BYTE_OFFSET`.
        unsafe { &*header_ptr }
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
                match fs::remove_file(path) {
                    Ok(()) => logger::info!(
                        "interprocess: unlinked queue backing file for '{}'",
                        self.queue_name
                    ),
                    Err(err) if err.kind() == io::ErrorKind::NotFound => {}
                    Err(err) => logger::warn!(
                        "interprocess: failed to unlink queue backing file for '{}': {}",
                        self.queue_name,
                        err
                    ),
                }
            }
        }
    }
}
