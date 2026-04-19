//! Shared read/write mapping backing the queue.

#[cfg(unix)]
mod unix;
#[cfg(windows)]
mod windows;

use std::path::PathBuf;

use crate::error::OpenError;
use crate::options::QueueOptions;
use crate::semaphore::Semaphore;

/// Read/write mapping of the queue file (Unix) or named section (Windows), plus a paired semaphore.
///
/// Obtained via [`SharedMapping::open_queue`]; size matches [`QueueOptions::actual_storage_size`].
pub(crate) struct SharedMapping {
    /// Platform-specific mapping implementation.
    #[cfg(unix)]
    inner: unix::UnixMapping,
    /// Platform-specific mapping implementation.
    #[cfg(windows)]
    inner: windows::WindowsMapping,
}

impl SharedMapping {
    /// Creates or opens the backing store for `options` and the companion wakeup semaphore.
    pub(crate) fn open_queue(options: &QueueOptions) -> Result<(Self, Semaphore), OpenError> {
        #[cfg(unix)]
        {
            let (m, s) = unix::open_queue(options)?;
            let mapping = Self { inner: m };
            debug_assert_eq!(mapping.len(), options.actual_storage_size() as usize);
            Ok((mapping, s))
        }
        #[cfg(windows)]
        {
            let (m, s) = windows::open_queue(options)?;
            let mapping = Self { inner: m };
            debug_assert_eq!(mapping.len(), options.actual_storage_size() as usize);
            Ok((mapping, s))
        }
    }

    /// Base pointer to the mapped region (includes [`crate::layout::QueueHeader`] at offset zero).
    ///
    /// # Safety
    ///
    /// The returned pointer is valid for reads and writes for [`Self::len`] bytes for the lifetime
    /// of `self`. Aliasing follows the queue wire protocol (atomics + slot state machine).
    pub(crate) fn as_ptr(&self) -> *const u8 {
        self.inner.as_ptr()
    }

    /// Byte length of the mapping (queue header plus ring).
    pub(crate) fn len(&self) -> usize {
        self.inner.len()
    }

    /// Path to the backing `.qu` file when file-backed; `None` on Windows named mappings.
    pub(crate) fn backing_file_path(&self) -> Option<&PathBuf> {
        self.inner.backing_file_path()
    }
}
