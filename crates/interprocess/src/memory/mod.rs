//! Shared read/write mapping backing the queue.

#[cfg(unix)]
mod unix;
#[cfg(windows)]
mod windows;

use std::path::PathBuf;

use crate::error::OpenError;
use crate::options::QueueOptions;
use crate::semaphore::Semaphore;

/// Mapped region shared between processes (+ companion semaphore at [`open_queue`]).
pub(crate) struct SharedMapping {
    #[cfg(unix)]
    inner: unix::UnixMapping,
    #[cfg(windows)]
    inner: windows::WindowsMapping,
}

impl SharedMapping {
    pub(crate) fn open_queue(options: &QueueOptions) -> Result<(Self, Semaphore), OpenError> {
        #[cfg(unix)]
        {
            let (m, s) = unix::open_queue(options)?;
            debug_assert_eq!(m.len(), options.actual_storage_size() as usize);
            Ok((Self { inner: m }, s))
        }
        #[cfg(windows)]
        {
            let (m, s) = windows::open_queue(options)?;
            debug_assert_eq!(m.len(), options.actual_storage_size() as usize);
            Ok((Self { inner: m }, s))
        }
    }

    pub(crate) fn as_ptr(&self) -> *const u8 {
        self.inner.as_ptr()
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut u8 {
        self.inner.as_mut_ptr()
    }

    /// Path to the backing `.qu` file when file-backed; `None` on Windows named mappings.
    pub(crate) fn backing_file_path(&self) -> Option<&PathBuf> {
        self.inner.backing_file_path()
    }
}
