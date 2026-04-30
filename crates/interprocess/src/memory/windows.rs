//! Named file mapping on Windows (`CT_IP_{name}`).

use std::ffi::OsStr;
use std::io;
use std::os::windows::ffi::OsStrExt;
use std::ptr::null;

use crate::error::OpenError;
use crate::naming;
use crate::options::QueueOptions;
use crate::semaphore::Semaphore;
use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, MapViewOfFile, OpenFileMappingW, UnmapViewOfFile, FILE_MAP_ALL_ACCESS,
    PAGE_READWRITE,
};

/// Maximum number of bootstrap-race retries for [`create_or_open_file_mapping`].
const MAP_RETRY_ATTEMPTS: usize = 14;

/// Initial sleep, in milliseconds, between bootstrap-race retries.
const MAP_RETRY_INITIAL_MS: u64 = 10;

/// Upper bound on the bootstrap-race retry sleep, in milliseconds.
const MAP_RETRY_MAX_MS: u64 = 1_000;

/// RAII for `CreateFileMappingW` / `OpenFileMappingW` plus `MapViewOfFile`.
pub(super) struct WindowsMapping {
    /// Handle from `CreateFileMappingW` or `OpenFileMappingW`.
    map_handle: windows_sys::Win32::Foundation::HANDLE,
    /// Mapped view of the queue bytes.
    view: windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS,
    /// Byte length of the view (header plus ring).
    len: usize,
    /// Logical queue name (matches the semaphore name); used in diagnostic log lines.
    queue_name: String,
}

impl WindowsMapping {
    /// Returns the start of the mapped section.
    pub(super) fn as_ptr(&self) -> *const u8 {
        self.view.Value as *const u8
    }

    /// Length of the mapping in bytes.
    pub(super) fn len(&self) -> usize {
        self.len
    }

    /// Always [`None`]; Windows uses named mappings, not a `.qu` path.
    pub(super) fn backing_file_path(&self) -> Option<&std::path::PathBuf> {
        None
    }
}

impl Drop for WindowsMapping {
    fn drop(&mut self) {
        if !self.view.Value.is_null() {
            // SAFETY: `self.view` was returned by `MapViewOfFile` and is owned by `self`; unmapped
            // exactly once on drop.
            let rc = unsafe { UnmapViewOfFile(self.view) };
            if rc == 0 {
                logger::warn!(
                    "interprocess: UnmapViewOfFile failed for queue '{}': {}",
                    self.queue_name,
                    io::Error::last_os_error()
                );
            }
        }
        if !self.map_handle.is_null() && self.map_handle != INVALID_HANDLE_VALUE {
            // SAFETY: `self.map_handle` was opened in `open_queue` and is owned by `self`; closed
            // exactly once on drop.
            let rc = unsafe { CloseHandle(self.map_handle) };
            if rc == 0 {
                logger::warn!(
                    "interprocess: CloseHandle on mapping failed for queue '{}': {}",
                    self.queue_name,
                    io::Error::last_os_error()
                );
            }
        }
    }
}

/// Creates or opens the named file mapping, maps it, and opens the paired Win32 semaphore.
pub(super) fn open_queue(options: &QueueOptions) -> Result<(WindowsMapping, Semaphore), OpenError> {
    let name = naming::windows_mapping_name(&options.memory_view_name);
    let storage_size = usize::try_from(options.actual_storage_size()).map_err(|err| {
        OpenError(io::Error::other(format!(
            "queue storage size does not fit usize (capacity {}): {err}",
            options.capacity
        )))
    })?;

    let name_wide: Vec<u16> = OsStr::new(&name)
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();

    let map_handle = create_or_open_file_mapping(&name_wide, storage_size)?;

    // SAFETY: `map_handle` was just returned as valid by `create_or_open_file_mapping`; zero
    // offsets request a full-length view.
    let view = unsafe { MapViewOfFile(map_handle, FILE_MAP_ALL_ACCESS, 0, 0, storage_size) };

    if view.Value.is_null() {
        // SAFETY: `map_handle` is live; closed exactly once on this error path.
        unsafe { CloseHandle(map_handle) };
        return Err(OpenError(io::Error::other(format!(
            "MapViewOfFile failed for queue: {}",
            options.memory_view_name
        ))));
    }

    let sem = Semaphore::open(&options.memory_view_name).map_err(OpenError)?;

    Ok((
        WindowsMapping {
            map_handle,
            view,
            len: storage_size,
            queue_name: options.memory_view_name.clone(),
        },
        sem,
    ))
}

/// Attempts `CreateFileMappingW` then `OpenFileMappingW` with exponential backoff.
///
/// Two processes often race during bootstrap: one creates the section while the other opens it.
/// `CreateFileMappingW` fails with `ERROR_ALREADY_EXISTS` when the object already exists; the
/// caller then falls back to `OpenFileMappingW`. Transient failures while the creator finishes
/// mapping setup are absorbed by retrying with bounded exponential sleep
/// ([`MAP_RETRY_INITIAL_MS`] initial, [`MAP_RETRY_MAX_MS`] cap, [`MAP_RETRY_ATTEMPTS`] attempts).
fn create_or_open_file_mapping(
    name: &[u16],
    size: usize,
) -> Result<windows_sys::Win32::Foundation::HANDLE, OpenError> {
    let mut wait_retries: usize = MAP_RETRY_ATTEMPTS;
    let mut wait_sleep_ms: u64 = 0;

    loop {
        // SAFETY: `name` is a NUL-terminated wide string; `INVALID_HANDLE_VALUE` plus a non-zero
        // size requests an anonymous (pagefile-backed) named mapping.
        let handle = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                null(),
                PAGE_READWRITE,
                (size >> 32) as u32,
                (size & 0xFFFF_FFFF) as u32,
                name.as_ptr(),
            )
        };

        if !handle.is_null() && handle != INVALID_HANDLE_VALUE {
            return Ok(handle);
        }

        // SAFETY: `name` is a NUL-terminated wide string.
        let handle = unsafe { OpenFileMappingW(FILE_MAP_ALL_ACCESS, 0, name.as_ptr()) };

        if !handle.is_null() && handle != INVALID_HANDLE_VALUE {
            return Ok(handle);
        }

        wait_retries = match wait_retries.checked_sub(1) {
            Some(n) => n,
            None => {
                return Err(OpenError(io::Error::other(
                    "Failed to create or open file mapping after retries",
                )));
            }
        };

        if wait_sleep_ms == 0 {
            wait_sleep_ms = MAP_RETRY_INITIAL_MS;
        } else {
            std::thread::sleep(std::time::Duration::from_millis(wait_sleep_ms));
            wait_sleep_ms = (wait_sleep_ms * 2).min(MAP_RETRY_MAX_MS);
        }
    }
}
