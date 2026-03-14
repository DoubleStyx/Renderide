//! Shared initialization for queue file, mmap, and semaphore.

use std::fs::{self, File, OpenOptions};
use std::path::PathBuf;

use libc::{sem_open, O_CREAT};
use memmap2::MmapMut;

use crate::queue::QueueOptions;

/// Opens the queue file, mmaps it, and creates the semaphore.
/// Returns (file, mmap, sem_handle, file_path) for use by Subscriber or Publisher.
pub(super) fn open_queue_backing(
    options: &QueueOptions,
) -> (File, MmapMut, *mut libc::sem_t, PathBuf) {
    let path = options.file_path();
    fs::create_dir_all(path.parent().unwrap()).ok();

    let storage_size = options.actual_storage_size() as u64;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(&path)
        .expect("Failed to open queue file");

    file.set_len(storage_size).expect("Failed to set file length");

    let mmap = unsafe {
        MmapMut::map_mut(&file).expect("Failed to mmap queue file")
    };

    let sem_name = options.semaphore_name();
    let sem_c_name = std::ffi::CString::new(sem_name.as_str()).expect("CString");
    let sem_handle = unsafe {
        sem_open(
            sem_c_name.as_ptr(),
            O_CREAT,
            0o777,
            0,
        )
    };

    if sem_handle == libc::SEM_FAILED {
        panic!("Failed to open semaphore: {}", sem_name);
    }

    (file, mmap, sem_handle, path)
}
