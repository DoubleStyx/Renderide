//! Subscriber - reads messages from the queue (bootstrapper receives from host).
//! Uses POSIX semaphores for blocking when the queue is empty.

use std::fs::{self, File};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicI64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use libc::{sem_close, sem_timedwait, sem_trywait, sem_wait, timespec};
use memmap2::MmapMut;

use crate::backend;
use crate::circular_buffer;
use crate::queue::{
    MessageHeader, QueueHeader, QueueOptions, MESSAGE_BODY_OFFSET, STATE_READY,
    padded_message_length,
};

const TICKS_PER_SECOND: i64 = 10_000_000;
const TICKS_FOR_TEN_SECONDS: i64 = 10 * TICKS_PER_SECOND;

fn utc_now_ticks() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as i64
        / 100
}

/// Reads messages from the queue. Use with QueueFactory.
pub struct Subscriber {
    _file: File,
    mmap: MmapMut,
    capacity: i64,
    sem_handle: *mut libc::sem_t,
    file_path: std::path::PathBuf,
    destroy_on_dispose: bool,
}

impl Subscriber {
    /// Creates a new Subscriber. Panics if the queue file or semaphore cannot be opened.
    pub fn new(options: QueueOptions) -> Self {
        let (file, mmap, sem_handle, file_path) = backend::open_queue_backing(&options);
        Self {
            _file: file,
            mmap,
            capacity: options.capacity,
            sem_handle,
            file_path,
            destroy_on_dispose: options.destroy_on_dispose,
        }
    }

    fn header_mut(&mut self) -> *mut QueueHeader {
        self.mmap.as_mut_ptr() as *mut QueueHeader
    }

    fn buffer_ptr(&self) -> *const u8 {
        unsafe { self.mmap.as_ptr().add(32) }
    }

    fn buffer_mut(&mut self) -> *mut u8 {
        unsafe { self.mmap.as_mut_ptr().add(32) }
    }

    fn sem_wait_timeout(&self, ms: i32) -> bool {
        if ms == -1 {
            unsafe { sem_wait(self.sem_handle) == 0 }
        } else if ms == 0 {
            unsafe { sem_trywait(self.sem_handle) == 0 }
        } else {
            let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
            let timeout = now + Duration::from_millis(ms as u64);
            let ts = timespec {
                tv_sec: timeout.as_secs() as i64,
                tv_nsec: timeout.subsec_nanos() as i64,
            };
            unsafe { sem_timedwait(self.sem_handle, &ts) == 0 }
        }
    }

    /// Dequeues a message, blocking until one is available or `cancel` is set.
    /// Returns an empty vec if cancelled. Uses sem_wait when backoff > 10 to avoid spinning.
    pub fn dequeue(&mut self, cancel: &AtomicBool) -> Vec<u8> {
        let mut backoff = -5i32;
        loop {
            if let Some(msg) = self.try_dequeue() {
                return msg;
            }
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            backoff += 1;
            if backoff > 10 {
                // Block up to 1s so we can check cancel periodically; avoids burning CPU
                self.sem_wait_timeout(1000);
            } else if backoff > 0 {
                self.sem_wait_timeout(backoff);
            } else {
                std::hint::spin_loop();
            }
        }
        vec![]
    }

    /// Tries to dequeue without blocking. Returns None if the queue is empty.
    pub fn try_dequeue(&mut self) -> Option<Vec<u8>> {
        let header_ptr = self.header_mut();
        let header = unsafe { &*header_ptr };

        if header.is_empty() {
            return None;
        }

        let ticks = utc_now_ticks();
        let read_lock = unsafe { (*header_ptr).read_lock_timestamp };
        if ticks - read_lock < TICKS_FOR_TEN_SECONDS && read_lock != 0 {
            return None;
        }

        // Try to acquire read lock
        let read_lock_ptr = unsafe { &*(&(*header_ptr).read_lock_timestamp as *const i64 as *const AtomicI64) };
        let prev = read_lock_ptr.compare_exchange(read_lock, ticks, Ordering::SeqCst, Ordering::SeqCst);
        if prev.is_err() {
            return None;
        }

        let result = {
            let header = unsafe { &*header_ptr };
            if header.is_empty() {
                None
            } else {
                let read_offset = header.read_offset;
                let write_offset = header.write_offset;
                let msg_header_ptr = unsafe {
                    self.buffer_ptr().add((read_offset % self.capacity) as usize) as *const MessageHeader
                };

                // Spin until message is ready (state == 2)
                let spin_ticks = ticks;
                loop {
                    let state = unsafe { (*msg_header_ptr).state };
                    if state == STATE_READY {
                        break;
                    }
                    if utc_now_ticks() - spin_ticks > TICKS_FOR_TEN_SECONDS {
                        let read_offset_ptr = unsafe { &*(&(*header_ptr).read_offset as *const i64 as *const AtomicI64) };
                        read_offset_ptr.store(write_offset, Ordering::SeqCst);
                        return None;
                    }
                    std::hint::spin_loop();
                }

                let body_len = unsafe { (*msg_header_ptr).body_length } as i64;
                let padded = padded_message_length(body_len);

                // Mark as locked (state=1) - we're consuming
                let state_ptr = unsafe { &*(&(*(msg_header_ptr as *const MessageHeader)).state as *const i32 as *const AtomicI32) };
                state_ptr.store(1, Ordering::SeqCst);

                let body_offset = read_offset + MESSAGE_BODY_OFFSET;
                let body_len_usize = body_len as usize;
                let msg_result = circular_buffer::read(
                    self.buffer_ptr(),
                    self.capacity,
                    body_offset,
                    body_len_usize,
                );

                circular_buffer::clear(
                    self.buffer_mut(),
                    self.capacity,
                    read_offset,
                    padded as usize,
                );

                let new_read = (read_offset + padded) % (self.capacity * 2);
                let read_offset_ptr = unsafe { &*(&(*header_ptr).read_offset as *const i64 as *const AtomicI64) };
                read_offset_ptr.store(new_read, Ordering::SeqCst);

                Some(msg_result)
            }
        };

        // Release read lock
        let read_lock_ptr = unsafe { &*(&(*header_ptr).read_lock_timestamp as *const i64 as *const AtomicI64) };
        read_lock_ptr.store(0, Ordering::SeqCst);

        result
    }
}

impl Drop for Subscriber {
    fn drop(&mut self) {
        unsafe {
            sem_close(self.sem_handle);
        }
        if self.destroy_on_dispose {
            let _ = fs::remove_file(&self.file_path);
        }
    }
}
