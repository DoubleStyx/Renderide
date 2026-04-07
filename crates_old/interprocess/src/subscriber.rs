//! Subscriber - reads messages from the queue (bootstrapper receives from host).
//! Uses polling when queue is empty (like zinterprocess); semaphore used only for blocking hint.

use std::fs;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicI64, Ordering};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::backend;
use crate::circular_buffer;
use crate::queue::{
    MESSAGE_BODY_OFFSET, MessageHeader, QueueHeader, QueueOptions, STATE_READY,
    padded_message_length,
};
use crate::sem;

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
/// SAFETY: MemoryBacking uses shared memory; safe to Send when used from one thread at a time per instance.
unsafe impl Send for Subscriber {}

pub struct Subscriber {
    backing: backend::MemoryBacking,
    capacity: i64,
    sem_handle: sem::SemHandle,
    destroy_on_dispose: bool,
}

impl Subscriber {
    /// Creates a new Subscriber. Returns error if the queue file or mmap cannot be opened.
    pub fn new(options: QueueOptions) -> Result<Self, crate::BackingError> {
        let (backing, sem_handle) = backend::open_queue_backing(&options)?;
        Ok(Self {
            backing,
            capacity: options.capacity,
            sem_handle,
            destroy_on_dispose: options.destroy_on_dispose,
        })
    }

    fn header_mut(&mut self) -> *mut QueueHeader {
        self.backing.as_mut_ptr() as *mut QueueHeader
    }

    fn buffer_ptr(&self) -> *const u8 {
        unsafe { self.backing.as_ptr().add(32) }
    }

    fn buffer_mut(&mut self) -> *mut u8 {
        unsafe { self.backing.as_mut_ptr().add(32) }
    }

    /// Sleep when queue is empty (zinterprocess style: no semaphore blocking, just poll + sleep).
    fn wait_or_sleep(&self, ms: i32) {
        if ms > 0 {
            thread::sleep(Duration::from_millis(ms.min(1000) as u64));
        }
    }

    /// Dequeues a message, blocking until one is available or `cancel` is set.
    /// Returns an empty vec if cancelled. Uses sleep/poll when empty (zinterprocess style).
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
                self.wait_or_sleep(1000);
            } else if backoff > 0 {
                self.wait_or_sleep(backoff);
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
        let read_lock_ptr =
            unsafe { &*(&(*header_ptr).read_lock_timestamp as *const i64 as *const AtomicI64) };
        let prev =
            read_lock_ptr.compare_exchange(read_lock, ticks, Ordering::SeqCst, Ordering::SeqCst);
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
                    self.buffer_ptr()
                        .add((read_offset % self.capacity) as usize)
                        as *const MessageHeader
                };

                // Spin until message is ready (state == 2)
                let spin_ticks = ticks;
                loop {
                    let state = unsafe { (*msg_header_ptr).state };
                    if state == STATE_READY {
                        break;
                    }
                    if utc_now_ticks() - spin_ticks > TICKS_FOR_TEN_SECONDS {
                        let read_offset_ptr = unsafe {
                            &*(&(*header_ptr).read_offset as *const i64 as *const AtomicI64)
                        };
                        read_offset_ptr.store(write_offset, Ordering::SeqCst);
                        return None;
                    }
                    std::hint::spin_loop();
                }

                let body_len = unsafe { (*msg_header_ptr).body_length } as i64;
                let padded = padded_message_length(body_len);

                // Mark as locked (state=1) - we're consuming
                let state_ptr =
                    unsafe { &*(&(*msg_header_ptr).state as *const i32 as *const AtomicI32) };
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
                let read_offset_ptr =
                    unsafe { &*(&(*header_ptr).read_offset as *const i64 as *const AtomicI64) };
                read_offset_ptr.store(new_read, Ordering::SeqCst);

                Some(msg_result)
            }
        };

        // Release read lock
        let read_lock_ptr =
            unsafe { &*(&(*header_ptr).read_lock_timestamp as *const i64 as *const AtomicI64) };
        read_lock_ptr.store(0, Ordering::SeqCst);

        result
    }
}

impl Drop for Subscriber {
    fn drop(&mut self) {
        sem::close(&self.sem_handle);
        if self.destroy_on_dispose
            && self.backing.has_file_to_remove()
            && let Some(path) = self.backing.file_path()
        {
            let _ = fs::remove_file(&path);
        }
    }
}
