//! Consumer side of the shared-memory queue.

use std::fs;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicI64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::circular_buffer;
use crate::layout::{
    padded_message_length, MessageHeader, MESSAGE_BODY_OFFSET, STATE_LOCKED, STATE_READY,
    TICKS_FOR_TEN_SECONDS,
};
use crate::memory::SharedMapping;
use crate::options::QueueOptions;
use crate::semaphore::Semaphore;
use crate::QueueHeader;

/// `DateTime.UtcNow.Ticks` value at the Unix epoch (100 ns ticks since 0001-01-01 UTC).
const DOTNET_TICKS_AT_UNIX_EPOCH: i64 = 621_355_968_000_000_000;

fn utc_now_ticks() -> i64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => {
            let since_unix_100ns = (d.as_nanos() / 100) as i64;
            since_unix_100ns.saturating_add(DOTNET_TICKS_AT_UNIX_EPOCH)
        }
        Err(_) => DOTNET_TICKS_AT_UNIX_EPOCH,
    }
}

/// Receives messages from the queue using the same contention and backoff pattern as the managed client.
pub struct Subscriber {
    mapping: SharedMapping,
    capacity: i64,
    sem: Semaphore,
    destroy_on_dispose: bool,
}

impl Subscriber {
    /// Opens the backing mapping and semaphore.
    pub fn new(options: QueueOptions) -> Result<Self, crate::OpenError> {
        let (mapping, sem) = SharedMapping::open_queue(&options)?;
        Ok(Self {
            mapping,
            capacity: options.capacity,
            sem,
            destroy_on_dispose: options.destroy_on_dispose,
        })
    }

    fn header_mut(&mut self) -> *mut QueueHeader {
        self.mapping.as_mut_ptr() as *mut QueueHeader
    }

    fn buffer_ptr(&self) -> *const u8 {
        unsafe { self.mapping.as_ptr().add(crate::layout::BUFFER_BYTE_OFFSET) }
    }

    fn buffer_mut(&mut self) -> *mut u8 {
        unsafe {
            self.mapping
                .as_mut_ptr()
                .add(crate::layout::BUFFER_BYTE_OFFSET)
        }
    }

    /// Blocks until a message arrives or `cancel` is set, using semaphore-backed backoff.
    pub fn dequeue(&mut self, cancel: &AtomicBool) -> Vec<u8> {
        let mut num = -5i32;
        loop {
            if let Some(msg) = self.try_dequeue() {
                return msg;
            }
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            if num > 10 {
                self.sem.wait_timeout(Duration::from_millis(10));
            } else {
                let old_num = num;
                num = num.saturating_add(1);
                if old_num > 0 {
                    self.sem.wait_timeout(Duration::from_millis(num as u64));
                } else {
                    std::thread::yield_now();
                }
            }
        }
        vec![]
    }

    /// Returns the next message if one is ready; non-blocking aside from contender spin windows.
    pub fn try_dequeue(&mut self) -> Option<Vec<u8>> {
        let header_ptr = self.header_mut();
        let header = unsafe { &*header_ptr };

        if header.is_empty() {
            return None;
        }

        let ticks = utc_now_ticks();
        let read_lock = unsafe { (*header_ptr).read_lock_timestamp };
        if ticks - read_lock < TICKS_FOR_TEN_SECONDS {
            return None;
        }

        let read_lock_ptr =
            unsafe { &*(&(*header_ptr).read_lock_timestamp as *const i64 as *const AtomicI64) };
        if read_lock_ptr
            .compare_exchange(read_lock, ticks, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return None;
        }

        let result = (|| {
            let header = unsafe { &*header_ptr };
            if header.is_empty() {
                return None;
            }
            let read_offset = header.read_offset;
            let write_offset = header.write_offset;
            let msg_header_ptr = unsafe {
                self.buffer_ptr()
                    .add((read_offset % self.capacity) as usize)
                    as *const MessageHeader
            };

            let state_ptr =
                unsafe { &*(&(*msg_header_ptr).state as *const i32 as *const AtomicI32) };
            let spin_ticks = ticks;
            loop {
                match state_ptr.compare_exchange(
                    STATE_READY,
                    STATE_LOCKED,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => break,
                    Err(_) => {
                        if utc_now_ticks() - spin_ticks > TICKS_FOR_TEN_SECONDS {
                            let read_offset_ptr = unsafe {
                                &*(&(*header_ptr).read_offset as *const i64 as *const AtomicI64)
                            };
                            read_offset_ptr.store(write_offset, Ordering::SeqCst);
                            return None;
                        }
                        std::hint::spin_loop();
                    }
                }
            }

            let body_len = unsafe { (*msg_header_ptr).body_length } as i64;
            let padded = padded_message_length(body_len);

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
        })();

        read_lock_ptr.store(0, Ordering::SeqCst);
        result
    }
}

impl Drop for Subscriber {
    fn drop(&mut self) {
        if self.destroy_on_dispose {
            if let Some(path) = self.mapping.backing_file_path() {
                let _ = fs::remove_file(path);
            }
        }
    }
}

unsafe impl Send for Subscriber {}
