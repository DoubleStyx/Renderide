//! Publisher - writes messages to the queue (bootstrapper sends to host).
//! Signals the semaphore when a message is enqueued.

use std::fs;
use std::sync::atomic::Ordering;

use crate::backend;
use crate::circular_buffer;
use crate::queue::{
    MESSAGE_BODY_OFFSET, MessageHeader, QueueOptions, STATE_READY, STATE_WRITING,
    padded_message_length,
};
use crate::sem;

/// Writes messages to the queue. Use with QueueFactory.
/// SAFETY: MemoryBacking uses shared memory; safe to Send when used from one thread at a time per instance.
unsafe impl Send for Publisher {}

pub struct Publisher {
    backing: backend::MemoryBacking,
    capacity: i64,
    sem_handle: sem::SemHandle,
    destroy_on_dispose: bool,
}

impl Publisher {
    /// Creates a new Publisher. Returns error if the queue file or mmap cannot be opened.
    pub fn new(options: QueueOptions) -> Result<Self, crate::BackingError> {
        let (backing, sem_handle) = backend::open_queue_backing(&options)?;
        Ok(Self {
            backing,
            capacity: options.capacity,
            sem_handle,
            destroy_on_dispose: options.destroy_on_dispose,
        })
    }

    fn header_mut(&mut self) -> *mut crate::queue::QueueHeader {
        self.backing.as_mut_ptr() as *mut crate::queue::QueueHeader
    }

    fn buffer_mut(&mut self) -> *mut u8 {
        unsafe { self.backing.as_mut_ptr().add(32) }
    }

    fn check_capacity(&self, header: &crate::queue::QueueHeader, message_len: i64) -> bool {
        if message_len > self.capacity {
            return false;
        }
        if header.is_empty() {
            return true;
        }
        let read_phys = header.read_offset % self.capacity;
        let write_phys = header.write_offset % self.capacity;
        if read_phys == write_phys {
            return false; // buffer full
        }
        let available = if read_phys < write_phys {
            self.capacity - write_phys + read_phys
        } else {
            read_phys - write_phys
        };
        message_len <= available
    }

    /// Tries to enqueue a message. Returns false if the queue is full.
    pub fn try_enqueue(&mut self, message: &[u8]) -> bool {
        let len = message.len() as i64;
        let padded = padded_message_length(len);

        let header_ptr = self.header_mut();

        loop {
            let header = unsafe { &*header_ptr };
            if !self.check_capacity(header, padded) {
                return false;
            }
            let write_offset = header.write_offset;
            let new_write = (write_offset + padded) % (self.capacity * 2);

            let write_offset_ptr = unsafe {
                &*(&(*header_ptr).write_offset as *const i64 as *const std::sync::atomic::AtomicI64)
            };
            let prev = write_offset_ptr.compare_exchange(
                write_offset,
                new_write,
                Ordering::SeqCst,
                Ordering::SeqCst,
            );
            if prev.is_err() {
                continue;
            }

            let msg_header = MessageHeader {
                state: STATE_WRITING,
                body_length: len as i32,
            };
            let header_bytes: [u8; 8] = unsafe { std::mem::transmute(msg_header) };
            circular_buffer::write(
                self.buffer_mut(),
                self.capacity,
                write_offset,
                &header_bytes,
            );

            circular_buffer::write(
                self.buffer_mut(),
                self.capacity,
                write_offset + MESSAGE_BODY_OFFSET,
                message,
            );

            let state_bytes = STATE_READY.to_le_bytes();
            circular_buffer::write(self.buffer_mut(), self.capacity, write_offset, &state_bytes);

            sem::post(&self.sem_handle);
            return true;
        }
    }
}

impl Drop for Publisher {
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
