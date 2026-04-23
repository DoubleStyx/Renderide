//! Byte ring view over the shared mapping after [`crate::layout::QueueHeader`].
//!
//! Logical offsets may be negative or larger than `capacity`; they are reduced with
//! Euclidean modulo before indexing. Cross-process ordering for message bodies is enforced by
//! [`crate::layout::MessageHeader::state`] and the shared [`crate::layout::QueueHeader`] atomics;
//! this type therefore exposes mutating helpers through `&self` while holding a `*mut u8` base.

use std::sync::atomic::Ordering;

use crate::layout::MessageHeader;

/// View of the ring bytes (`capacity` is the user ring length only; excludes [`crate::layout::QueueHeader`]).
#[derive(Copy, Clone)]
pub(crate) struct RingView {
    /// Base pointer to the first byte of the ring (immediately after the queue header in the mapping).
    ptr: *mut u8,
    /// Ring length in bytes (matches [`crate::QueueOptions::capacity`]).
    capacity: i64,
}

/// # Safety
///
/// `ptr` must be valid for reads and writes for `capacity` bytes for the lifetime of queue usage,
/// and `capacity` must be positive. The pointer must refer to the ring region inside the mapping
/// opened by [`crate::memory::SharedMapping::open_queue`].
// SAFETY: see the doc comment above — the pointer is valid for the lifetime of queue usage.
unsafe impl Send for RingView {}

/// # Safety
///
/// All synchronisation for queue data races is provided by atomics in the wire format and by
/// single-writer / single-reader protocol on message bodies; concurrent raw access is allowed
/// only through those contracts.
// SAFETY: see the doc comment above — all data-race synchronization is handled by the wire protocol.
unsafe impl Sync for RingView {}

impl RingView {
    /// Wraps a raw ring base pointer and capacity.
    ///
    /// # Safety
    ///
    /// See [`RingView`] type-level safety requirements. `capacity` must match the options used to
    /// open the mapping.
    pub(crate) unsafe fn from_raw(ptr: *mut u8, capacity: i64) -> Self {
        debug_assert!(capacity > 0, "ring capacity must be positive");
        Self { ptr, capacity }
    }

    /// Message header at the logical start of the current slot.
    ///
    /// # Safety
    ///
    /// The wire protocol requires the eight-byte [`MessageHeader`] to lie in contiguous physical
    /// bytes at `(logical_offset % capacity)`; callers must only use offsets produced by the
    /// publisher after a successful space check.
    pub(crate) unsafe fn message_header_at(&self, logical_offset: i64) -> &MessageHeader {
        let phys = (logical_offset.rem_euclid(self.capacity)) as usize;
        // SAFETY: `phys < capacity` by construction (Euclidean modulo); the ring is a contiguous
        // `capacity` byte region per the `RingView` type-level invariant; the caller guarantees the
        // eight-byte header lies contiguously at this physical offset.
        unsafe { &*self.ptr.add(phys).cast::<MessageHeader>() }
    }

    /// Copies `len` bytes starting at logical `offset` into a new vector.
    pub(crate) fn read(self, offset: i64, len: usize) -> Vec<u8> {
        if len == 0 {
            return Vec::new();
        }
        let (phys, first, second) = split_at_wrap(offset, self.capacity, len);
        let mut result = vec![0u8; len];
        if first > 0 {
            // SAFETY: `split_at_wrap` guarantees `phys + first <= capacity`; the ring region is
            // live and readable for `capacity` bytes per the type invariant.
            unsafe {
                result[..first]
                    .copy_from_slice(std::slice::from_raw_parts(self.ptr.add(phys), first));
            }
        }
        if second > 0 {
            // SAFETY: `split_at_wrap` guarantees `second <= capacity`; reads from the ring base.
            unsafe {
                result[first..].copy_from_slice(std::slice::from_raw_parts(self.ptr, second));
            }
        }
        result
    }

    /// Writes `data` at logical `offset`, wrapping at `capacity`.
    pub(crate) fn write(self, offset: i64, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        let len = data.len();
        let (phys, first, second) = split_at_wrap(offset, self.capacity, len);
        if first > 0 {
            // SAFETY: `phys + first <= capacity`; `data[..first]` and the ring region do not alias
            // (the ring is shared memory; `data` is caller-owned stack/heap). Single-writer wire
            // protocol forbids concurrent writes to this slot.
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr.add(phys), first);
            }
        }
        if second > 0 {
            // SAFETY: same invariants — `second <= capacity`, distinct allocations.
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr().add(first), self.ptr, second);
            }
        }
    }

    /// Zero-fills `len` bytes at logical `offset`, wrapping at `capacity`.
    pub(crate) fn clear(self, offset: i64, len: usize) {
        if len == 0 {
            return;
        }
        let (phys, first, second) = split_at_wrap(offset, self.capacity, len);
        if first > 0 {
            // SAFETY: `phys + first <= capacity`; single-writer protocol guards the slot.
            unsafe {
                std::ptr::write_bytes(self.ptr.add(phys), 0, first);
            }
        }
        if second > 0 {
            // SAFETY: `second <= capacity`.
            unsafe {
                std::ptr::write_bytes(self.ptr, 0, second);
            }
        }
    }
}

/// Returns `(physical_start, first_segment_len, second_segment_len)` for `len` bytes at logical `offset`.
fn split_at_wrap(offset: i64, capacity: i64, len: usize) -> (usize, usize, usize) {
    debug_assert!(capacity > 0, "capacity must be positive");
    let cap = capacity as usize;
    let phys = (offset.rem_euclid(capacity)) as usize;
    let first = (cap - phys).min(len);
    let second = len - first;
    (phys, first, second)
}

/// Returns free bytes in the ring for a new message, or `0` when full.
///
/// When [`crate::layout::QueueHeader::read_offset`] equals [`crate::layout::QueueHeader::write_offset`],
/// the queue is empty and the full `capacity` is available.
pub(crate) fn available_space(header: &crate::layout::QueueHeader, capacity: i64) -> i64 {
    if capacity <= 0 {
        return 0;
    }
    let read_off = header.read_offset.load(Ordering::SeqCst);
    let write_off = header.write_offset.load(Ordering::SeqCst);
    if read_off == write_off {
        return capacity;
    }
    let read_phys = read_off.rem_euclid(capacity);
    let write_phys = write_off.rem_euclid(capacity);
    if read_phys == write_phys {
        return 0;
    }
    let free = if read_phys < write_phys {
        capacity - write_phys + read_phys
    } else {
        read_phys - write_phys
    };
    free.clamp(0, capacity)
}

#[cfg(test)]
mod tests {
    //! # Safety (tests)
    //!
    //! All `unsafe` calls below operate on caller-owned local stack buffers (`buf`) whose lifetime
    //! exceeds the `RingView` and which are not aliased by any other thread for the duration of
    //! the test. `capacity` matches the buffer length. `message_header_at(0)` reads/writes the
    //! first eight bytes of `buf`, which fit entirely inside the allocation.
    use std::mem::size_of;
    use std::sync::atomic::Ordering;

    use super::*;
    use crate::layout::QueueHeader;

    #[test]
    fn split_no_wrap() {
        let (p, a, b) = split_at_wrap(2, 10, 3);
        assert_eq!((p, a, b), (2, 3, 0));
    }

    #[test]
    fn split_exact_end_then_wrap() {
        let (p, a, b) = split_at_wrap(8, 10, 4);
        assert_eq!((p, a, b), (8, 2, 2));
    }

    #[test]
    fn split_full_second_segment() {
        let (p, a, b) = split_at_wrap(0, 6, 6);
        assert_eq!((p, a, b), (0, 6, 0));
    }

    #[test]
    fn split_negative_logical_offset() {
        let (p, a, b) = split_at_wrap(-2, 5, 4);
        assert_eq!(p, 3);
        assert_eq!(a + b, 4);
    }

    #[test]
    fn write_read_roundtrip_wrap() {
        let mut buf = [0u8; 6];
        let cap = 6i64;
        // SAFETY: see module `# Safety (tests)` — `buf` outlives `ring`, `cap` matches length.
        let ring = unsafe { RingView::from_raw(buf.as_mut_ptr(), cap) };
        ring.write(4, &[1, 2, 3]);
        let got = ring.read(4, 3);
        assert_eq!(got, vec![1, 2, 3]);
        assert_eq!(buf[4], 1);
        assert_eq!(buf[5], 2);
        assert_eq!(buf[0], 3);
    }

    #[test]
    fn read_zero_len_returns_empty() {
        let buf = [9u8; 4];
        // SAFETY: read-only test; `buf` outlives `ring`, capacity matches length.
        let ring = unsafe { RingView::from_raw(buf.as_ptr().cast_mut(), 4) };
        let got = ring.read(0, 0);
        assert!(got.is_empty());
    }

    #[test]
    fn write_empty_is_noop() {
        let mut buf = [7u8; 4];
        // SAFETY: see module `# Safety (tests)` — `buf` outlives `ring`, capacity matches length.
        let ring = unsafe { RingView::from_raw(buf.as_mut_ptr(), 4) };
        ring.write(2, &[]);
        assert_eq!(buf, [7u8; 4]);
    }

    #[test]
    fn clear_zero_len_is_noop() {
        let mut buf = [5u8; 4];
        // SAFETY: see module `# Safety (tests)` — `buf` outlives `ring`, capacity matches length.
        let ring = unsafe { RingView::from_raw(buf.as_mut_ptr(), 4) };
        ring.clear(0, 0);
        assert_eq!(buf, [5u8; 4]);
    }

    #[test]
    fn read_spans_wrap_when_offset_near_capacity_end() {
        let buf = [10u8, 20u8, 30u8, 40u8, 50u8];
        // SAFETY: read-only test; `buf` outlives `ring`, capacity matches length.
        let ring = unsafe { RingView::from_raw(buf.as_ptr().cast_mut(), 5) };
        let got = ring.read(3, 4);
        assert_eq!(got, vec![40u8, 50u8, 10u8, 20u8]);
    }

    #[test]
    fn write_spans_wrap_from_negative_logical_offset() {
        let mut buf = [0u8; 5];
        // SAFETY: see module `# Safety (tests)` — `buf` outlives `ring`, capacity matches length.
        let ring = unsafe { RingView::from_raw(buf.as_mut_ptr(), 5) };
        ring.write(-2, &[1, 2, 3, 4]);
        assert_eq!(buf, [3, 4, 0, 1, 2]);
    }

    #[test]
    fn clear_spans_wrap() {
        let mut buf = [9u8; 6];
        // SAFETY: see module `# Safety (tests)` — `buf` outlives `ring`, capacity matches length.
        let ring = unsafe { RingView::from_raw(buf.as_mut_ptr(), 6) };
        ring.clear(4, 4);
        assert_eq!(buf, [0u8, 0u8, 9u8, 9u8, 0u8, 0u8]);
    }

    #[test]
    fn read_full_ring_length() {
        let buf = [1u8, 2, 3, 4, 5, 6];
        // SAFETY: read-only test; `buf` outlives `ring`, capacity matches length.
        let ring = unsafe { RingView::from_raw(buf.as_ptr().cast_mut(), 6) };
        let got = ring.read(0, 6);
        assert_eq!(got, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn available_space_empty_queue() {
        let h = QueueHeader::default();
        assert_eq!(available_space(&h, 64), 64);
    }

    #[test]
    fn available_space_when_physically_full() {
        let h = QueueHeader::default();
        h.read_offset.store(0, Ordering::SeqCst);
        h.write_offset.store(8, Ordering::SeqCst);
        assert_eq!(available_space(&h, 8), 0);
    }

    #[test]
    fn available_space_in_use() {
        let h = QueueHeader::default();
        h.read_offset.store(0, Ordering::SeqCst);
        h.write_offset.store(8, Ordering::SeqCst);
        assert_eq!(available_space(&h, 24), 24 - 8);
    }

    #[test]
    fn message_header_at_reads_state() {
        use crate::layout::{MessageHeader, STATE_WRITING};

        let mut buf = [0u8; 64];
        // SAFETY: see module `# Safety (tests)` — `buf` outlives `ring`, capacity matches length.
        let ring = unsafe { RingView::from_raw(buf.as_mut_ptr(), 64) };
        // SAFETY: offset 0 is a valid 8-byte header slot inside the 64-byte buffer.
        let mh = unsafe { ring.message_header_at(0) };
        mh.state.store(STATE_WRITING, Ordering::SeqCst);
        // SAFETY: same slot, unchanged layout.
        let mh2 = unsafe { ring.message_header_at(0) };
        assert_eq!(mh2.state.load(Ordering::SeqCst), STATE_WRITING);
        assert_eq!(size_of::<MessageHeader>(), 8);
    }
}
