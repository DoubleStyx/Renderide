//! Circular buffer read/write/clear helpers for the queue backing store.
//! Handles wrap-around when the logical offset plus length exceeds capacity.

/// Reads `len` bytes from the circular buffer at logical `offset`.
pub(super) fn read(buffer: *const u8, capacity: i64, offset: i64, len: usize) -> Vec<u8> {
    if len == 0 {
        return Vec::new();
    }
    let cap = capacity as usize;
    let phys_offset = (offset.rem_euclid(capacity)) as usize;
    let mut result = vec![0u8; len];

    let first = (cap - phys_offset).min(len);
    if first > 0 {
        unsafe {
            result[..first]
                .copy_from_slice(std::slice::from_raw_parts(buffer.add(phys_offset), first));
        }
    }
    if first < len {
        unsafe {
            result[first..].copy_from_slice(std::slice::from_raw_parts(buffer, len - first));
        }
    }
    result
}

/// Writes `data` to the circular buffer at logical `offset`.
pub(super) fn write(buffer: *mut u8, capacity: i64, offset: i64, data: &[u8]) {
    if data.is_empty() {
        return;
    }
    let cap = capacity as usize;
    let phys_offset = (offset.rem_euclid(capacity)) as usize;
    let len = data.len();

    let first = (cap - phys_offset).min(len);
    if first > 0 {
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buffer.add(phys_offset), first);
        }
    }
    if first < len {
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr().add(first), buffer, len - first);
        }
    }
}

/// Zeros `len` bytes in the circular buffer at logical `offset`.
pub(super) fn clear(buffer: *mut u8, capacity: i64, offset: i64, len: usize) {
    if len == 0 {
        return;
    }
    let cap = capacity as usize;
    let phys_offset = (offset.rem_euclid(capacity)) as usize;

    let clear_first = (cap - phys_offset).min(len);
    if clear_first > 0 {
        unsafe {
            std::ptr::write_bytes(buffer.add(phys_offset), 0, clear_first);
        }
    }
    if clear_first < len {
        unsafe {
            std::ptr::write_bytes(buffer, 0, len - clear_first);
        }
    }
}
