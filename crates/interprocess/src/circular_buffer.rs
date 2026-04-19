//! Helpers for a byte ring inside a shared mapping.
//!
//! `offset` is a logical position in the ring; it is reduced modulo `capacity` to index the buffer.

/// Copies `len` bytes starting at logical `offset` (wrapping at `capacity`) into a new vector.
pub(crate) fn read(buffer: *const u8, capacity: i64, offset: i64, len: usize) -> Vec<u8> {
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

/// Writes `data` at logical `offset`, wrapping at `capacity`.
pub(crate) fn write(buffer: *mut u8, capacity: i64, offset: i64, data: &[u8]) {
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

/// Zero-fills `len` bytes at logical `offset`, wrapping at `capacity`.
pub(crate) fn clear(buffer: *mut u8, capacity: i64, offset: i64, len: usize) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_read_roundtrip_wrap() {
        let mut buf = [0u8; 6];
        let cap = 6i64;
        write(buf.as_mut_ptr(), cap, 4, &[1, 2, 3]);
        let got = read(buf.as_ptr(), cap, 4, 3);
        assert_eq!(got, vec![1, 2, 3]);
        assert_eq!(buf[4], 1);
        assert_eq!(buf[5], 2);
        assert_eq!(buf[0], 3);
    }

    #[test]
    fn read_zero_len_returns_empty() {
        let buf = [9u8; 4];
        let got = read(buf.as_ptr(), 4, 0, 0);
        assert!(got.is_empty());
    }

    #[test]
    fn write_empty_is_noop() {
        let mut buf = [7u8; 4];
        write(buf.as_mut_ptr(), 4, 2, &[]);
        assert_eq!(buf, [7u8; 4]);
    }

    #[test]
    fn clear_zero_len_is_noop() {
        let mut buf = [5u8; 4];
        clear(buf.as_mut_ptr(), 4, 0, 0);
        assert_eq!(buf, [5u8; 4]);
    }

    #[test]
    fn read_spans_wrap_when_offset_near_capacity_end() {
        let buf = [10u8, 20u8, 30u8, 40u8, 50u8];
        let cap = 5i64;
        let got = read(buf.as_ptr(), cap, 3, 4);
        assert_eq!(got, vec![40u8, 50u8, 10u8, 20u8]);
    }

    #[test]
    fn write_spans_wrap_from_negative_logical_offset() {
        let mut buf = [0u8; 5];
        let cap = 5i64;
        write(buf.as_mut_ptr(), cap, -2, &[1, 2, 3, 4]);
        assert_eq!(buf, [3, 4, 0, 1, 2]);
    }

    #[test]
    fn clear_spans_wrap() {
        let mut buf = [9u8; 6];
        let cap = 6i64;
        clear(buf.as_mut_ptr(), cap, 4, 4);
        assert_eq!(buf, [0u8, 0u8, 9u8, 9u8, 0u8, 0u8]);
    }
}
