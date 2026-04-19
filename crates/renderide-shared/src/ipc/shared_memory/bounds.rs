//! Validates `[offset, offset+length)` against a mapped buffer’s total length.

use crate::buffer::SharedMemoryBufferDescriptor;

/// Minimum byte capacity required to map `descriptor`’s byte range (`offset + length`), capped by host `buffer_capacity`.
///
/// Returns [`None`] when `length <= 0` or the computed capacity is non-positive.
pub(super) fn required_view_capacity(d: &SharedMemoryBufferDescriptor) -> Option<i32> {
    if d.length <= 0 {
        return None;
    }
    let cap = d.buffer_capacity.max(d.offset.saturating_add(d.length));
    if cap > 0 {
        Some(cap)
    } else {
        None
    }
}

/// Converts `offset`/`length` into a valid byte subrange of `total_len`, or `None`.
pub(super) fn byte_subrange(total_len: usize, offset: i32, length: i32) -> Option<(usize, usize)> {
    let offset = usize::try_from(offset).ok()?;
    let length = usize::try_from(length).ok()?;
    let end = offset.checked_add(length)?;
    if end <= total_len {
        Some((offset, end))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for [`byte_subrange`] and [`required_view_capacity`].

    use super::{byte_subrange, required_view_capacity};
    use crate::buffer::SharedMemoryBufferDescriptor;

    #[test]
    fn byte_subrange_ok_and_rejects_overflow() {
        assert_eq!(byte_subrange(100, 10, 5), Some((10, 15)));
        assert_eq!(byte_subrange(100, 0, 100), Some((0, 100)));
        assert_eq!(byte_subrange(100, 99, 2), None);
        assert_eq!(byte_subrange(100, -1, 5), None);
    }

    #[test]
    fn required_view_capacity_none_when_length_non_positive() {
        let d = SharedMemoryBufferDescriptor {
            buffer_id: 1,
            buffer_capacity: 1000,
            offset: 0,
            length: 0,
        };
        assert_eq!(required_view_capacity(&d), None);
    }

    #[test]
    fn required_view_capacity_uses_max_of_capacity_and_offset_plus_length() {
        let within = SharedMemoryBufferDescriptor {
            buffer_id: 1,
            buffer_capacity: 200,
            offset: 10,
            length: 50,
        };
        assert_eq!(required_view_capacity(&within), Some(200));

        let needs_more = SharedMemoryBufferDescriptor {
            buffer_id: 1,
            buffer_capacity: 40,
            offset: 10,
            length: 50,
        };
        assert_eq!(required_view_capacity(&needs_more), Some(60));
    }

    #[test]
    fn required_view_capacity_saturating_add_handles_large_offset() {
        let d = SharedMemoryBufferDescriptor {
            buffer_id: 1,
            buffer_capacity: 0,
            offset: i32::MAX - 5,
            length: 10,
        };
        assert_eq!(required_view_capacity(&d), Some(i32::MAX));
    }
}
