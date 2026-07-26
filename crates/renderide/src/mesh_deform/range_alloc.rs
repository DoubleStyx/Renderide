//! First-fit byte-range allocator for GPU arena suballocation (GPU skin cache).
//!
//! Free ranges are kept sorted by offset; adjacent free ranges are merged on [`RangeAllocator::free`].

/// Byte offset and length inside a parent arena buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Range {
    /// Inclusive byte offset from the start of the arena.
    pub offset_bytes: u64,
    /// Length of this region in bytes.
    pub len_bytes: u64,
}

impl Range {
    /// `self.offset_bytes / stride` as element index (integer division).
    #[inline]
    pub fn first_element_index(self, element_stride_bytes: u64) -> u32 {
        self.offset_bytes
            .checked_div(element_stride_bytes)
            .map_or(0, |v| v.min(u64::from(u32::MAX)) as u32)
    }

    /// Number of elements of `element_stride_bytes` that fit in this range (may truncate if `len_bytes` is not a multiple).
    #[cfg(test)]
    #[inline]
    pub fn element_count(self, element_stride_bytes: u64) -> u32 {
        if element_stride_bytes == 0 {
            return 0;
        }
        let n = self.len_bytes / element_stride_bytes;
        n.min(u64::from(u32::MAX)) as u32
    }

    /// Half-open byte range for [`wgpu::Buffer::slice`].
    #[inline]
    pub fn byte_range(self) -> std::ops::Range<u64> {
        self.offset_bytes..self.offset_bytes.saturating_add(self.len_bytes)
    }
}

/// First-fit allocator over a fixed capacity arena.
#[derive(Debug)]
pub struct RangeAllocator {
    capacity: u64,
    align: u64,
    /// Sorted by offset; non-overlapping free holes.
    free: Vec<Range>,
}

impl RangeAllocator {
    /// Creates an allocator with `capacity` bytes, all free. `align` must be > 0 (typically 256 for WebGPU storage offsets).
    pub fn new(capacity: u64, align: u64) -> Self {
        debug_assert!(align > 0);
        let mut free = Vec::new();
        if capacity > 0 {
            free.push(Range {
                offset_bytes: 0,
                len_bytes: capacity,
            });
        }
        Self {
            capacity,
            align,
            free,
        }
    }

    /// Total arena capacity in bytes.
    #[inline]
    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// Bytes currently allocated (capacity minus free holes).
    pub fn used_bytes(&self) -> u64 {
        self.capacity
            .saturating_sub(self.free.iter().map(|r| r.len_bytes).sum::<u64>())
    }

    /// Aligns `len` up to [`Self::align`], then allocates the first free region that fits.
    pub fn allocate(&mut self, len_bytes: u64) -> Option<Range> {
        let need = align_up(len_bytes, self.align);
        if need == 0 || need > self.capacity {
            return None;
        }
        let idx = self.free.iter().position(|r| r.len_bytes >= need)?;
        let old = self.free[idx];
        let used = Range {
            offset_bytes: old.offset_bytes,
            len_bytes: need,
        };
        let remainder = old.len_bytes.saturating_sub(need);
        if remainder > 0 {
            self.free[idx] = Range {
                offset_bytes: old.offset_bytes.saturating_add(need),
                len_bytes: remainder,
            };
        } else {
            self.free.remove(idx);
        }
        debug_assert!(self.debug_free_invariant());
        Some(used)
    }

    /// Aligns `len` up, then allocates from the top of the highest free region that fits.
    ///
    /// Callers that suballocate parallel per element buffers indexed by this arena's offsets use
    /// this for entries that need none of those buffers, so the low region stays reserved for the
    /// entries that do and the parallel buffers stay dense.
    pub fn allocate_high(&mut self, len_bytes: u64) -> Option<Range> {
        let need = align_up(len_bytes, self.align);
        if need == 0 || need > self.capacity {
            return None;
        }
        let idx = self.free.iter().rposition(|r| r.len_bytes >= need)?;
        let old = self.free[idx];
        let offset_bytes = align_down(
            old.offset_bytes
                .saturating_add(old.len_bytes)
                .saturating_sub(need),
            self.align,
        );
        if offset_bytes < old.offset_bytes {
            return None;
        }
        let used = Range {
            offset_bytes,
            len_bytes: need,
        };
        let head = offset_bytes.saturating_sub(old.offset_bytes);
        let tail = old
            .offset_bytes
            .saturating_add(old.len_bytes)
            .saturating_sub(offset_bytes.saturating_add(need));
        if head > 0 {
            self.free[idx].len_bytes = head;
        } else {
            self.free.remove(idx);
        }
        if tail > 0 {
            let insert_at = if head > 0 { idx + 1 } else { idx };
            self.free.insert(
                insert_at,
                Range {
                    offset_bytes: offset_bytes.saturating_add(need),
                    len_bytes: tail,
                },
            );
        }
        debug_assert!(self.debug_free_invariant());
        Some(used)
    }

    /// Returns a free range to the allocator and merges with adjacent free holes.
    pub fn free(&mut self, range: Range) {
        if range.len_bytes == 0 {
            return;
        }
        let end = range.offset_bytes.saturating_add(range.len_bytes);
        debug_assert!(end <= self.capacity);
        self.free.push(range);
        self.free.sort_by_key(|r| r.offset_bytes);
        self.merge_adjacent();
        debug_assert!(self.debug_free_invariant());
    }

    /// Extends the arena to `new_capacity` (must be >= current capacity). New free space is appended at the end.
    pub fn grow_to(&mut self, new_capacity: u64) {
        if new_capacity < self.capacity {
            return;
        }
        let extra = new_capacity - self.capacity;
        if extra > 0 {
            self.free.push(Range {
                offset_bytes: self.capacity,
                len_bytes: extra,
            });
            self.capacity = new_capacity;
            self.free.sort_by_key(|r| r.offset_bytes);
            self.merge_adjacent();
        }
        debug_assert!(self.debug_free_invariant());
    }

    fn merge_adjacent(&mut self) {
        if self.free.len() < 2 {
            return;
        }
        let mut merged: Vec<Range> = Vec::with_capacity(self.free.len());
        for r in self.free.drain(..) {
            if let Some(last) = merged.last_mut() {
                let last_end = last.offset_bytes.saturating_add(last.len_bytes);
                if last_end == r.offset_bytes {
                    last.len_bytes = last.len_bytes.saturating_add(r.len_bytes);
                    continue;
                }
            }
            merged.push(r);
        }
        self.free = merged;
    }

    fn debug_free_invariant(&self) -> bool {
        for w in self.free.windows(2) {
            let a = w[0];
            let b = w[1];
            let a_end = a.offset_bytes.saturating_add(a.len_bytes);
            if a_end > b.offset_bytes {
                return false;
            }
        }
        true
    }
}

#[inline]
fn align_up(n: u64, align: u64) -> u64 {
    if align == 0 {
        return n;
    }
    (n.saturating_add(align - 1)) / align * align
}

#[inline]
fn align_down(n: u64, align: u64) -> u64 {
    if align == 0 {
        return n;
    }
    n / align * align
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_aligns_and_splits() {
        let mut a = RangeAllocator::new(1024, 256);
        let r0 = a.allocate(256).unwrap();
        assert_eq!(r0.offset_bytes, 0);
        assert_eq!(r0.len_bytes, 256);
        assert_eq!(r0.element_count(64), 4);
        let r1 = a.allocate(256).unwrap();
        assert_eq!(r1.offset_bytes, 256);
        assert_eq!(a.free.len(), 1);
        assert_eq!(a.free[0].offset_bytes, 512);
        assert_eq!(a.free[0].len_bytes, 512);
    }

    #[test]
    fn allocate_aligns_up() {
        let mut a = RangeAllocator::new(1024, 256);
        let r = a.allocate(1).unwrap();
        assert_eq!(r.len_bytes, 256);
    }

    #[test]
    fn allocate_high_takes_the_top_and_keeps_the_low_region_free() {
        let mut a = RangeAllocator::new(1024, 256);

        let high = a.allocate_high(256).unwrap();
        assert_eq!(high.offset_bytes, 768);
        let low = a.allocate(256).unwrap();
        assert_eq!(low.offset_bytes, 0);

        assert_eq!(a.free.len(), 1);
        assert_eq!(a.free[0].offset_bytes, 256);
        assert_eq!(a.free[0].len_bytes, 512);
    }

    #[test]
    fn allocate_high_reuses_freed_top_ranges() {
        let mut a = RangeAllocator::new(1024, 256);
        let first = a.allocate_high(256).unwrap();
        let second = a.allocate_high(256).unwrap();
        assert_eq!(second.offset_bytes, 512);

        a.free(first);
        assert_eq!(a.allocate_high(256).unwrap().offset_bytes, 768);
    }

    #[test]
    fn free_merges_adjacent() {
        let mut a = RangeAllocator::new(1024, 256);
        let r0 = a.allocate(256).unwrap();
        let r1 = a.allocate(256).unwrap();
        a.free(r0);
        a.free(r1);
        assert_eq!(a.free.len(), 1);
        assert_eq!(a.free[0].offset_bytes, 0);
        assert_eq!(a.free[0].len_bytes, 1024);
    }

    #[test]
    fn grow_appends_tail() {
        let mut a = RangeAllocator::new(1024, 256);
        let _ = a.allocate(1024).unwrap();
        a.free(Range {
            offset_bytes: 0,
            len_bytes: 1024,
        });
        a.grow_to(2048);
        let r = a.allocate(2048).unwrap();
        assert_eq!(r.offset_bytes, 0);
        assert_eq!(r.len_bytes, 2048);
    }

    #[test]
    fn fragmentation_returns_none() {
        let mut a = RangeAllocator::new(512, 256);
        let r0 = a.allocate(256).unwrap();
        let r1 = a.allocate(256).unwrap();
        a.free(r0);
        // need 512 aligned but only two holes of 256 non-adjacent
        assert!(a.allocate(512).is_none());
        a.free(r1);
        assert!(a.allocate(512).is_some());
    }

    #[test]
    fn first_element_index() {
        let r = Range {
            offset_bytes: 512,
            len_bytes: 256,
        };
        assert_eq!(r.first_element_index(16), 32);
    }
}
