//! Bit-packed boolean buffer over a `u32` slab, matching `Renderite.Shared.BitSpan`.
//!
//! The host allocates `SharedMemoryBufferDescriptor<uint>` regions (e.g.
//! [`crate::shared::MaterialsUpdateBatch::instance_changed_buffer`]) that the renderer must write
//! one bit per target into. Bit `i` lives in element `i / BITS_PER_ELEMENT` at bit position
//! `i % BITS_PER_ELEMENT` (LSB-first), identical to the C# `BitSpan` indexer.

/// Number of bits packed per `u32` element.
pub const BITS_PER_ELEMENT: usize = u32::BITS as usize;

/// Minimum `u32` slab length needed to hold `total_bits` bits.
#[inline]
pub const fn elements_for_bits(total_bits: usize) -> usize {
    total_bits.div_ceil(BITS_PER_ELEMENT)
}

/// Mutable bit-span view over a `u32` slab; `set` / `get` index by absolute bit position.
pub struct BitSpanMut<'a> {
    /// Backing storage; bit ordering matches `Renderite.Shared.BitSpan` (LSB-first within each element).
    pub data: &'a mut [u32],
}

impl<'a> BitSpanMut<'a> {
    /// Wraps the given slab; capacity in bits is `data.len() * BITS_PER_ELEMENT`.
    #[inline]
    pub const fn new(data: &'a mut [u32]) -> Self {
        Self { data }
    }

    /// Bit capacity of the underlying slab.
    #[inline]
    pub const fn len(&self) -> usize {
        self.data.len() * BITS_PER_ELEMENT
    }

    /// Returns `true` when the slab has no capacity.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Zeros every bit in the slab (use before writing fresh values into a reused shared region).
    #[inline]
    pub fn clear(&mut self) {
        for e in self.data.iter_mut() {
            *e = 0;
        }
    }

    /// Reads bit `bit_index`. Returns `false` when the index is out of range (matches a host
    /// reader that would also read past the end as zero rather than panic).
    #[inline]
    pub fn get(&self, bit_index: usize) -> bool {
        let element_index = bit_index / BITS_PER_ELEMENT;
        let bit_in_element = bit_index % BITS_PER_ELEMENT;
        match self.data.get(element_index) {
            Some(e) => (*e & (1u32 << bit_in_element)) != 0,
            None => false,
        }
    }

    /// Writes bit `bit_index`. Out-of-range writes are silently dropped so callers can pass a
    /// shorter buffer than expected without crashing the IPC ack path.
    #[inline]
    pub fn set(&mut self, bit_index: usize, value: bool) {
        let element_index = bit_index / BITS_PER_ELEMENT;
        let bit_in_element = bit_index % BITS_PER_ELEMENT;
        let Some(e) = self.data.get_mut(element_index) else {
            return;
        };
        let mask = 1u32 << bit_in_element;
        if value {
            *e |= mask;
        } else {
            *e &= !mask;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elements_for_bits_rounds_up() {
        assert_eq!(elements_for_bits(0), 0);
        assert_eq!(elements_for_bits(1), 1);
        assert_eq!(elements_for_bits(BITS_PER_ELEMENT), 1);
        assert_eq!(elements_for_bits(BITS_PER_ELEMENT + 1), 2);
        assert_eq!(elements_for_bits(2 * BITS_PER_ELEMENT), 2);
    }

    #[test]
    fn set_get_round_trip_within_one_element() {
        let mut data = [0u32; 1];
        let mut span = BitSpanMut::new(&mut data);
        span.set(0, true);
        span.set(5, true);
        span.set(31, true);
        assert!(span.get(0));
        assert!(!span.get(1));
        assert!(span.get(5));
        assert!(span.get(31));
        assert_eq!(data[0], 1u32 | (1u32 << 5) | (1u32 << 31));
    }

    #[test]
    fn boundary_index_lands_in_correct_element() {
        let mut data = [0u32; 3];
        let mut span = BitSpanMut::new(&mut data);
        // Bit 0 -> element 0 / bit 0
        span.set(0, true);
        // Bit 31 -> element 0 / bit 31
        span.set(31, true);
        // Bit 32 -> element 1 / bit 0
        span.set(32, true);
        // Bit 33 -> element 1 / bit 1
        span.set(33, true);
        // Bit 63 -> element 1 / bit 31
        span.set(63, true);
        // Bit 64 -> element 2 / bit 0
        span.set(64, true);
        assert_eq!(data[0], 1u32 | (1u32 << 31));
        assert_eq!(data[1], 1u32 | (1u32 << 1) | (1u32 << 31));
        assert_eq!(data[2], 1u32);
    }

    #[test]
    fn set_false_clears_bit() {
        let mut data = [0xFFFF_FFFFu32; 1];
        let mut span = BitSpanMut::new(&mut data);
        span.set(3, false);
        assert!(!span.get(3));
        assert_eq!(data[0], !(1u32 << 3));
    }

    #[test]
    fn out_of_range_set_is_dropped() {
        let mut data = [0u32; 1];
        let mut span = BitSpanMut::new(&mut data);
        span.set(64, true);
        assert!(!span.get(64));
        assert_eq!(data[0], 0);
    }

    #[test]
    fn clear_zeros_every_element() {
        let mut data = [0xDEAD_BEEFu32; 4];
        let mut span = BitSpanMut::new(&mut data);
        span.clear();
        assert!(data.iter().all(|&e| e == 0));
    }

    #[test]
    fn get_out_of_range_returns_false() {
        let mut data = [0xFFFF_FFFFu32; 1];
        let span = BitSpanMut::new(&mut data);
        assert!(!span.get(64));
    }
}
