//! Dense bitset helpers for view-local draw preparation.

/// Compact growable bitset keyed by dense `usize` ordinals.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct DenseBitSet {
    words: Vec<u64>,
}

impl DenseBitSet {
    /// Clears all set bits while retaining allocation.
    #[inline]
    pub(super) fn clear(&mut self) {
        self.words.fill(0);
    }

    /// Clears the bitset and ensures it can address `bit_count` bits without reallocating later.
    #[inline]
    pub(super) fn clear_and_resize(&mut self, bit_count: usize) {
        let word_count = word_count_for_bits(bit_count);
        self.words.resize(word_count, 0);
        self.clear();
    }

    /// Inserts `bit`, growing the set if necessary. Returns `true` when the bit was not set.
    #[inline]
    pub(super) fn insert(&mut self, bit: usize) -> bool {
        self.ensure_bit(bit);
        let (word, mask) = bit_address(bit);
        let was_set = self.words[word] & mask != 0;
        self.words[word] |= mask;
        !was_set
    }

    /// Returns whether `bit` is set.
    #[inline]
    pub(super) fn contains(&self, bit: usize) -> bool {
        let (word, mask) = bit_address(bit);
        self.words
            .get(word)
            .is_some_and(|word_bits| *word_bits & mask != 0)
    }

    /// Number of set bits.
    #[inline]
    #[cfg(test)]
    pub(super) fn count_ones(&self) -> usize {
        self.words
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    /// Allocated capacity in addressable bits.
    #[inline]
    #[cfg(test)]
    pub(super) fn bit_capacity(&self) -> usize {
        self.words.len() * u64::BITS as usize
    }

    fn ensure_bit(&mut self, bit: usize) {
        let required = bit / u64::BITS as usize + 1;
        if required > self.words.len() {
            self.words.resize(required, 0);
        }
    }
}

#[inline]
fn word_count_for_bits(bit_count: usize) -> usize {
    bit_count.div_ceil(u64::BITS as usize)
}

#[inline]
fn bit_address(bit: usize) -> (usize, u64) {
    let word = bit / u64::BITS as usize;
    let mask = 1u64 << (bit % u64::BITS as usize);
    (word, mask)
}

#[cfg(test)]
mod tests {
    use super::DenseBitSet;

    #[test]
    fn insert_reports_new_bits_and_supports_lookup() {
        let mut bits = DenseBitSet::default();

        assert!(bits.insert(65));
        assert!(!bits.insert(65));
        assert!(bits.contains(65));
        assert!(!bits.contains(64));
        assert_eq!(bits.count_ones(), 1);
    }

    #[test]
    fn clear_retains_capacity() {
        let mut bits = DenseBitSet::default();
        bits.insert(100);
        let capacity = bits.bit_capacity();

        bits.clear();

        assert_eq!(bits.bit_capacity(), capacity);
        assert!(!bits.contains(100));
    }
}
