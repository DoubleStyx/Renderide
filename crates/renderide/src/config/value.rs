//! Validating value algebra for renderer config: bounded numeric ranges plus power-of-two flooring
//! for bloom-style mip pyramids.
//!
//! ## Why
//!
//! Several config fields kept their raw user-supplied integer in the section struct, then
//! re-validated it on every read through a hand-written helper (clamp to `[MIN, MAX]` or round
//! down to a power of two for graph use). The helpers were structurally similar and scattered
//! across [`super::types::rendering`] and [`super::types::post_processing`].
//!
//! [`Clamped`] consolidates the clamp-then-extract step into one type-driven primitive, and
//! [`power_of_two_floor`] consolidates the bloom-style rounding step into one place. Field types
//! stay as raw `u32` (so `config.toml` keeps loading numeric literals), but the resolver methods
//! now return [`Clamped`] / use [`power_of_two_floor`] internally rather than reimplementing the
//! arithmetic.

use std::fmt;

/// A `u32` known to satisfy `MIN <= value <= MAX`. Construct via [`Clamped::new`], which clamps an
/// arbitrary input into the configured range.
///
/// `MIN` and `MAX` are const generics so the bounds are visible in error messages and the
/// returned value is structurally distinct from a plain `u32` at type-checking time. Callers
/// extract the raw value with [`Clamped::get`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct Clamped<const MIN: u32, const MAX: u32>(u32);

impl<const MIN: u32, const MAX: u32> Clamped<MIN, MAX> {
    /// Clamps `raw` into `[MIN, MAX]`.
    pub const fn new(raw: u32) -> Self {
        let v = if raw < MIN {
            MIN
        } else if raw > MAX {
            MAX
        } else {
            raw
        };
        Self(v)
    }

    /// Returns the underlying clamped value.
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl<const MIN: u32, const MAX: u32> fmt::Display for Clamped<MIN, MAX> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<const MIN: u32, const MAX: u32> From<Clamped<MIN, MAX>> for u32 {
    fn from(c: Clamped<MIN, MAX>) -> Self {
        c.get()
    }
}

/// Rounds a non-zero `u32` down to the nearest power of two (`0` and `1` both map to `1`).
///
/// Used by bloom mip-pyramid sizing: arbitrary continuous dimensions in `config.toml` and the
/// HUD slider get rounded to a power of two before the graph builds the pyramid so every
/// downsample rung is stable.
pub const fn power_of_two_floor(value: u32) -> u32 {
    let v = if value == 0 { 1 } else { value };
    1_u32 << (u32::BITS - v.leading_zeros() - 1)
}

#[cfg(test)]
mod tests {
    use super::{Clamped, power_of_two_floor};

    #[test]
    fn clamped_clamps_into_range() {
        type C = Clamped<1, 3>;
        assert_eq!(C::new(0).get(), 1);
        assert_eq!(C::new(1).get(), 1);
        assert_eq!(C::new(2).get(), 2);
        assert_eq!(C::new(3).get(), 3);
        assert_eq!(C::new(99).get(), 3);
    }

    #[test]
    fn power_of_two_floor_rounds_correctly() {
        assert_eq!(power_of_two_floor(0), 1);
        assert_eq!(power_of_two_floor(1), 1);
        assert_eq!(power_of_two_floor(2), 2);
        assert_eq!(power_of_two_floor(3), 2);
        assert_eq!(power_of_two_floor(4), 4);
        assert_eq!(power_of_two_floor(7), 4);
        assert_eq!(power_of_two_floor(64), 64);
        assert_eq!(power_of_two_floor(65), 64);
        assert_eq!(power_of_two_floor(127), 64);
        assert_eq!(power_of_two_floor(128), 128);
        assert_eq!(power_of_two_floor(2048), 2048);
        assert_eq!(power_of_two_floor(2049), 2048);
    }

    #[test]
    fn clamped_into_u32_extracts_value() {
        type C = Clamped<10, 20>;
        let c = C::new(15);
        let raw: u32 = c.into();
        assert_eq!(raw, 15);
    }
}
