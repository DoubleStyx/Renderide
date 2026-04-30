//! Float → Unity-domain integer conversions used by the material render-state resolver.

/// Rounds and clamps an `f32` Unity property value to a `u8` enum byte.
pub(super) fn unity_u8(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

/// Rounds and clamps an `f32` Unity stencil mask value into the `u32` mask space.
pub(super) fn unity_mask(v: f32) -> u32 {
    v.round().clamp(0.0, 255.0) as u32
}

/// Rounds and saturates an `f32` Unity `Offset units` value into `i32`.
pub(super) fn unity_offset_units(v: f32) -> i32 {
    v.round().clamp(i32::MIN as f32, i32::MAX as f32) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unity_u8_clamps_and_rounds() {
        assert_eq!(unity_u8(-10.0), 0);
        assert_eq!(unity_u8(0.4), 0);
        assert_eq!(unity_u8(0.6), 1);
        assert_eq!(unity_u8(254.7), 255);
        assert_eq!(unity_u8(1_000.0), 255);
    }

    #[test]
    fn unity_offset_units_saturates_at_i32_bounds() {
        assert_eq!(unity_offset_units(0.4), 0);
        assert_eq!(unity_offset_units(5.6), 6);
        assert_eq!(unity_offset_units(-5.6), -6);
        assert_eq!(unity_offset_units(1e12), i32::MAX);
        assert_eq!(unity_offset_units(-1e12), i32::MIN);
    }
}
