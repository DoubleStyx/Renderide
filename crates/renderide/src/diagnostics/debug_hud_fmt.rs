//! Right-aligned numeric [`format!`] helpers so HUD columns keep a stable width.

/// Formats `value` as a right-aligned decimal with `decimals` places and total width `width`.
pub fn f64_field(width: usize, decimals: usize, value: f64) -> String {
    format!("{value:>w$.d$}", w = width, d = decimals)
}

/// Human-readable gibibytes from bytes (numeric part only; caller adds `GiB` suffix).
pub fn gib_value(width: usize, decimals: usize, bytes: u64) -> String {
    let g = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    f64_field(width, decimals, g)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hud_fmt_produces_stable_field_width() {
        assert_eq!(f64_field(8, 2, 1.0).len(), 8);
        assert_eq!(f64_field(8, 2, 123.456).len(), 8);
    }
}
