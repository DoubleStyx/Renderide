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

/// Formats byte counts for dense allocator tables (B / KiB / MiB / GiB / TiB).
pub fn bytes_compact(bytes: u64) -> String {
    const SUFFIX: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut idx = 0usize;
    let mut amount = bytes as f64;
    while amount >= 1024.0 && idx < SUFFIX.len() - 1 {
        amount /= 1024.0;
        idx += 1;
    }
    format!("{amount:.2} {}", SUFFIX[idx])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hud_fmt_produces_stable_field_width() {
        assert_eq!(f64_field(8, 2, 1.0).len(), 8);
        assert_eq!(f64_field(8, 2, 123.456).len(), 8);
    }

    #[test]
    fn bytes_compact_zero() {
        assert_eq!(bytes_compact(0), "0.00 B");
    }

    #[test]
    fn bytes_compact_boundaries() {
        assert_eq!(bytes_compact(1023), "1023.00 B");
        assert_eq!(bytes_compact(1024), "1.00 KiB");
        assert_eq!(bytes_compact(1024 * 1024), "1.00 MiB");
        assert_eq!(bytes_compact(1024_u64.pow(3)), "1.00 GiB");
        assert_eq!(bytes_compact(1024_u64.pow(4)), "1.00 TiB");
        // Saturates at TiB (largest suffix).
        assert_eq!(bytes_compact(1024_u64.pow(5)), "1024.00 TiB");
    }

    #[test]
    fn gib_value_zero_and_exact_gib() {
        assert_eq!(gib_value(6, 2, 0), "  0.00");
        assert_eq!(gib_value(6, 2, 1024_u64.pow(3)), "  1.00");
        assert_eq!(gib_value(6, 2, 2 * 1024_u64.pow(3)), "  2.00");
    }

    #[test]
    fn f64_field_overflows_width_when_value_too_wide() {
        // Width is a minimum: oversized values are not truncated.
        let s = f64_field(4, 2, 12345.678);
        assert!(s.len() >= 4);
        assert!(s.trim_start().starts_with("12345.68"));
    }
}
