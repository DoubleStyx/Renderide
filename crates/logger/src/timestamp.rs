//! UTC wall-clock timestamps for log file names (`YYYY-MM-DD_HH-MM-SS`) and per-line prefixes
//! (`HH:MM:SS.mmm`), derived from [`std::time::SystemTime`] without extra dependencies.

use std::time::{SystemTime, UNIX_EPOCH};

/// Returns a filename-safe UTC timestamp: `YYYY-MM-DD_HH-MM-SS`, used for log file names.
///
/// If [`SystemTime::now`] is before [`UNIX_EPOCH`], returns the literal `unknown`.
pub fn log_filename_timestamp() -> String {
    let Ok(dur) = SystemTime::now().duration_since(UNIX_EPOCH) else {
        return "unknown".to_string();
    };
    let secs = dur.as_secs();
    let day_secs = secs % 86400;
    let h = day_secs / 3600;
    let m = (day_secs / 60) % 60;
    let s = day_secs % 60;
    let (y, mo, d) = days_since_epoch_to_ymd(secs / 86400);
    format!("{y:04}-{mo:02}-{d:02}_{h:02}-{m:02}-{s:02}")
}

/// Returns a line-prefix timestamp: `HH:MM:SS.mmm` in UTC (derived from the Unix epoch wall time).
///
/// If wall time is before [`UNIX_EPOCH`], returns `"?"`.
pub(crate) fn format_line_timestamp() -> String {
    let Ok(dur) = SystemTime::now().duration_since(UNIX_EPOCH) else {
        return "?".to_string();
    };
    let secs = dur.as_secs();
    let millis = dur.subsec_millis();
    let mins = (secs / 60) % 60;
    let hours = (secs / 3600) % 24;
    let secs = secs % 60;
    format!("{hours:02}:{mins:02}:{secs:02}.{millis:03}")
}

/// Converts days since Unix epoch (1970-01-01) to `(year, month, day)`.
///
/// Algorithm: <http://howardhinnant.github.io/date_algorithms.html> `civil_from_days`.
///
/// `days` is whole days since 1970-01-01 UTC; the result is calendar `(year, month, day)`.
fn days_since_epoch_to_ymd(days: u64) -> (u32, u32, u32) {
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe.min(146_096) / 146_096) / 365;
    let y = (yoe as i64 + era * 400) as u32;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn days_since_epoch_unix_epoch_is_1970_01_01() {
        assert_eq!(days_since_epoch_to_ymd(0), (1970, 1, 1));
    }

    #[test]
    fn days_since_epoch_one_day_later() {
        assert_eq!(days_since_epoch_to_ymd(1), (1970, 1, 2));
    }

    #[test]
    fn days_since_epoch_to_ymd_known_dates() {
        for (days, ymd) in [
            (0_u64, (1970_u32, 1_u32, 1_u32)),
            (1, (1970, 1, 2)),
            (31, (1970, 2, 1)),
            (365, (1971, 1, 1)),
            (366, (1971, 1, 2)),
            (11_017, (2000, 3, 1)),
            (20_562, (2026, 4, 19)),
        ] {
            assert_eq!(days_since_epoch_to_ymd(days), ymd, "days={days}");
        }
    }

    #[test]
    fn format_line_timestamp_matches_pattern() {
        let s = format_line_timestamp();
        assert_eq!(s.len(), 12);
        assert_eq!(s.as_bytes()[2], b':');
        assert_eq!(s.as_bytes()[5], b':');
        assert_eq!(s.as_bytes()[8], b'.');
    }

    #[test]
    fn format_line_timestamp_zero_pads_all_components() {
        let s = format_line_timestamp();
        for (i, chunk) in [(0usize, 2usize), (3, 2), (6, 2), (9, 3)] {
            let slice = &s[i..i + chunk];
            assert!(
                slice.chars().all(|c| c.is_ascii_digit()),
                "expected digits in {slice:?} from {s:?}"
            );
        }
    }

    #[test]
    fn log_filename_timestamp_format() {
        let s = log_filename_timestamp();
        assert_eq!(s.len(), 19);
        assert!(s.contains('_'));
        let (date, time) = s.split_once('_').expect("timestamp contains underscore");
        assert_eq!(date.len(), 10);
        assert_eq!(time.len(), 8);
        assert!(date.chars().filter(|c| *c == '-').count() == 2);
        assert!(time.chars().filter(|c| *c == '-').count() == 2);
    }
}
