//! UTC wall-clock timestamps for log file names and line prefixes.

use std::time::{SystemTime, UNIX_EPOCH};

/// Returns a filename-safe UTC timestamp: `YYYY-MM-DD_HH-MM-SS`. Used for log file names.
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

/// Returns a simple line-prefix timestamp: `HH:MM:SS.mmm` (UTC, from Unix epoch day fraction).
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
