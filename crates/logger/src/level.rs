//! Log severity ordering, stable numeric tags for atomic max-level storage, and `-LogLevel` argv
//! scanning.

/// Log level for filtering. Lower ordinal = higher priority.
///
/// The discriminants are pinned via `#[repr(u8)]` because the global logger stores the active max
/// level in an [`std::sync::atomic::AtomicU8`]. `level as u8` is the canonical encoding and
/// [`tag_to_level`] decodes it back, saturating to [`LogLevel::Trace`] for unexpected bytes.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum LogLevel {
    /// Critical errors.
    Error = 0,
    /// Warnings.
    Warn = 1,
    /// Informational messages.
    Info = 2,
    /// Debug diagnostics.
    Debug = 3,
    /// Verbose trace.
    Trace = 4,
}

impl LogLevel {
    /// Every [`LogLevel`] variant in ascending severity order (Error first, Trace last).
    #[inline]
    pub const fn all() -> [Self; 5] {
        [
            Self::Error,
            Self::Warn,
            Self::Info,
            Self::Debug,
            Self::Trace,
        ]
    }

    /// Parses a level string (case-insensitive). Returns [`None`] for invalid values.
    ///
    /// Leading or trailing whitespace is **not** trimmed; use a trimmed string if the source may
    /// contain spaces.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "error" | "e" => Some(Self::Error),
            "warn" | "warning" | "w" => Some(Self::Warn),
            "info" | "i" => Some(Self::Info),
            "debug" | "d" => Some(Self::Debug),
            "trace" | "t" => Some(Self::Trace),
            _ => None,
        }
    }

    /// Returns the string to pass as `-LogLevel` value.
    pub const fn as_arg(&self) -> &'static str {
        match self {
            Self::Error => "error",
            Self::Warn => "warn",
            Self::Info => "info",
            Self::Debug => "debug",
            Self::Trace => "trace",
        }
    }

    /// Uppercase label written to log lines (`"ERROR"`, `"WARN"`, `"INFO"`, `"DEBUG"`, `"TRACE"`).
    ///
    /// Returning a `&'static str` lets the per-line formatter avoid re-running a match for
    /// [`std::fmt::Display`] on every log call and is shared by [`std::fmt::Display`] and
    /// [`std::fmt::Debug`].
    #[inline]
    pub(crate) const fn as_label(self) -> &'static str {
        match self {
            Self::Error => "ERROR",
            Self::Warn => "WARN",
            Self::Info => "INFO",
            Self::Debug => "DEBUG",
            Self::Trace => "TRACE",
        }
    }
}

impl std::fmt::Debug for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_label())
    }
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_label())
    }
}

/// Maps a stored `0..=4` tag back to [`LogLevel`]. Values above `4` clamp to [`LogLevel::Trace`].
///
/// The forward direction is `level as u8`; this decoder is needed because `AtomicU8` storage can
/// in principle hold any byte and the logger must remain robust to unexpected values.
#[inline]
pub fn tag_to_level(tag: u8) -> LogLevel {
    match tag.min(4) {
        0 => LogLevel::Error,
        1 => LogLevel::Warn,
        2 => LogLevel::Info,
        3 => LogLevel::Debug,
        _ => LogLevel::Trace,
    }
}

/// Scans `exe` then args for a case-insensitive `-LogLevel` flag followed by a level value.
///
/// If multiple `-LogLevel` flags appear, the **first** valid flag-value pair wins; remaining argv is
/// not scanned for overrides.
fn parse_loglevel_from_string_iter<I>(iter: I) -> Option<LogLevel>
where
    I: Iterator<Item = String>,
{
    let mut it = iter;
    while let Some(arg) = it.next() {
        if arg.eq_ignore_ascii_case("-LogLevel") {
            return it.next().and_then(|s| LogLevel::parse(&s));
        }
    }
    None
}

/// Parses `-LogLevel` from command line args (case-insensitive).
///
/// Returns [`None`] if not present or invalid; otherwise the parsed level.
///
/// Scans [`std::env::args`] without collecting argv into a [`Vec`].
pub fn parse_log_level_from_args() -> Option<LogLevel> {
    parse_loglevel_from_string_iter(std::env::args())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn parse_aliases_full_table() {
        for (s, expected) in [
            ("error", LogLevel::Error),
            ("e", LogLevel::Error),
            ("ERROR", LogLevel::Error),
            ("warn", LogLevel::Warn),
            ("warning", LogLevel::Warn),
            ("w", LogLevel::Warn),
            ("WaRn", LogLevel::Warn),
            ("info", LogLevel::Info),
            ("i", LogLevel::Info),
            ("debug", LogLevel::Debug),
            ("d", LogLevel::Debug),
            ("trace", LogLevel::Trace),
            ("t", LogLevel::Trace),
        ] {
            assert_eq!(LogLevel::parse(s), Some(expected), "token {s:?}");
        }
    }

    #[test]
    fn parse_rejects_empty_and_whitespace() {
        assert_eq!(LogLevel::parse(""), None);
        assert_eq!(LogLevel::parse("   "), None);
        assert_eq!(LogLevel::parse("warn "), None);
    }

    #[test]
    fn parse_rejects_unknown() {
        assert_eq!(LogLevel::parse("verbose"), None);
        assert_eq!(LogLevel::parse("5"), None);
    }

    #[test]
    fn as_arg_is_lowercase_and_round_trips_via_parse() {
        for level in LogLevel::all() {
            let s = level.as_arg();
            assert!(
                s.chars().all(|c| !c.is_uppercase()),
                "expected lowercase: {s}"
            );
            assert_eq!(LogLevel::parse(s), Some(level));
        }
    }

    #[test]
    fn log_level_ordering_matches_severity() {
        assert!(LogLevel::Error < LogLevel::Warn);
        assert!(LogLevel::Warn < LogLevel::Info);
        assert!(LogLevel::Trace > LogLevel::Debug);
    }

    #[test]
    fn log_level_debug_fmt() {
        assert_eq!(format!("{:?}", LogLevel::Error), "ERROR");
        assert_eq!(format!("{:?}", LogLevel::Trace), "TRACE");
    }

    #[test]
    fn log_level_display_matches_debug() {
        for level in LogLevel::all() {
            assert_eq!(format!("{level}"), format!("{level:?}"));
        }
    }

    #[test]
    fn as_label_matches_display() {
        for level in LogLevel::all() {
            assert_eq!(level.as_label(), format!("{level}"));
        }
    }

    #[test]
    fn discriminants_are_pinned() {
        assert_eq!(LogLevel::Error as u8, 0);
        assert_eq!(LogLevel::Warn as u8, 1);
        assert_eq!(LogLevel::Info as u8, 2);
        assert_eq!(LogLevel::Debug as u8, 3);
        assert_eq!(LogLevel::Trace as u8, 4);
    }

    #[test]
    fn level_as_u8_returns_distinct_increasing_tags() {
        let tags: Vec<u8> = LogLevel::all().iter().copied().map(|l| l as u8).collect();
        assert_eq!(tags, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn tag_to_level_clamps_above_max() {
        assert_eq!(tag_to_level(7), LogLevel::Trace);
    }

    #[test]
    fn parse_log_level_from_slice_finds_flag() {
        assert_eq!(
            parse_loglevel_from_string_iter(tokens(&["prog", "-LogLevel", "debug"]).into_iter(),),
            Some(LogLevel::Debug)
        );
    }

    #[test]
    fn parse_log_level_from_slice_case_insensitive_flag() {
        assert_eq!(
            parse_loglevel_from_string_iter(tokens(&["prog", "-loglevel", "INFO"]).into_iter(),),
            Some(LogLevel::Info)
        );
    }

    #[test]
    fn parse_log_level_from_slice_ignores_other_tokens() {
        assert_eq!(
            parse_loglevel_from_string_iter(
                tokens(&["prog", "-x", "-LogLevel", "warn", "y"]).into_iter(),
            ),
            Some(LogLevel::Warn)
        );
    }

    #[test]
    fn parse_log_level_from_slice_missing_value() {
        assert!(
            parse_loglevel_from_string_iter(tokens(&["prog", "-LogLevel"]).into_iter()).is_none()
        );
    }

    #[test]
    fn parse_log_level_from_slice_absent() {
        assert!(parse_loglevel_from_string_iter(tokens(&["prog", "a", "b"]).into_iter()).is_none());
    }

    #[test]
    fn parse_loglevel_from_string_iter_first_loglevel_wins() {
        assert_eq!(
            parse_loglevel_from_string_iter(
                tokens(&["p", "-LogLevel", "warn", "-LogLevel", "debug"]).into_iter(),
            ),
            Some(LogLevel::Warn)
        );
    }

    #[test]
    fn parse_loglevel_from_string_iter_consumes_value_after_first_flag() {
        assert_eq!(
            parse_loglevel_from_string_iter(
                tokens(&["p", "-LogLevel", "debug", "-LogLevel", "oops"]).into_iter(),
            ),
            Some(LogLevel::Debug)
        );
    }

    #[test]
    fn parse_loglevel_from_string_iter_invalid_value_returns_none() {
        assert!(
            parse_loglevel_from_string_iter(tokens(&["p", "-LogLevel", "nope"]).into_iter(),)
                .is_none()
        );
    }

    #[test]
    fn level_tag_roundtrip() {
        for l in LogLevel::all() {
            assert_eq!(tag_to_level(l as u8), l);
        }
    }
}
