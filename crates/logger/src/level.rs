//! Log severity ordering, stable numeric tags for atomic max-level storage, and `-LogLevel` argv
//! scanning.

/// Log level for filtering. Lower ordinal = higher priority.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    /// Critical errors.
    Error,
    /// Warnings.
    Warn,
    /// Informational messages.
    Info,
    /// Debug diagnostics.
    Debug,
    /// Verbose trace.
    Trace,
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
    pub fn as_arg(&self) -> &'static str {
        match self {
            Self::Error => "error",
            Self::Warn => "warn",
            Self::Info => "info",
            Self::Debug => "debug",
            Self::Trace => "trace",
        }
    }
}

impl std::fmt::Debug for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Error => write!(f, "ERROR"),
            Self::Warn => write!(f, "WARN"),
            Self::Info => write!(f, "INFO"),
            Self::Debug => write!(f, "DEBUG"),
            Self::Trace => write!(f, "TRACE"),
        }
    }
}

/// Stable `0..=4` tag for [`LogLevel`] (matches [`PartialOrd`] order).
#[inline]
pub(crate) fn level_to_tag(level: LogLevel) -> u8 {
    match level {
        LogLevel::Error => 0,
        LogLevel::Warn => 1,
        LogLevel::Info => 2,
        LogLevel::Debug => 3,
        LogLevel::Trace => 4,
    }
}

/// Maps a stored `0..=4` tag back to [`LogLevel`]. Values above `4` clamp to [`LogLevel::Trace`].
#[inline]
pub(crate) fn tag_to_level(tag: u8) -> LogLevel {
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
/// If multiple `-LogLevel` flags appear, the **first** valid flag–value pair wins; remaining argv is
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
    fn log_level_parse_aliases() {
        assert_eq!(LogLevel::parse("error"), Some(LogLevel::Error));
        assert_eq!(LogLevel::parse("E"), Some(LogLevel::Error));
        assert_eq!(LogLevel::parse("WARN"), Some(LogLevel::Warn));
        assert_eq!(LogLevel::parse("trace"), Some(LogLevel::Trace));
        assert_eq!(LogLevel::parse("bogus"), None);
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
    fn as_arg_is_lowercase_for_each_level() {
        for level in LogLevel::all() {
            let s = level.as_arg();
            assert!(s.chars().all(|c| !c.is_uppercase()));
            assert_eq!(LogLevel::parse(s), Some(level));
        }
    }

    #[test]
    fn log_level_as_arg_roundtrip() {
        for level in LogLevel::all() {
            assert_eq!(LogLevel::parse(level.as_arg()), Some(level));
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
    fn level_to_tag_returns_distinct_increasing_tags() {
        let tags: Vec<u8> = LogLevel::all().iter().copied().map(level_to_tag).collect();
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
            assert_eq!(tag_to_level(level_to_tag(l)), l);
        }
    }
}
