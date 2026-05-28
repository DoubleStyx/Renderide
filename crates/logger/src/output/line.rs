//! Log line formatting and reusable per-thread buffers.

use std::cell::RefCell;
use std::fmt::Write as _;

use crate::level::LogLevel;
use crate::timestamp::write_line_timestamp;

/// Default capacity reserved on a thread's reusable line buffer so that steady-state log calls
/// avoid reallocation.
const LINE_BUF_INITIAL_CAPACITY: usize = 256;

thread_local! {
    /// Per-thread reusable buffer for log line formatting. Cleared on every successful borrow so a
    /// panic mid-format leaves no observable corruption for the next caller.
    static LINE_BUF: RefCell<String> = const { RefCell::new(String::new()) };
}

/// Writes a full log line into `out` in the canonical `[HH:MM:SS.mmm] [TARGET] LEVEL message\n`
/// shape.
///
/// `out` is cleared first so the buffer can be reused across calls without observable carry-over.
pub(super) fn format_log_line_into(
    out: &mut String,
    target: &str,
    level: LogLevel,
    args: std::fmt::Arguments<'_>,
) {
    out.clear();
    if out.capacity() < LINE_BUF_INITIAL_CAPACITY {
        out.reserve(LINE_BUF_INITIAL_CAPACITY - out.capacity());
    }
    out.push('[');
    write_line_timestamp(out);
    out.push_str("] [");
    out.push_str(target);
    out.push_str("] ");
    out.push_str(level.as_label());
    out.push(' ');
    let _ = out.write_fmt(args);
    out.push('\n');
}

/// Calls `f` with a thread-local reusable line buffer when available, otherwise with a
/// stack-managed fallback so a [`std::fmt::Display`] impl that recursively logs cannot panic on a
/// borrow conflict. If the thread-local has been destroyed (only possible during thread teardown),
/// returns `R::default()` so the logger remains a no-op rather than panicking.
pub(super) fn with_line_buf<R: Default>(f: impl FnOnce(&mut String) -> R) -> R {
    LINE_BUF
        .try_with(|cell| {
            if let Ok(mut buf) = cell.try_borrow_mut() {
                f(&mut buf)
            } else {
                let mut fallback = String::with_capacity(LINE_BUF_INITIAL_CAPACITY);
                f(&mut fallback)
            }
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies reused buffers are cleared before formatting fresh content.
    #[test]
    fn format_log_line_into_clears_existing_buffer_content() {
        let mut buf = String::from("stale_should_be_overwritten");
        format_log_line_into(
            &mut buf,
            "renderide",
            LogLevel::Info,
            format_args!("fresh_line"),
        );
        assert!(!buf.contains("stale_should_be_overwritten"));
        assert!(buf.contains(" [renderide] INFO fresh_line\n"));
        assert!(buf.starts_with('['));
    }

    /// Verifies target-aware formatting preserves the caller-supplied target.
    #[test]
    fn format_log_line_into_preserves_custom_target() {
        let mut buf = String::new();
        format_log_line_into(
            &mut buf,
            "renderide::asset",
            LogLevel::Debug,
            format_args!("loaded"),
        );

        assert!(buf.contains(" [renderide::asset] DEBUG loaded\n"));
    }

    /// Verifies new buffers reserve the steady-state initial capacity.
    #[test]
    fn format_log_line_into_grows_capacity_to_initial() {
        let mut buf = String::new();
        format_log_line_into(&mut buf, "renderide", LogLevel::Trace, format_args!("x"));
        assert!(buf.capacity() >= LINE_BUF_INITIAL_CAPACITY);
    }

    /// Verifies the canonical `[ts] LEVEL message\n` shape: bracketed timestamp prefix, level
    /// label surrounded by spaces, original message body, and a trailing newline.
    #[test]
    fn format_log_line_into_emits_bracketed_timestamp_level_and_newline() {
        let mut buf = String::new();
        format_log_line_into(
            &mut buf,
            "renderide",
            LogLevel::Info,
            format_args!("hello_world"),
        );
        assert!(buf.starts_with('['), "expected leading '[' in {buf:?}");
        assert!(buf.contains("] "), "expected '] ' separator in {buf:?}");
        assert!(
            buf.contains(" [renderide] "),
            "expected ' [renderide] ' token in {buf:?}"
        );
        assert!(buf.contains(" INFO "), "expected ' INFO ' token in {buf:?}");
        assert!(buf.contains("hello_world"), "expected message in {buf:?}");
        assert!(buf.ends_with('\n'), "expected trailing newline in {buf:?}");
        assert_eq!(
            buf.matches('\n').count(),
            1,
            "expected exactly one newline in {buf:?}"
        );
    }

    /// Verifies every [`LogLevel`] variant emits its [`LogLevel::as_label`] token surrounded by
    /// spaces. Guards against accidental label drift in the formatter.
    #[test]
    fn format_log_line_into_uses_correct_label_for_each_level() {
        let mut buf = String::new();
        for level in LogLevel::all() {
            format_log_line_into(&mut buf, "renderide", level, format_args!("body"));
            let token = format!(" {} body", level.as_label());
            assert!(
                buf.contains(&token),
                "expected token {token:?} in line for {level:?}: {buf:?}"
            );
        }
    }

    /// Verifies an empty message still produces a well-formed line ending in `LEVEL \n` so
    /// downstream parsers can rely on the level-then-space invariant.
    #[test]
    fn format_log_line_into_handles_empty_message() {
        let mut buf = String::new();
        format_log_line_into(&mut buf, "renderide", LogLevel::Warn, format_args!(""));
        assert!(buf.starts_with('['));
        assert!(buf.ends_with(" WARN \n"), "got {buf:?}");
        assert_eq!(buf.matches('\n').count(), 1);
    }

    /// Pins the documented behavior that newlines inside the formatted message are written
    /// verbatim. If a future change introduces sanitization this test will fail and force the
    /// decision to be deliberate.
    #[test]
    fn format_log_line_into_preserves_embedded_newlines() {
        let mut buf = String::new();
        format_log_line_into(
            &mut buf,
            "renderide",
            LogLevel::Error,
            format_args!("first\nsecond"),
        );
        assert!(buf.contains("first\nsecond"), "got {buf:?}");
        assert!(buf.ends_with('\n'));
        assert_eq!(
            buf.matches('\n').count(),
            2,
            "expected embedded + trailing newline: {buf:?}"
        );
    }

    /// Verifies recursive logging falls back to a stack-owned buffer instead of panicking on a
    /// nested mutable borrow of the thread-local buffer.
    #[test]
    fn with_line_buf_uses_fallback_when_thread_local_buffer_is_already_borrowed() {
        let len = LINE_BUF.with(|cell| {
            let _borrow = cell.borrow_mut();
            with_line_buf(|buf| {
                buf.push_str("fallback");
                buf.len()
            })
        });

        assert_eq!(len, "fallback".len());
    }
}
