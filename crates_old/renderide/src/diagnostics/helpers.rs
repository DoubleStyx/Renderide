//! IPC drop throttling and value-change logging helpers used outside the debug HUD.

use std::time::{Duration, Instant};

/// Event emitted by [`ThrottledDropLog::record_drop`] when a log line should be written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DropLogEvent {
    /// First drop on this channel in the process lifetime.
    First {
        /// Byte length of the dropped payload.
        bytes: usize,
    },
    /// Additional drops since the last log, after the throttle interval elapsed.
    Burst {
        /// Number of dropped sends in this burst (excluding the separately logged first drop).
        count: u32,
        /// Sum of payload bytes in this burst.
        bytes: u64,
    },
}

/// Aggregates outbound IPC drops: log the first immediately, then at most one summary per interval.
pub struct ThrottledDropLog {
    interval: Duration,
    last_log: Option<Instant>,
    pending_count: u32,
    pending_bytes: u64,
    had_first: bool,
}

impl ThrottledDropLog {
    /// Creates a throttle with the given minimum interval between burst summaries.
    pub fn new(interval: Duration) -> Self {
        Self {
            interval,
            last_log: None,
            pending_count: 0,
            pending_bytes: 0,
            had_first: false,
        }
    }

    /// Records one dropped send of `bytes`. Returns an event to log, if any.
    pub fn record_drop(&mut self, bytes: usize) -> Option<DropLogEvent> {
        let now = Instant::now();
        if !self.had_first {
            self.had_first = true;
            self.last_log = Some(now);
            return Some(DropLogEvent::First { bytes });
        }
        self.pending_count = self.pending_count.saturating_add(1);
        self.pending_bytes = self.pending_bytes.saturating_add(bytes as u64);
        let last = self.last_log?;
        if now.duration_since(last) >= self.interval {
            let count = self.pending_count;
            let b = self.pending_bytes;
            self.pending_count = 0;
            self.pending_bytes = 0;
            self.last_log = Some(now);
            return Some(DropLogEvent::Burst { count, bytes: b });
        }
        None
    }
}

/// Remembers the last value and reports whether a new value differs.
#[derive(Debug, Default)]
pub struct LogOnChange<T: Eq> {
    last: Option<T>,
}

impl<T: Eq + Clone> LogOnChange<T> {
    /// Creates an empty tracker.
    pub fn new() -> Self {
        Self { last: None }
    }

    /// Returns `true` when `value` is different from the previously seen value (including first set).
    pub fn changed(&mut self, value: T) -> bool {
        if self.last.as_ref() == Some(&value) {
            return false;
        }
        self.last = Some(value);
        true
    }

    /// Clears the last value so the next `changed` call compares only against `None`.
    pub fn reset(&mut self) {
        self.last = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn throttled_first_drop_always_emits() {
        let mut t = ThrottledDropLog::new(Duration::from_secs(2));
        assert_eq!(t.record_drop(9), Some(DropLogEvent::First { bytes: 9 }));
    }

    #[test]
    fn throttled_second_drop_before_interval_does_not_emit() {
        let mut t = ThrottledDropLog::new(Duration::from_secs(60));
        let _ = t.record_drop(9);
        assert_eq!(t.record_drop(10), None);
        assert_eq!(t.record_drop(11), None);
    }

    #[test]
    fn throttled_burst_after_interval() {
        let mut t = ThrottledDropLog::new(Duration::ZERO);
        let _ = t.record_drop(9);
        let ev = t.record_drop(7).expect("burst");
        match ev {
            DropLogEvent::Burst { count, bytes } => {
                assert_eq!(count, 1);
                assert_eq!(bytes, 7);
            }
            DropLogEvent::First { .. } => panic!("expected burst"),
        }
    }

    #[test]
    fn log_on_change_first_and_repeat() {
        let mut c = LogOnChange::new();
        assert!(c.changed(1u32));
        assert!(!c.changed(1));
        assert!(c.changed(2));
        assert!(!c.changed(2));
    }

    #[test]
    fn log_on_change_reset() {
        let mut c = LogOnChange::new();
        assert!(c.changed(1u8));
        assert!(!c.changed(1));
        c.reset();
        assert!(c.changed(1));
    }
}
