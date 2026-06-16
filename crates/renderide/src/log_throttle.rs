//! Small counters for non-spammy repeated diagnostic logs.

use std::sync::atomic::{AtomicU64, Ordering};

/// Thread-safe first-occurrence plus periodic log gate.
pub(crate) struct LogThrottle {
    occurrences: AtomicU64,
}

impl LogThrottle {
    /// Creates a throttle with no recorded occurrences.
    pub(crate) const fn new() -> Self {
        Self {
            occurrences: AtomicU64::new(0),
        }
    }

    /// Returns the occurrence number when a caller should log.
    ///
    /// `first` controls how many initial occurrences pass through. After that, every `every`
    /// occurrences pass through. Passing `every == 0` disables periodic logging after the first
    /// window.
    pub(crate) fn should_log(&self, first: u64, every: u64) -> Option<u64> {
        let occurrence = self.occurrences.fetch_add(1, Ordering::Relaxed) + 1;
        if occurrence <= first {
            return Some(occurrence);
        }
        if every == 0 {
            return None;
        }
        (occurrence - first)
            .is_multiple_of(every)
            .then_some(occurrence)
    }

    /// Returns the total number of recorded occurrences.
    #[cfg(test)]
    pub(crate) fn occurrences(&self) -> u64 {
        self.occurrences.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::LogThrottle;

    #[test]
    fn logs_first_window_then_every_interval() {
        let throttle = LogThrottle::new();
        let hits = (0..9)
            .filter_map(|_| throttle.should_log(2, 3))
            .collect::<Vec<_>>();
        assert_eq!(hits, vec![1, 2, 5, 8]);
        assert_eq!(throttle.occurrences(), 9);
    }

    #[test]
    fn zero_interval_disables_periodic_hits() {
        let throttle = LogThrottle::new();
        let hits = (0..5)
            .filter_map(|_| throttle.should_log(1, 0))
            .collect::<Vec<_>>();
        assert_eq!(hits, vec![1]);
        assert_eq!(throttle.occurrences(), 5);
    }
}
