//! Graceful windowed-driver shutdown coordination.

use std::time::{Duration, Instant};

/// Maximum time the winit driver will keep polling OpenXR shutdown before leaving the event loop.
const GRACEFUL_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(2);
/// Delay between shutdown polls while waiting for OpenXR lifecycle events or deferred finalizers.
const GRACEFUL_SHUTDOWN_POLL_INTERVAL: Duration = Duration::from_millis(10);

/// Small state machine for the renderer's cooperative shutdown drain.
#[derive(Debug)]
pub(super) struct GracefulShutdown {
    started_at: Option<Instant>,
    openxr_exit_requested: bool,
    timeout: Duration,
    poll_interval: Duration,
}

impl Default for GracefulShutdown {
    fn default() -> Self {
        Self {
            started_at: None,
            openxr_exit_requested: false,
            timeout: GRACEFUL_SHUTDOWN_TIMEOUT,
            poll_interval: GRACEFUL_SHUTDOWN_POLL_INTERVAL,
        }
    }
}

impl GracefulShutdown {
    /// Starts the shutdown drain. Returns `true` only on the first call.
    pub(super) fn begin(&mut self, now: Instant) -> bool {
        if self.started_at.is_some() {
            return false;
        }
        self.started_at = Some(now);
        true
    }

    /// Whether shutdown draining has started.
    pub(super) const fn is_started(&self) -> bool {
        self.started_at.is_some()
    }

    /// Whether the OpenXR session has already received `xrRequestExitSession`.
    pub(super) const fn openxr_exit_requested(&self) -> bool {
        self.openxr_exit_requested
    }

    /// Marks that `xrRequestExitSession` was attempted.
    pub(super) fn mark_openxr_exit_requested(&mut self) {
        self.openxr_exit_requested = true;
    }

    /// Returns whether the shutdown drain exceeded its bounded wait.
    pub(super) fn timed_out(&self, now: Instant) -> bool {
        self.started_at
            .is_some_and(|started_at| now.duration_since(started_at) >= self.timeout)
    }

    /// Configured shutdown timeout.
    pub(super) const fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Poll cadence while the drain is pending.
    pub(super) const fn poll_interval(&self) -> Duration {
        self.poll_interval
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::GracefulShutdown;

    #[test]
    fn begin_only_reports_first_start() {
        let mut shutdown = GracefulShutdown::default();
        let now = Instant::now();
        assert!(shutdown.begin(now));
        assert!(!shutdown.begin(now + Duration::from_millis(1)));
        assert!(shutdown.is_started());
    }

    #[test]
    fn timeout_uses_start_instant() {
        let mut shutdown = GracefulShutdown::default();
        let now = Instant::now();
        shutdown.begin(now);
        let just_before_timeout = (now + shutdown.timeout())
            .checked_sub(Duration::from_millis(1))
            .expect("test timeout is larger than one millisecond");
        assert!(!shutdown.timed_out(just_before_timeout));
        assert!(shutdown.timed_out(now + shutdown.timeout()));
    }

    #[test]
    fn openxr_exit_request_is_tracked_once_marked() {
        let mut shutdown = GracefulShutdown::default();
        assert!(!shutdown.openxr_exit_requested());
        shutdown.mark_openxr_exit_requested();
        assert!(shutdown.openxr_exit_requested());
    }
}
