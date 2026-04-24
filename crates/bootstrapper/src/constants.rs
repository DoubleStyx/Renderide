//! Shared timing constants for IPC watchdogs, the queue loop, and Host exit polling.

use std::time::Duration;

/// Idle timeout before the first Host message; extended on each [`crate::protocol::HostCommand::Heartbeat`].
pub(crate) const INITIAL_HEARTBEAT_TIMEOUT_SECS: u64 = 120;

/// Deadline refresh duration after each heartbeat from the Host.
pub(crate) const HEARTBEAT_REFRESH_TIMEOUT_SECS: u64 = 15;

/// Poll interval for the heartbeat watchdog thread.
pub(crate) const WATCHDOG_POLL_INTERVAL_MS: u64 = 250;

/// How often the queue loop calls [`logger::flush`].
pub(crate) const QUEUE_LOOP_FLUSH_INTERVAL_SECS: u64 = 1;

/// Interval between "still waiting for Host" log lines in the queue loop.
pub(crate) const QUEUE_WAIT_LOG_INTERVAL_SECS: u64 = 5;

/// Host process exit watcher polling interval.
pub(crate) const HOST_EXIT_WATCHER_POLL_INTERVAL_SECS: u64 = 1;

/// Poll interval for the renderer exit watcher (`Child::try_wait` loop).
pub(crate) const RENDERER_EXIT_WATCHER_POLL_INTERVAL_MS: u64 = 250;

/// Returns [`Duration`] for the initial IPC idle watchdog.
#[inline]
pub(crate) fn initial_heartbeat_timeout() -> Duration {
    Duration::from_secs(INITIAL_HEARTBEAT_TIMEOUT_SECS)
}

/// Returns [`Duration`] applied when a [`crate::protocol::HostCommand::Heartbeat`] is received.
#[inline]
pub(crate) fn heartbeat_refresh_timeout() -> Duration {
    Duration::from_secs(HEARTBEAT_REFRESH_TIMEOUT_SECS)
}

/// Returns [`Duration`] for heartbeat watchdog thread sleeps.
#[inline]
pub(crate) fn watchdog_poll_interval() -> Duration {
    Duration::from_millis(WATCHDOG_POLL_INTERVAL_MS)
}

/// Returns [`Duration`] between queue-loop log flushes.
#[inline]
pub(crate) fn queue_loop_flush_interval() -> Duration {
    Duration::from_secs(QUEUE_LOOP_FLUSH_INTERVAL_SECS)
}

/// Returns [`Duration`] between "waiting for Host" info logs.
#[inline]
pub(crate) fn queue_wait_log_interval() -> Duration {
    Duration::from_secs(QUEUE_WAIT_LOG_INTERVAL_SECS)
}

/// Returns [`Duration`] for Host exit watcher polling.
#[inline]
pub(crate) fn host_exit_watcher_poll_interval() -> Duration {
    Duration::from_secs(HOST_EXIT_WATCHER_POLL_INTERVAL_SECS)
}

/// Returns [`Duration`] between renderer `Child::try_wait` polls.
#[inline]
pub(crate) fn renderer_exit_watcher_poll_interval() -> Duration {
    Duration::from_millis(RENDERER_EXIT_WATCHER_POLL_INTERVAL_MS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_helpers_match_constants() {
        assert_eq!(
            initial_heartbeat_timeout().as_secs(),
            INITIAL_HEARTBEAT_TIMEOUT_SECS
        );
        assert_eq!(
            heartbeat_refresh_timeout().as_secs(),
            HEARTBEAT_REFRESH_TIMEOUT_SECS
        );
        assert_eq!(
            watchdog_poll_interval().as_millis(),
            u128::from(WATCHDOG_POLL_INTERVAL_MS)
        );
        assert_eq!(
            queue_loop_flush_interval().as_secs(),
            QUEUE_LOOP_FLUSH_INTERVAL_SECS
        );
        assert_eq!(
            queue_wait_log_interval().as_secs(),
            QUEUE_WAIT_LOG_INTERVAL_SECS
        );
        assert_eq!(
            host_exit_watcher_poll_interval().as_secs(),
            HOST_EXIT_WATCHER_POLL_INTERVAL_SECS
        );
        assert_eq!(
            renderer_exit_watcher_poll_interval().as_millis(),
            u128::from(RENDERER_EXIT_WATCHER_POLL_INTERVAL_MS)
        );
    }
}
