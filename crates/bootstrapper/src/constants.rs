//! Shared timing constants for IPC watchdogs, the queue loop, and Host exit polling.

use std::time::Duration;

/// Idle timeout before the first Host message; extended on each [`crate::protocol::HostCommand::Heartbeat`].
pub const INITIAL_HEARTBEAT_TIMEOUT_SECS: u64 = 120;

/// Deadline refresh duration after each heartbeat from the Host.
pub const HEARTBEAT_REFRESH_TIMEOUT_SECS: u64 = 15;

/// Poll interval for the heartbeat watchdog thread.
pub const WATCHDOG_POLL_INTERVAL_MS: u64 = 250;

/// How often the queue loop calls [`logger::flush`].
pub const QUEUE_LOOP_FLUSH_INTERVAL_SECS: u64 = 1;

/// Interval between "still waiting for Host" log lines in the queue loop.
pub const QUEUE_WAIT_LOG_INTERVAL_SECS: u64 = 5;

/// Host process exit watcher polling interval.
pub const HOST_EXIT_WATCHER_POLL_INTERVAL_SECS: u64 = 1;

/// Poll interval for the renderer exit watcher (`Child::try_wait` loop).
pub const RENDERER_EXIT_WATCHER_POLL_INTERVAL_MS: u64 = 250;

/// macOS: grace period between `SIGINT` and `SIGTERM` during child shutdown escalation.
#[cfg(target_os = "macos")]
pub(crate) const MACOS_SHUTDOWN_SIGINT_TO_SIGTERM_DELAY_MS: u64 = 400;

/// macOS: grace period between `SIGTERM` and `SIGKILL` during child shutdown escalation.
#[cfg(target_os = "macos")]
pub(crate) const MACOS_SHUTDOWN_SIGTERM_TO_SIGKILL_DELAY_MS: u64 = 800;

/// Returns [`Duration`] for the initial IPC idle watchdog.
#[inline]
pub const fn initial_heartbeat_timeout() -> Duration {
    Duration::from_secs(INITIAL_HEARTBEAT_TIMEOUT_SECS)
}

/// Returns [`Duration`] applied when a [`crate::protocol::HostCommand::Heartbeat`] is received.
#[inline]
pub const fn heartbeat_refresh_timeout() -> Duration {
    Duration::from_secs(HEARTBEAT_REFRESH_TIMEOUT_SECS)
}

/// Returns [`Duration`] for heartbeat watchdog thread sleeps.
#[inline]
pub const fn watchdog_poll_interval() -> Duration {
    Duration::from_millis(WATCHDOG_POLL_INTERVAL_MS)
}

/// Returns [`Duration`] between queue-loop log flushes.
#[inline]
pub const fn queue_loop_flush_interval() -> Duration {
    Duration::from_secs(QUEUE_LOOP_FLUSH_INTERVAL_SECS)
}

/// Returns [`Duration`] between "waiting for Host" info logs.
#[inline]
pub const fn queue_wait_log_interval() -> Duration {
    Duration::from_secs(QUEUE_WAIT_LOG_INTERVAL_SECS)
}

/// Returns [`Duration`] for Host exit watcher polling.
#[inline]
pub const fn host_exit_watcher_poll_interval() -> Duration {
    Duration::from_secs(HOST_EXIT_WATCHER_POLL_INTERVAL_SECS)
}

/// Returns [`Duration`] between renderer `Child::try_wait` polls.
#[inline]
pub const fn renderer_exit_watcher_poll_interval() -> Duration {
    Duration::from_millis(RENDERER_EXIT_WATCHER_POLL_INTERVAL_MS)
}

/// Returns the macOS `SIGINT` -> `SIGTERM` shutdown grace period.
#[cfg(target_os = "macos")]
#[inline]
pub(crate) fn macos_shutdown_sigint_to_sigterm_delay() -> Duration {
    Duration::from_millis(MACOS_SHUTDOWN_SIGINT_TO_SIGTERM_DELAY_MS)
}

/// Returns the macOS `SIGTERM` -> `SIGKILL` shutdown grace period.
#[cfg(target_os = "macos")]
#[inline]
pub(crate) fn macos_shutdown_sigterm_to_sigkill_delay() -> Duration {
    Duration::from_millis(MACOS_SHUTDOWN_SIGTERM_TO_SIGKILL_DELAY_MS)
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

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_shutdown_helpers_match_constants() {
        assert_eq!(
            macos_shutdown_sigint_to_sigterm_delay().as_millis(),
            u128::from(MACOS_SHUTDOWN_SIGINT_TO_SIGTERM_DELAY_MS)
        );
        assert_eq!(
            macos_shutdown_sigterm_to_sigkill_delay().as_millis(),
            u128::from(MACOS_SHUTDOWN_SIGTERM_TO_SIGKILL_DELAY_MS)
        );
    }
}
