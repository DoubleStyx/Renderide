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

/// Returns [`Duration`] for the initial IPC idle watchdog.
#[inline]
pub fn initial_heartbeat_timeout() -> Duration {
    Duration::from_secs(INITIAL_HEARTBEAT_TIMEOUT_SECS)
}

/// Returns [`Duration`] applied when a [`crate::protocol::HostCommand::Heartbeat`] is received.
#[inline]
pub fn heartbeat_refresh_timeout() -> Duration {
    Duration::from_secs(HEARTBEAT_REFRESH_TIMEOUT_SECS)
}

/// Returns [`Duration`] for heartbeat watchdog thread sleeps.
#[inline]
pub fn watchdog_poll_interval() -> Duration {
    Duration::from_millis(WATCHDOG_POLL_INTERVAL_MS)
}

/// Returns [`Duration`] between queue-loop log flushes.
#[inline]
pub fn queue_loop_flush_interval() -> Duration {
    Duration::from_secs(QUEUE_LOOP_FLUSH_INTERVAL_SECS)
}

/// Returns [`Duration`] between "waiting for Host" info logs.
#[inline]
pub fn queue_wait_log_interval() -> Duration {
    Duration::from_secs(QUEUE_WAIT_LOG_INTERVAL_SECS)
}

/// Returns [`Duration`] for Host exit watcher polling.
#[inline]
pub fn host_exit_watcher_poll_interval() -> Duration {
    Duration::from_secs(HOST_EXIT_WATCHER_POLL_INTERVAL_SECS)
}
