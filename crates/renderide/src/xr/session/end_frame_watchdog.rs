//! Watchdog that logs when a wrapped OpenXR call exceeds a deadline.
//!
//! Wraps calls that may block on the compositor (`xrEndFrame`, `xrWaitSwapchainImage`) so a stalled
//! runtime surfaces in `logs/renderer/*.log` instead of silently freezing the frame loop. Normal
//! frame stalls are errors; shutdown stalls are warnings because the compositor may already be
//! unwinding the session. OpenXR has no per-call cancellation API, so the watchdog observes but
//! cannot interrupt.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

/// Arms a background thread that emits [`logger::error!`] if [`Self::disarm`] is not called within
/// `timeout`. Dropping the watchdog without calling [`Self::disarm`] also disarms it silently, so a
/// panic between arm and disarm still tears down the worker cleanly.
pub(crate) struct EndFrameWatchdog {
    /// Sender whose disconnect signals the worker to exit.
    tx: Option<mpsc::SyncSender<()>>,
    /// Joined during [`Self::disarm`] to guarantee the worker has observed the disconnect before we
    /// return to the caller; `None` after disarm.
    handle: Option<thread::JoinHandle<()>>,
}

impl EndFrameWatchdog {
    /// Spawns the watchdog thread. The returned guard must be [`Self::disarm`]ed within `timeout`
    /// or the worker will log one error and then wait for the final disconnect.
    pub(crate) fn arm(timeout: Duration, label: &'static str) -> Self {
        Self::arm_inner(timeout, label, None)
    }

    /// Spawns a watchdog that lowers stall severity after cooperative shutdown starts.
    pub(crate) fn arm_shutdown_aware(
        timeout: Duration,
        label: &'static str,
        shutdown_requested: Arc<AtomicBool>,
    ) -> Self {
        Self::arm_inner(timeout, label, Some(shutdown_requested))
    }

    fn arm_inner(
        timeout: Duration,
        label: &'static str,
        shutdown_requested: Option<Arc<AtomicBool>>,
    ) -> Self {
        let (tx, rx) = mpsc::sync_channel::<()>(0);
        let handle = thread::Builder::new()
            .name(format!("xr-end-frame-watchdog:{label}"))
            .spawn(move || match rx.recv_timeout(timeout) {
                Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => {}
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    log_watchdog_timeout(label, timeout, shutdown_requested.as_deref());
                    // Block until the caller finally disarms so the log line pairs with the eventual
                    // unblock; otherwise the operator sees an error with no follow-up.
                    let _ = rx.recv();
                }
            })
            .ok();
        Self {
            tx: Some(tx),
            handle,
        }
    }

    /// Signals the worker to exit and waits for it to observe the disconnect.
    pub(crate) fn disarm(mut self) {
        drop(self.tx.take());
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn log_watchdog_timeout(
    label: &'static str,
    timeout: Duration,
    shutdown_requested: Option<&AtomicBool>,
) {
    if shutdown_requested.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        logger::warn!(
            "xr::{label} exceeded {}ms during shutdown -- compositor may be stalled",
            timeout.as_millis()
        );
        return;
    }
    logger::error!(
        "xr::{label} exceeded {}ms -- compositor may be stalled",
        timeout.as_millis()
    );
}

impl Drop for EndFrameWatchdog {
    /// Disarms the watchdog if the caller forgot (e.g. a panic unwinds past [`Self::disarm`]).
    fn drop(&mut self) {
        drop(self.tx.take());
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn disarm_before_timeout_does_not_block() {
        let start = Instant::now();
        let wd = EndFrameWatchdog::arm(Duration::from_secs(5), "test_disarm");
        wd.disarm();
        assert!(
            start.elapsed() < Duration::from_millis(500),
            "disarm should return promptly"
        );
    }

    #[test]
    fn drop_without_disarm_does_not_hang() {
        let start = Instant::now();
        {
            let _wd = EndFrameWatchdog::arm(Duration::from_secs(5), "test_drop");
        }
        assert!(
            start.elapsed() < Duration::from_millis(500),
            "drop should disarm promptly"
        );
    }

    #[test]
    fn timeout_fires_then_disarm_still_returns() {
        let wd = EndFrameWatchdog::arm(Duration::from_millis(10), "test_timeout");
        thread::sleep(Duration::from_millis(50));
        wd.disarm();
    }
}
