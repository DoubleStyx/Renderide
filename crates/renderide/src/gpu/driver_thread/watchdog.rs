//! Watchdog for blocking GPU driver-thread companion calls.
//!
//! Used by OpenXR calls that may block on the compositor so a stalled runtime surfaces in
//! `logs/renderer/*.log` instead of silently freezing the frame loop. The watchdog observes but
//! cannot interrupt blocking external calls.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

/// Arms a background thread that logs if [`Self::disarm`] is not called within `timeout`.
pub(crate) struct BlockingCallWatchdog {
    /// Sender carrying the instant when the guarded call completed.
    tx: Option<mpsc::Sender<Instant>>,
    /// Joined during [`Self::disarm`] so the worker exits before return.
    handle: Option<thread::JoinHandle<()>>,
}

impl BlockingCallWatchdog {
    /// Spawns the watchdog thread with a callback that fires once on timeout.
    pub(crate) fn arm_with_timeout_hook(
        timeout: Duration,
        label: &'static str,
        on_timeout: impl FnOnce() + Send + 'static,
    ) -> Self {
        Self::arm_inner(timeout, label, None, Some(Box::new(on_timeout)))
    }

    /// Spawns a shutdown-aware watchdog with a callback that fires once on timeout.
    pub(crate) fn arm_shutdown_aware_with_timeout_hook(
        timeout: Duration,
        label: &'static str,
        shutdown_requested: Arc<AtomicBool>,
        on_timeout: impl FnOnce() + Send + 'static,
    ) -> Self {
        Self::arm_inner(
            timeout,
            label,
            Some(shutdown_requested),
            Some(Box::new(on_timeout)),
        )
    }

    fn arm_inner(
        timeout: Duration,
        label: &'static str,
        shutdown_requested: Option<Arc<AtomicBool>>,
        on_timeout: Option<Box<dyn FnOnce() + Send + 'static>>,
    ) -> Self {
        let armed_at = Instant::now();
        let (tx, rx) = mpsc::channel::<Instant>();
        let handle = thread::Builder::new()
            .name(format!("gpu-blocking-call-watchdog:{label}"))
            .spawn(move || {
                run_watchdog(rx, armed_at, timeout, label, shutdown_requested, on_timeout);
            })
            .ok();
        Self {
            tx: Some(tx),
            handle,
        }
    }

    /// Signals the worker to exit and waits for it to observe completion.
    pub(crate) fn disarm(mut self) {
        self.signal_disarm();
    }

    fn signal_disarm(&mut self) {
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(Instant::now());
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn run_watchdog(
    rx: mpsc::Receiver<Instant>,
    armed_at: Instant,
    timeout: Duration,
    label: &'static str,
    shutdown_requested: Option<Arc<AtomicBool>>,
    mut on_timeout: Option<Box<dyn FnOnce() + Send + 'static>>,
) {
    let Some(deadline) = armed_at.checked_add(timeout) else {
        let _ = rx.recv();
        return;
    };

    if let Ok(disarmed_at) = rx.try_recv() {
        fire_if_disarm_missed_deadline(
            disarmed_at,
            deadline,
            label,
            timeout,
            shutdown_requested.as_deref(),
            &mut on_timeout,
        );
        return;
    }

    match rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
        Ok(disarmed_at) => {
            fire_if_disarm_missed_deadline(
                disarmed_at,
                deadline,
                label,
                timeout,
                shutdown_requested.as_deref(),
                &mut on_timeout,
            );
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {}
        Err(mpsc::RecvTimeoutError::Timeout) => {
            fire_watchdog_timeout(
                label,
                timeout,
                shutdown_requested.as_deref(),
                &mut on_timeout,
            );
            let _ = rx.recv();
        }
    }
}

fn fire_if_disarm_missed_deadline(
    disarmed_at: Instant,
    deadline: Instant,
    label: &'static str,
    timeout: Duration,
    shutdown_requested: Option<&AtomicBool>,
    on_timeout: &mut Option<Box<dyn FnOnce() + Send + 'static>>,
) {
    if disarmed_at >= deadline {
        fire_watchdog_timeout(label, timeout, shutdown_requested, on_timeout);
    }
}

fn fire_watchdog_timeout(
    label: &'static str,
    timeout: Duration,
    shutdown_requested: Option<&AtomicBool>,
    on_timeout: &mut Option<Box<dyn FnOnce() + Send + 'static>>,
) {
    log_watchdog_timeout(label, timeout, shutdown_requested);
    if let Some(on_timeout) = on_timeout.take() {
        on_timeout();
    }
}

fn log_watchdog_timeout(
    label: &'static str,
    timeout: Duration,
    shutdown_requested: Option<&AtomicBool>,
) {
    if shutdown_requested.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        logger::warn!(
            "{label} exceeded {}ms during shutdown -- external runtime may be stalled",
            timeout.as_millis()
        );
        return;
    }
    logger::error!(
        "{label} exceeded {}ms -- external runtime may be stalled",
        timeout.as_millis()
    );
}

impl Drop for BlockingCallWatchdog {
    fn drop(&mut self) {
        self.signal_disarm();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn instant_before_now(offset: Duration) -> Instant {
        let now = Instant::now();
        now.checked_sub(offset).unwrap_or(now)
    }

    #[test]
    fn disarm_before_timeout_does_not_block() {
        let start = Instant::now();
        let watchdog = BlockingCallWatchdog::arm_with_timeout_hook(
            Duration::from_secs(5),
            "test_disarm",
            || {},
        );
        watchdog.disarm();
        assert!(
            start.elapsed() < Duration::from_millis(500),
            "disarm should return promptly"
        );
    }

    #[test]
    fn drop_without_disarm_does_not_hang() {
        let start = Instant::now();
        {
            let _watchdog = BlockingCallWatchdog::arm_with_timeout_hook(
                Duration::from_secs(5),
                "test_drop",
                || {},
            );
        }
        assert!(
            start.elapsed() < Duration::from_millis(500),
            "drop should disarm promptly"
        );
    }

    #[test]
    fn timeout_fires_then_disarm_still_returns() {
        let watchdog = BlockingCallWatchdog::arm_with_timeout_hook(
            Duration::from_millis(10),
            "test_timeout",
            || {},
        );
        thread::sleep(Duration::from_millis(50));
        watchdog.disarm();
    }

    #[test]
    fn timeout_hook_fires_once() {
        let fired = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let fired_for_hook = Arc::clone(&fired);
        let watchdog = BlockingCallWatchdog::arm_with_timeout_hook(
            Duration::from_millis(10),
            "test_timeout_hook",
            move || {
                fired_for_hook.fetch_add(1, Ordering::Relaxed);
            },
        );
        thread::sleep(Duration::from_millis(50));
        watchdog.disarm();
        assert_eq!(fired.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn early_disarm_before_delayed_worker_start_does_not_fire() {
        let (tx, rx) = mpsc::channel();
        let armed_at = instant_before_now(Duration::from_millis(50));
        let disarmed_at = armed_at + Duration::from_millis(1);
        let fired = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let fired_for_hook = Arc::clone(&fired);

        assert!(tx.send(disarmed_at).is_ok());
        run_watchdog(
            rx,
            armed_at,
            Duration::from_millis(10),
            "test_early_disarm_before_delayed_worker_start",
            None,
            Some(Box::new(move || {
                fired_for_hook.fetch_add(1, Ordering::Relaxed);
            })),
        );

        assert_eq!(fired.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn late_disarm_before_delayed_worker_start_fires_once() {
        let (tx, rx) = mpsc::channel();
        let armed_at = instant_before_now(Duration::from_millis(50));
        let disarmed_at = armed_at + Duration::from_millis(20);
        let fired = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let fired_for_hook = Arc::clone(&fired);

        assert!(tx.send(disarmed_at).is_ok());
        run_watchdog(
            rx,
            armed_at,
            Duration::from_millis(10),
            "test_late_disarm_before_delayed_worker_start",
            None,
            Some(Box::new(move || {
                fired_for_hook.fetch_add(1, Ordering::Relaxed);
            })),
        );

        assert_eq!(fired.load(Ordering::Relaxed), 1);
    }
}
