//! Watchdog thread: spawns on [`Watchdog::install`], polls heartbeat slots, emits hitch / hang
//! reports, and joins on [`Drop`].
//!
//! Per iteration the thread snapshots the registry, calls [`super::registry::evaluate_slot`] on
//! each slot, and dispatches `warn` (hitch) or `error` (hang) lines through the file logger.
//! For hangs on POSIX it asks the stuck thread for a backtrace via [`super::signal`] and
//! includes the symbolicated trace in the report.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use parking_lot::RwLock;

use crate::config::{WatchdogAction, WatchdogSettings};

use super::registry::{
    Heartbeat, HeartbeatRegistry, HeartbeatSlot, SlotEvaluation, evaluate_slot,
    record_hang_reported, record_hitch_reported,
};
#[cfg(any(target_os = "linux", target_os = "android", target_os = "macos"))]
use super::signal;

/// Process-global watchdog. Owns the background poll thread and the heartbeat registry.
///
/// Construct via [`Self::install`] from `app::run` and keep the returned [`Watchdog`]
/// alive for the lifetime of the renderer process -- when it drops, the poll thread joins.
pub struct Watchdog {
    inner: Arc<WatchdogInner>,
    join: Option<JoinHandle<()>>,
}

struct WatchdogInner {
    registry: HeartbeatRegistry,
    /// Effective settings cloned at install time. Live edits to `RendererSettings.watchdog`
    /// from the ImGui config window are not picked up automatically; restart the renderer to
    /// apply them. (The watchdog deliberately captures its own copy so the poll loop never
    /// races with config reloads.)
    settings: RwLock<WatchdogSettings>,
    /// Set by [`Watchdog::drop`]; the poll loop exits on the next iteration.
    shutdown: AtomicBool,
}

impl Watchdog {
    /// Install the watchdog according to `settings`. Returns `None` (and spawns nothing) when
    /// `settings.enabled` is `false`.
    ///
    /// Failures spawning the watchdog thread or installing the capture signal handler are
    /// logged and treated as "watchdog unavailable" -- startup continues without it.
    pub fn install(settings: WatchdogSettings) -> Option<Self> {
        if !settings.enabled {
            logger::info!("Watchdog disabled by configuration");
            return None;
        }
        #[cfg(any(target_os = "linux", target_os = "android", target_os = "macos"))]
        if let Err(e) = signal::install() {
            logger::warn!(
                "Watchdog: failed to install SIGUSR2 capture handler ({e}); hang reports will omit stack traces"
            );
        }

        let inner = Arc::new(WatchdogInner {
            registry: HeartbeatRegistry::new(),
            settings: RwLock::new(settings),
            shutdown: AtomicBool::new(false),
        });
        let inner_for_thread = Arc::clone(&inner);
        let join = match std::thread::Builder::new()
            .name("renderide-watchdog".to_owned())
            .spawn(move || run_watchdog(inner_for_thread))
        {
            Ok(j) => j,
            Err(e) => {
                logger::error!("Watchdog: failed to spawn poll thread: {e}");
                return None;
            }
        };
        let poll_ms = inner.settings.read().poll_interval_ms;
        let hitch_ms = inner.settings.read().hitch_threshold_ms;
        let hang_ms = inner.settings.read().hang_threshold_ms;
        logger::info!("Watchdog installed: poll={poll_ms}ms hitch={hitch_ms}ms hang={hang_ms}ms");
        Some(Self {
            inner,
            join: Some(join),
        })
    }

    /// Register a heartbeat for the calling thread using the watchdog's configured hitch / hang
    /// thresholds.
    ///
    /// The returned [`Heartbeat`] should be stored on the watched thread's stack-rooted state
    /// (e.g. the app driver); calling [`Heartbeat::pet`] is the per-iteration liveness signal.
    pub fn register_current_thread(&self, name: &'static str) -> Heartbeat {
        let s = self.inner.settings.read();
        let hitch = Duration::from_millis(u64::from(s.hitch_threshold_ms));
        let hang = Duration::from_millis(u64::from(s.hang_threshold_ms));
        drop(s);
        let os_tid = current_os_tid();
        let pthread_handle = current_pthread_handle();
        let slot = self
            .inner
            .registry
            .register(name, os_tid, pthread_handle, hitch, hang);
        Heartbeat::from_slot(slot)
    }
}

impl Drop for Watchdog {
    fn drop(&mut self) {
        self.inner.shutdown.store(true, Ordering::Release);
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

/// Returns the calling thread's OS identity, or `0` on platforms without a stack-capture path.
fn current_os_tid() -> i64 {
    #[cfg(any(target_os = "linux", target_os = "android", target_os = "macos"))]
    {
        signal::current_os_tid()
    }
    #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "macos")))]
    {
        0
    }
}

/// Returns the calling thread's macOS `pthread_t` encoded as `usize`, or `0` when unused.
fn current_pthread_handle() -> usize {
    #[cfg(any(target_os = "linux", target_os = "android", target_os = "macos"))]
    {
        signal::current_pthread_handle()
    }
    #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "macos")))]
    {
        0
    }
}

/// Watchdog poll loop entry point.
fn run_watchdog(inner: Arc<WatchdogInner>) {
    let poll_interval =
        Duration::from_millis(u64::from(inner.settings.read().poll_interval_ms.max(10)));
    while !inner.shutdown.load(Ordering::Acquire) {
        for slot in inner.registry.snapshot() {
            handle_slot(&inner, &slot);
            if inner.shutdown.load(Ordering::Acquire) {
                return;
            }
        }
        std::thread::sleep(poll_interval);
    }
}

/// Inspect one slot and emit a hitch / hang report if applicable.
fn handle_slot(inner: &WatchdogInner, slot: &Arc<HeartbeatSlot>) {
    let now_ns = slot.now_ns();
    match evaluate_slot(slot, now_ns) {
        SlotEvaluation::Quiet => {}
        SlotEvaluation::Hitch {
            elapsed_ns,
            pet_value,
        } => {
            let elapsed_ms = elapsed_ns / 1_000_000;
            logger::warn!(
                "Watchdog: thread '{}' hitch -- last pet {elapsed_ms} ms ago (threshold {} ms)",
                slot.name,
                slot.hitch_ns / 1_000_000
            );
            record_hitch_reported(slot, pet_value);
        }
        SlotEvaluation::Hang {
            elapsed_ns,
            pet_value,
        } => {
            let elapsed_ms = elapsed_ns / 1_000_000;
            let trace = capture_hang_trace(slot);
            match trace {
                Some(t) => logger::error!(
                    "Watchdog: thread '{}' HANG -- last pet {elapsed_ms} ms ago (threshold {} ms)\n{t}",
                    slot.name,
                    slot.hang_ns / 1_000_000
                ),
                None => logger::error!(
                    "Watchdog: thread '{}' HANG -- last pet {elapsed_ms} ms ago (threshold {} ms); stack capture unavailable",
                    slot.name,
                    slot.hang_ns / 1_000_000
                ),
            }
            // Flush so the report is on disk before any subsequent abort.
            logger::flush();
            record_hang_reported(slot, pet_value);

            let action = inner.settings.read().action;
            if matches!(action, WatchdogAction::LogAndAbort) {
                logger::error!("Watchdog: WatchdogAction::LogAndAbort -- aborting process");
                logger::flush();
                std::process::abort();
            }
        }
    }
}

/// Capture a symbolicated stack trace of the stuck thread on supported platforms.
fn capture_hang_trace(slot: &HeartbeatSlot) -> Option<String> {
    #[cfg(any(target_os = "linux", target_os = "android", target_os = "macos"))]
    {
        let frames =
            signal::request_capture(slot.os_tid, slot.pthread_handle, Duration::from_millis(500))?;
        Some(signal::symbolicate(&frames))
    }
    #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "macos")))]
    {
        let _ = slot;
        None
    }
}
