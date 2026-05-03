//! Cooperative renderer hang/hitch detection.
//!
//! # Why
//!
//! The existing fatal-crash handler ([`crate::fatal_crash_log`]) covers signals, SEH, and Mach
//! exceptions, but a stuck render thread that does not panic produces no log line and no exit
//! code -- only a frozen window. The watchdog adds a second observer thread that, when the main
//! thread (or any other registered thread) fails to update its heartbeat within the configured
//! deadline, captures a stack trace of the stuck thread and emits a hang report.
//!
//! # Design
//!
//! - **Single pet site per thread, per loop iteration.** Mirrors Unreal's `FThreadHeartBeat`.
//!   Long-but-legitimate stalls (initial pipeline compile, `xrWaitFrame`, swapchain reconfigure)
//!   are bracketed with [`WatchdogPause`] rather than sprinkling extra pets through the frame.
//! - **Two thresholds**: a *hitch* threshold ([`crate::config::WatchdogSettings::hitch_threshold_ms`])
//!   produces a `warn` log line; a *hang* threshold ([`crate::config::WatchdogSettings::hang_threshold_ms`])
//!   captures stacks and emits an `error` line.
//! - **Stack capture** uses `pthread_kill`+`SIGUSR2` on Linux/macOS so the watchdog thread can
//!   ask the stuck thread to walk its own stack into a static buffer. Symbolication runs back on
//!   the watchdog thread (heap-allocating, off the signal context). Windows currently logs the
//!   hang without a stack trace; cross-platform parity is tracked as follow-up work.
//! - **Action**: [`crate::config::WatchdogAction::LogAndContinue`] (default) keeps the process
//!   alive after the report; [`crate::config::WatchdogAction::LogAndAbort`] calls
//!   [`std::process::abort`] so a supervisor (the bootstrapper) can restart cleanly.

mod registry;
#[cfg(any(target_os = "linux", target_os = "android", target_os = "macos"))]
mod signal;
mod thread;

pub use registry::{Heartbeat, WatchdogPause};
pub use thread::Watchdog;
