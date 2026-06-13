//! Winit application driver state and event-loop integration.

mod events;
mod frame;
mod logging;
mod present;
mod shortcuts;
mod shutdown;
mod target;
mod xr;

use std::cell::RefCell;
use std::rc::Rc;

use logger::LogLevel;
use winit::event_loop::OwnedDisplayHandle;

use crate::crash_context;
use crate::frontend::input::{CursorOutputTracking, WindowInputAccumulator};
use crate::runtime::RendererRuntime;
use crate::xr::OpenxrHaptics;

use self::logging::LogFlushCadence;
use self::target::RenderTarget;
use self::xr::XrInputCache;
use super::bootstrap::{ExternalShutdownCoordinator, GpuStartupConfig};
use super::exit::ExitState;
use super::frame_clock::FrameClock;

/// Prefix for per-phase trace lines in the app frame tick.
const TICK_TRACE_PREFIX: &str = "renderide::tick";

/// Emits a trace line naming the current frame phase.
pub(super) fn tick_phase_trace(phase: &'static str) {
    crash_context::set_tick_phase_label(phase);
    logger::trace!("{} phase={phase}", TICK_TRACE_PREFIX);
}

/// Winit application handler for the renderer process.
pub(crate) struct AppDriver {
    pub(in crate::app::driver) runtime: RendererRuntime,
    pub(in crate::app::driver) startup_gpu: GpuStartupConfig,
    pub(in crate::app::driver) log_level_cli: Option<LogLevel>,
    pub(in crate::app::driver) target: Option<RenderTarget>,
    pub(in crate::app::driver) exit: Rc<RefCell<ExitState>>,
    pub(in crate::app::driver) log_flush: LogFlushCadence,
    pub(in crate::app::driver) shutdown: shutdown::GracefulShutdown,
    pub(in crate::app::driver) input: WindowInputAccumulator,
    pub(in crate::app::driver) cursor_output_tracking: CursorOutputTracking,
    pub(in crate::app::driver) frame_clock: FrameClock,
    /// Open Tracy span that attributes the winit wait gap to `app::about_to_wait`.
    pub(in crate::app::driver) about_to_wait_span: crate::profiling::DeferredCpuSpan,
    /// Whether the last completed tick queued an HMD projection layer for compositor pacing.
    pub(in crate::app::driver) hmd_compositor_paced_last_frame: bool,
    pub(in crate::app::driver) external_shutdown: Option<ExternalShutdownCoordinator>,
    pub(in crate::app::driver) main_heartbeat: Option<crate::diagnostics::Heartbeat>,
    /// RAII guard suppressing main-thread watchdog hang reports for the duration of the
    /// graceful shutdown drain. Acquired once when [`shutdown::GracefulShutdown::begin`]
    /// first succeeds; dropped when `AppDriver` itself drops after
    /// [`winit::event_loop::EventLoop::run_app`] returns. Keeps the watchdog quiet across
    /// the cooperative OpenXR exit handshake, target Drop, and GPU Drop sequence -- all of
    /// which are bounded by their own timeouts but legitimately stall the main thread past
    /// the watchdog's hang threshold.
    pub(in crate::app::driver) shutdown_watchdog_pause: Option<crate::diagnostics::WatchdogPause>,
    pub(in crate::app::driver) xr_input_cache: XrInputCache,
    pub(in crate::app::driver) xr_haptics: OpenxrHaptics,
    pub(in crate::app::driver) display_handle: OwnedDisplayHandle,
}

impl AppDriver {
    /// Builds initial app state after process bootstrap; window/GPU target creation is lazy.
    pub(crate) fn new(
        runtime: RendererRuntime,
        startup_gpu: GpuStartupConfig,
        log_level_cli: Option<LogLevel>,
        external_shutdown: Option<ExternalShutdownCoordinator>,
        main_heartbeat: Option<crate::diagnostics::Heartbeat>,
        display_handle: OwnedDisplayHandle,
        exit: Rc<RefCell<ExitState>>,
    ) -> Self {
        Self {
            runtime,
            startup_gpu,
            log_level_cli,
            target: None,
            exit,
            log_flush: LogFlushCadence::default(),
            shutdown: shutdown::GracefulShutdown::default(),
            input: WindowInputAccumulator::default(),
            cursor_output_tracking: CursorOutputTracking::default(),
            frame_clock: FrameClock::default(),
            about_to_wait_span: crate::profiling::DeferredCpuSpan::default(),
            hmd_compositor_paced_last_frame: false,
            external_shutdown,
            main_heartbeat,
            shutdown_watchdog_pause: None,
            xr_input_cache: XrInputCache::default(),
            xr_haptics: OpenxrHaptics::default(),
            display_handle,
        }
    }

    /// Returns whether an exit has already been requested.
    pub(crate) fn exit_is_requested(&self) -> bool {
        self.exit
            .try_borrow()
            .map(|exit_state| exit_state.is_requested())
            .unwrap_or(true)
    }
}
