//! Normal renderer-process exits shared by windowed and headless app drivers.

/// Normal renderer-process exit result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RunExit {
    /// The renderer completed without requesting a specific process exit code.
    Clean,
    /// The renderer completed with an explicit process exit code.
    Code(i32),
}

impl RunExit {
    /// Returns the explicit process code when this exit carries one.
    pub const fn code(self) -> Option<i32> {
        match self {
            Self::Clean => None,
            Self::Code(code) => Some(code),
        }
    }

    /// Returns the process code that should be passed to `std::process::exit`.
    pub const fn process_code(self) -> i32 {
        match self {
            Self::Clean => 0,
            Self::Code(code) => code,
        }
    }
}

/// Windowed-driver reason for leaving the event loop.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ExitReason {
    /// User or shell requested the main window close.
    WindowClosed,
    /// The OS signal / console-control handler requested cooperative shutdown.
    ExternalShutdown,
    /// OpenXR asked the application to exit.
    OpenxrExit,
    /// The host requested renderer shutdown over IPC.
    HostShutdown,
    /// IPC entered a fatal state.
    FatalIpc,
    /// Winit could not create the main window.
    WindowCreateFailed,
    /// Desktop GPU initialization failed.
    DesktopGpuInitFailed,
    /// OpenXR initialization failed.
    OpenxrInitFailed,
    /// The OpenXR device could not create the desktop mirror surface.
    OpenxrMirrorSurfaceFailed,
}

impl ExitReason {
    /// Maps the reason to the externally visible normal exit result.
    pub(crate) const fn run_exit(self) -> RunExit {
        match self {
            Self::FatalIpc => RunExit::Code(4),
            Self::WindowCreateFailed | Self::DesktopGpuInitFailed => RunExit::Code(1),
            Self::OpenxrInitFailed | Self::OpenxrMirrorSurfaceFailed => RunExit::Code(2),
            Self::WindowClosed | Self::ExternalShutdown | Self::OpenxrExit | Self::HostShutdown => {
                RunExit::Clean
            }
        }
    }

    /// Whether this exit reason should run the cooperative renderer shutdown drain.
    pub(crate) const fn uses_graceful_shutdown(self) -> bool {
        matches!(
            self,
            Self::WindowClosed | Self::ExternalShutdown | Self::OpenxrExit | Self::HostShutdown
        )
    }
}

/// First normal exit request observed by the app driver.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ExitRequest {
    reason: ExitReason,
    run_exit: RunExit,
}

impl ExitRequest {
    /// Builds an exit request from the source reason.
    pub(crate) const fn from_reason(reason: ExitReason) -> Self {
        Self {
            reason,
            run_exit: reason.run_exit(),
        }
    }

    /// Source reason for diagnostics and tests.
    pub(crate) const fn reason(self) -> ExitReason {
        self.reason
    }

    /// Normal run exit result.
    pub(crate) const fn run_exit(self) -> RunExit {
        self.run_exit
    }
}

/// Tracks whether the app has requested event-loop exit.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ExitState {
    request: Option<ExitRequest>,
}

impl ExitState {
    /// Records the first exit request and returns it.
    pub(crate) fn request(&mut self, reason: ExitReason) -> ExitRequest {
        if let Some(request) = self.request {
            return request;
        }
        let request = ExitRequest::from_reason(reason);
        self.request = Some(request);
        request
    }

    /// Returns whether an exit has already been requested.
    pub(crate) const fn is_requested(&self) -> bool {
        self.request.is_some()
    }

    /// Returns the eventual normal run exit, defaulting to clean process completion.
    pub(crate) const fn run_exit(&self) -> RunExit {
        match self.request {
            Some(request) => request.run_exit(),
            None => RunExit::Clean,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ExitReason, ExitState, RunExit};

    #[test]
    fn clean_exit_has_no_explicit_code_and_process_zero() {
        assert_eq!(RunExit::Clean.code(), None);
        assert_eq!(RunExit::Clean.process_code(), 0);
    }

    #[test]
    fn coded_exit_reports_code_and_process_code() {
        let exit = RunExit::Code(4);
        assert_eq!(exit.code(), Some(4));
        assert_eq!(exit.process_code(), 4);
    }

    #[test]
    fn exit_state_keeps_first_request() {
        let mut state = ExitState::default();
        let first = state.request(ExitReason::HostShutdown);
        let second = state.request(ExitReason::FatalIpc);
        assert_eq!(first.reason(), ExitReason::HostShutdown);
        assert_eq!(second.reason(), ExitReason::HostShutdown);
        assert_eq!(state.run_exit(), RunExit::Clean);
    }

    #[test]
    fn fatal_ipc_maps_to_code_four() {
        assert_eq!(ExitReason::FatalIpc.run_exit(), RunExit::Code(4));
    }

    #[test]
    fn openxr_startup_failures_map_to_code_two() {
        assert_eq!(ExitReason::OpenxrInitFailed.run_exit(), RunExit::Code(2));
        assert_eq!(
            ExitReason::OpenxrMirrorSurfaceFailed.run_exit(),
            RunExit::Code(2)
        );
    }

    #[test]
    fn only_normal_runtime_exits_use_graceful_shutdown() {
        assert!(ExitReason::WindowClosed.uses_graceful_shutdown());
        assert!(ExitReason::ExternalShutdown.uses_graceful_shutdown());
        assert!(ExitReason::OpenxrExit.uses_graceful_shutdown());
        assert!(ExitReason::HostShutdown.uses_graceful_shutdown());
        assert!(!ExitReason::FatalIpc.uses_graceful_shutdown());
        assert!(!ExitReason::OpenxrInitFailed.uses_graceful_shutdown());
    }
}
