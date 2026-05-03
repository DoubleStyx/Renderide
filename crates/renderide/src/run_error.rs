//! Fatal failures encountered while starting the renderer (before or instead of a normal event-loop exit).

use std::io;

use thiserror::Error;
use winit::error::EventLoopError;

use crate::connection::InitError as ConnectionInitError;
use crate::gpu::GpuError;

/// Startup or early abort before the winit loop returns an optional process exit code.
#[derive(Debug, Error)]
pub enum RunError {
    /// Singleton guard, IPC connect, or other [`ConnectionInitError`] from bootstrap.
    #[error(transparent)]
    Connection(#[from] ConnectionInitError),
    /// File logging could not be initialized (see `logger::init_for`).
    #[error("failed to initialize logging: {0}")]
    LoggingInit(#[from] io::Error),
    /// The host did not send [`crate::shared::RendererInitData`](crate::shared::RendererInitData) within the startup timeout.
    #[error("timed out waiting for RendererInitData from host")]
    RendererInitDataTimeout,
    /// IPC reported a fatal error while waiting for init data.
    #[error("fatal IPC error while waiting for RendererInitData")]
    RendererInitDataFatalIpc,
    /// [`winit`] could not create the event loop (display backend unavailable, etc.).
    #[error(transparent)]
    EventLoopCreate(#[from] EventLoopError),
    /// [`crate::gpu::GpuContext`] initialization (desktop or headless) failed.
    #[error("GPU init: {0}")]
    Gpu(#[from] GpuError),
}

#[cfg(test)]
mod tests {
    use super::RunError;

    /// Constant-string variants have stable [`std::fmt::Display`] output used by the process exit
    /// log line -- regressions here change what operators see in `logs/renderer/*.log`.
    #[test]
    fn timeout_and_fatal_ipc_variants_have_stable_display_messages() {
        assert_eq!(
            RunError::RendererInitDataTimeout.to_string(),
            "timed out waiting for RendererInitData from host"
        );
        assert_eq!(
            RunError::RendererInitDataFatalIpc.to_string(),
            "fatal IPC error while waiting for RendererInitData"
        );
    }

    /// The [`std::io::Error`] conversion is wired via [`thiserror`] `#[from]`, so any
    /// [`std::io::Error`] can propagate into the logging-init branch without explicit mapping.
    #[test]
    fn logging_init_prefix_and_io_from_impl() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "no write");
        let run_err: RunError = io_err.into();
        let rendered = run_err.to_string();
        assert!(
            rendered.starts_with("failed to initialize logging:"),
            "expected logging prefix, got {rendered:?}"
        );
        assert!(
            rendered.contains("no write"),
            "io source message missing: {rendered:?}"
        );
        assert!(matches!(run_err, RunError::LoggingInit(_)));
    }
}
