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
