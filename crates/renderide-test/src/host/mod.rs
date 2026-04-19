//! Mock-host harness: spawns the renderer in `--headless` mode, drives the full IPC + lockstep
//! handshake + sphere upload + scene submission, then returns the path of the freshly written PNG.
//!
//! The implementation is split across:
//!
//! - [`ipc_setup`] — opens the four authority Cloudtoid queues + tempdir for SHM backing files.
//! - [`handshake`] — `RendererInitData` → `RendererInitResult` → `RendererInitFinalizeData`.
//! - [`lockstep`] — drains both `…S` queues and replies to every `FrameStartData` with a
//!   `FrameSubmitData`. Per-frame counter starts at 0 (matches `RenderSystem.cs:111`).
//! - [`asset_upload`] — writes the sphere mesh into shared memory and waits for `MeshUploadResult`.
//! - [`scene_session`] — top-level orchestration.

use std::path::PathBuf;
use std::time::Duration;

use crate::error::HarnessError;

mod asset_upload;
mod handshake;
mod ipc_setup;
mod lockstep;
mod scene_session;

pub use scene_session::SceneSessionConfig;

/// Configuration for [`HostHarness::start`].
#[derive(Clone, Debug)]
pub struct HostHarnessConfig {
    /// Path to the `renderide` binary to spawn.
    pub renderer_path: PathBuf,
    /// Optional explicit PNG output path (overrides the default tempfile under the OS temp dir).
    pub forced_output_path: Option<PathBuf>,
    /// Offscreen render target width.
    pub width: u32,
    /// Offscreen render target height.
    pub height: u32,
    /// Renderer interval between consecutive PNG writes (ms).
    pub interval_ms: u64,
    /// Wall-clock budget for the entire pipeline (handshake + asset acks + first stable PNG).
    pub timeout: Duration,
    /// When `true`, inherit the renderer's stdout/stderr.
    pub verbose_renderer: bool,
}

/// Outcome of a successful harness run. Holds an optional tempdir guard so callers (e.g. the
/// `generate` subcommand) can read the PNG file before the directory is reaped.
#[derive(Debug)]
pub struct HarnessRunOutcome {
    /// Path to the freshly written PNG produced by the renderer.
    pub png_path: PathBuf,
    /// When the output path was auto-allocated under a tempdir, this guard keeps the directory
    /// alive until the outcome is dropped. Otherwise [`None`].
    pub _output_dir_guard: Option<tempfile::TempDir>,
}

/// Live harness state. The renderer process itself is owned by the underlying
/// [`SceneSessionConfig`] flow and exits via `RendererShutdownRequest` on success.
pub struct HostHarness {
    cfg: HostHarnessConfig,
    output_path: PathBuf,
    output_dir_guard: Option<tempfile::TempDir>,
}

impl HostHarness {
    /// Prepares an output PNG path (either the caller-supplied one or a tempfile) and stashes the
    /// configuration; the actual session runs in [`HostHarness::run`].
    pub fn start(cfg: HostHarnessConfig) -> Result<Self, HarnessError> {
        let (output_path, output_dir_guard) = match cfg.forced_output_path.clone() {
            Some(p) => (p, None),
            None => {
                let dir = tempfile::Builder::new()
                    .prefix("renderide-test-")
                    .tempdir()?;
                let path = dir.path().join("headless.png");
                (path, Some(dir))
            }
        };
        Ok(Self {
            cfg,
            output_path,
            output_dir_guard,
        })
    }

    /// Drives the full vertical slice end-to-end. Returns the PNG path on success and transfers
    /// the (optional) tempdir guard to the outcome so the file persists for downstream consumers.
    pub fn run(&mut self) -> Result<HarnessRunOutcome, HarnessError> {
        let session_cfg = SceneSessionConfig {
            renderer_path: self.cfg.renderer_path.clone(),
            output_path: self.output_path.clone(),
            width: self.cfg.width,
            height: self.cfg.height,
            interval_ms: self.cfg.interval_ms,
            timeout: self.cfg.timeout,
            verbose_renderer: self.cfg.verbose_renderer,
        };
        let outcome = scene_session::run_session(&session_cfg)?;
        Ok(HarnessRunOutcome {
            png_path: outcome.png_path,
            _output_dir_guard: self.output_dir_guard.take(),
        })
    }

    /// Output PNG path the renderer was instructed to write. Useful for callers that want to
    /// inspect or copy the file before [`HostHarness::run`] is called.
    #[allow(dead_code)]
    pub fn output_path(&self) -> &PathBuf {
        &self.output_path
    }
}

impl Drop for HostHarness {
    fn drop(&mut self) {
        let _ = self.output_dir_guard.take();
    }
}
