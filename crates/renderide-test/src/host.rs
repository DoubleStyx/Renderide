//! Mock-host harness: spawns the renderer in `--headless` mode, drives the full IPC + lockstep
//! handshake + sphere upload + scene submission, then returns the path of the freshly written PNG.
//!
//! The implementation is split across:
//!
//! - [`ipc_setup`] -- opens the four authority Cloudtoid queues + tempdir for SHM backing files.
//! - [`handshake`] -- `RendererInitData` -> `RendererInitResult` -> `RendererInitFinalizeData`.
//! - [`lockstep`] -- drains both `...S` queues and replies to every `FrameStartData` with a
//!   `FrameSubmitData`. Per-frame counter starts at 0 (matches `RenderSystem.cs:111`).
//! - [`asset_upload`] -- writes the sphere mesh into shared memory and waits for `MeshUploadResult`.
//! - [`scene_session`] -- top-level orchestration.

use std::path::PathBuf;
use std::time::Duration;

use crate::error::HarnessError;

mod asset_upload;
mod handshake;
pub mod ipc_setup;
pub mod lockstep;
pub mod scene_session;

pub use scene_session::{SceneSessionConfig, SessionTemplate};

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
    /// Procedural geometry template the harness should upload (sphere by default for the
    /// historical CLI flow).
    pub template: SessionTemplate,
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
    pub(crate) fn start(cfg: HostHarnessConfig) -> Result<Self, HarnessError> {
        let _ = crate::logging::init_renderer_test_logging()?;
        let (output_path, output_dir_guard) = if let Some(p) = cfg.forced_output_path.clone() {
            logger::info!("Harness: using forced PNG output path {}", p.display());
            (p, None)
        } else {
            let dir = tempfile::Builder::new()
                .prefix("renderide-test-")
                .tempdir()?;
            let path = dir.path().join("headless.png");
            logger::info!(
                "Harness: allocated temporary PNG output path {}",
                path.display()
            );
            (path, Some(dir))
        };
        logger::info!(
            "Harness: configured renderer_path={}, resolution={}x{}, interval_ms={}, timeout={:?}, verbose_renderer={}",
            cfg.renderer_path.display(),
            cfg.width,
            cfg.height,
            cfg.interval_ms,
            cfg.timeout,
            cfg.verbose_renderer
        );
        Ok(Self {
            cfg,
            output_path,
            output_dir_guard,
        })
    }

    /// Drives the full vertical slice end-to-end. Returns the PNG path on success and transfers
    /// the (optional) tempdir guard to the outcome so the file persists for downstream consumers.
    pub(crate) fn run(&mut self) -> Result<HarnessRunOutcome, HarnessError> {
        let session_cfg = SceneSessionConfig {
            renderer_path: self.cfg.renderer_path.clone(),
            output_path: self.output_path.clone(),
            width: self.cfg.width,
            height: self.cfg.height,
            interval_ms: self.cfg.interval_ms,
            timeout: self.cfg.timeout,
            verbose_renderer: self.cfg.verbose_renderer,
        };
        logger::info!(
            "Harness: starting scene session (output_path={})",
            session_cfg.output_path.display()
        );
        match scene_session::run_session(&session_cfg, self.cfg.template.clone()) {
            Ok(outcome) => {
                logger::info!(
                    "Harness: scene session completed (png_path={})",
                    outcome.png_path.display()
                );
                Ok(HarnessRunOutcome {
                    png_path: outcome.png_path,
                    _output_dir_guard: self.output_dir_guard.take(),
                })
            }
            Err(err) => {
                logger::error!("Harness: scene session failed: {err}");
                Err(err)
            }
        }
    }

    /// Output PNG path the renderer was instructed to write. Useful for callers that want to
    /// inspect or copy the file before [`HostHarness::run`] is called.
    #[cfg_attr(not(test), expect(dead_code, reason = "only used by unit tests today"))]
    pub(crate) const fn output_path(&self) -> &PathBuf {
        &self.output_path
    }
}

impl Drop for HostHarness {
    fn drop(&mut self) {
        let _ = self.output_dir_guard.take();
    }
}

#[cfg(test)]
mod harness_start_tests {
    use std::path::PathBuf;
    use std::time::Duration;

    use super::{HostHarness, HostHarnessConfig};

    fn minimal_config(forced_output_path: Option<PathBuf>) -> HostHarnessConfig {
        HostHarnessConfig {
            renderer_path: PathBuf::from("/nonexistent/renderide"),
            forced_output_path,
            width: 1,
            height: 1,
            interval_ms: 1,
            timeout: Duration::from_secs(1),
            verbose_renderer: false,
            template: super::SessionTemplate::Sphere,
        }
    }

    #[test]
    fn start_uses_forced_output_path_when_set() {
        let custom = PathBuf::from("/tmp/harness_forced_out.png");
        let h = HostHarness::start(minimal_config(Some(custom.clone()))).expect("start");
        assert_eq!(h.output_path(), &custom);
    }

    #[test]
    fn start_allocates_temp_headless_png_when_not_forced() {
        let h = HostHarness::start(minimal_config(None)).expect("start");
        let out = h.output_path();
        assert_eq!(
            out.file_name().and_then(|n| n.to_str()),
            Some("headless.png")
        );
        let parent = out.parent().expect("parent");
        let dir_name = parent
            .file_name()
            .expect("dir name")
            .to_string_lossy()
            .into_owned();
        assert!(
            dir_name.starts_with("renderide-test-"),
            "expected tempfile prefix, got {dir_name:?}"
        );
    }
}
