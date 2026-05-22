//! Configuration and outcome types for the scene-session orchestrator.

use std::path::PathBuf;
use std::time::Duration;

/// Configuration for a scene-session run.
#[derive(Clone, Debug)]
pub struct SceneSessionConfig {
    /// Path to the `renderide-renderer` binary to spawn.
    pub renderer_path: PathBuf,
    /// Output PNG path the renderer writes to (also where the harness reads from).
    pub output_path: PathBuf,
    /// Offscreen render target width.
    pub width: u32,
    /// Offscreen render target height.
    pub height: u32,
    /// Renderer interval between consecutive PNG writes (ms).
    pub interval_ms: u64,
    /// Wall-clock budget for the entire session (handshake + upload + first stable PNG).
    pub timeout: Duration,
    /// When `true`, inherit the renderer's stdout/stderr.
    pub verbose_renderer: bool,
}

/// Result of a successful scene-session run.
#[derive(Clone, Debug)]
pub(in crate::host) struct SceneSessionOutcome {
    /// Path to the freshly written PNG produced by the renderer.
    pub png_path: PathBuf,
}
