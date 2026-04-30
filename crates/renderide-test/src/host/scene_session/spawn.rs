//! Renderer process lifecycle: arg construction, spawn, RAII guard.

use std::path::Path;
use std::process::{Child, Command, Stdio};

use crate::error::HarnessError;

use super::super::ipc_setup::DEFAULT_QUEUE_CAPACITY_BYTES;
use super::config::SceneSessionConfig;

/// RAII-guarded spawned renderer process. [`Drop`] kills the child if still running.
pub(super) struct SpawnedRenderer {
    /// Live child process; `None` after a clean shutdown via the shutdown helper.
    pub child: Option<Child>,
}

impl Drop for SpawnedRenderer {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            logger::warn!("SpawnedRenderer: dropping with live child; killing");
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Spawns the renderer binary with all flags wired up for headless operation.
pub(super) fn spawn_renderer(
    cfg: &SceneSessionConfig,
    queue_name: &str,
    backing_dir: &Path,
) -> Result<SpawnedRenderer, HarnessError> {
    let mut cmd = Command::new(&cfg.renderer_path);
    let args = renderer_spawn_args(cfg, queue_name);
    cmd.args(&args);
    cmd.env("RENDERIDE_INTERPROCESS_DIR", backing_dir);

    if cfg.verbose_renderer {
        cmd.stdout(Stdio::inherit()).stderr(Stdio::inherit());
    } else {
        cmd.stdout(Stdio::null()).stderr(Stdio::null());
    }

    logger::info!(
        "Spawning renderer: {} {}",
        cfg.renderer_path.display(),
        args.join(" "),
    );

    let child = cmd.spawn().map_err(HarnessError::SpawnRenderer)?;
    Ok(SpawnedRenderer { child: Some(child) })
}

/// Builds the renderer process arguments for one harness session.
fn renderer_spawn_args(cfg: &SceneSessionConfig, queue_name: &str) -> Vec<String> {
    vec![
        "--headless".to_string(),
        "--headless-output".to_string(),
        cfg.output_path.display().to_string(),
        "--headless-resolution".to_string(),
        format!("{}x{}", cfg.width, cfg.height),
        "--headless-interval-ms".to_string(),
        cfg.interval_ms.to_string(),
        "-QueueName".to_string(),
        queue_name.to_string(),
        "-QueueCapacity".to_string(),
        DEFAULT_QUEUE_CAPACITY_BYTES.to_string(),
        "-LogLevel".to_string(),
        "debug".to_string(),
        "--ignore-config".to_string(),
    ]
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::Duration;

    use super::super::super::ipc_setup::DEFAULT_QUEUE_CAPACITY_BYTES;
    use super::{renderer_spawn_args, SceneSessionConfig};

    fn minimal_config() -> SceneSessionConfig {
        SceneSessionConfig {
            renderer_path: PathBuf::from("target/debug/renderide"),
            output_path: PathBuf::from("target/headless.png"),
            width: 64,
            height: 32,
            interval_ms: 250,
            timeout: Duration::from_secs(5),
            verbose_renderer: false,
        }
    }

    #[test]
    fn spawn_args_preserve_required_ipc_and_headless_values() {
        let args = renderer_spawn_args(&minimal_config(), "queue-a");
        let capacity = DEFAULT_QUEUE_CAPACITY_BYTES.to_string();
        assert_eq!(args[0], "--headless");
        assert!(args
            .windows(2)
            .any(|w| w == ["--headless-output", "target/headless.png"]));
        assert!(args
            .windows(2)
            .any(|w| w == ["--headless-resolution", "64x32"]));
        assert!(args
            .windows(2)
            .any(|w| w == ["--headless-interval-ms", "250"]));
        assert!(args.windows(2).any(|w| w == ["-QueueName", "queue-a"]));
        assert!(args
            .windows(2)
            .any(|w| w[0] == "-QueueCapacity" && w[1] == capacity));
    }
}
