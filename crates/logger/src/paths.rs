//! Resolves `Renderide/logs/<component>/`, applies the `RENDERIDE_LOGS_ROOT` override, and wires
//! [`init_for`] to the global file sink in [`crate::output`].

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use crate::level::LogLevel;
use crate::output;

/// Environment variable that overrides the default `Renderide/logs` root directory.
pub(crate) const LOGS_ROOT_ENV: &str = "RENDERIDE_LOGS_ROOT";

/// Failure to resolve the default `Renderide/logs` root from a crate manifest path.
#[derive(Debug, thiserror::Error)]
pub enum LogsRootError {
    /// `manifest_dir` did not have enough ancestors to reach the workspace `Renderide/` directory.
    #[error(
        "logger manifest path must live under .../Renderide/crates/logger (need 3+ path segments); got {manifest_dir:?}"
    )]
    ManifestPathTooShort {
        /// Path passed to [`logs_root_with`].
        manifest_dir: PathBuf,
    },
}

/// Which part of the system produces a log stream under [`logs_root`] / `<component>/`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LogComponent {
    /// Bootstrapper process (Rust).
    Bootstrapper,
    /// Host process output captured by the bootstrapper (stdout/stderr into one file).
    Host,
    /// Renderer process (Rust).
    Renderer,
}

impl LogComponent {
    /// Subdirectory name under `logs/` for this component.
    pub fn subdir(self) -> &'static str {
        match self {
            Self::Bootstrapper => "bootstrapper",
            Self::Host => "host",
            Self::Renderer => "renderer",
        }
    }
}

impl std::fmt::Display for LogComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.subdir())
    }
}

/// Resolves where all Renderide logs live, for use in tests without touching process environment.
///
/// If `override_root` is [`Some`], that path is used as the logs root (same role as the
/// `RENDERIDE_LOGS_ROOT` environment variable). Otherwise `manifest_dir` must be
/// `.../Renderide/crates/logger`;
/// [`Path::ancestors`] yields `logger` → `crates` → `Renderide`, so index `2` is the repository root.
pub fn logs_root_with(
    manifest_dir: &Path,
    override_root: Option<&OsStr>,
) -> Result<PathBuf, LogsRootError> {
    if let Some(root) = override_root {
        return Ok(PathBuf::from(root));
    }
    let renderide_root =
        manifest_dir
            .ancestors()
            .nth(2)
            .ok_or_else(|| LogsRootError::ManifestPathTooShort {
                manifest_dir: manifest_dir.to_path_buf(),
            })?;
    Ok(renderide_root.join("logs"))
}

/// Root directory containing per-component folders (`bootstrapper`, `host`, `renderer`).
///
/// By default this is `Renderide/logs` next to the workspace `crates/` directory. If the
/// `RENDERIDE_LOGS_ROOT` environment variable is set, that path is used instead (no subdirectory
/// insertion; you get exactly that folder as the root for all components).
pub fn logs_root() -> PathBuf {
    logs_root_with(
        Path::new(env!("CARGO_MANIFEST_DIR")),
        std::env::var_os(LOGS_ROOT_ENV).as_deref(),
    )
    .unwrap_or_else(|e| {
        // Can't route through the logger — this is the logger bootstrap path.
        #[expect(
            clippy::print_stderr,
            reason = "logger not yet initialized at bootstrap"
        )]
        {
            eprintln!("Renderide logger: {e}; using fallback logs directory");
        }
        std::env::current_dir()
            .map(|p| p.join("logs"))
            .unwrap_or_else(|_| PathBuf::from("logs"))
    })
}

/// `logs_root()` joined with [`LogComponent::subdir`].
pub fn log_dir_for(component: LogComponent) -> PathBuf {
    logs_root().join(component.subdir())
}

/// Full path to a timestamped log file: `<logs>/<component>/<timestamp>.log`.
pub fn log_file_path(component: LogComponent, timestamp: &str) -> PathBuf {
    log_dir_for(component).join(format!("{timestamp}.log"))
}

/// Ensures `<logs>/<component>/` exists.
pub fn ensure_log_dir(component: LogComponent) -> std::io::Result<PathBuf> {
    let dir = log_dir_for(component);
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Creates the component log directory, ensures [`log_file_path`] parent exists, initializes the
/// global logger, and returns the log file path for panic hooks or host output redirection.
///
/// Equivalent to [`crate::ensure_log_dir`] plus [`crate::init`] with the resolved [`PathBuf`].
///
/// # Errors
///
/// Returns [`Err`] if the directory cannot be created or the log file cannot be opened.
pub fn init_for(
    component: LogComponent,
    timestamp: &str,
    max_level: LogLevel,
    append: bool,
) -> std::io::Result<PathBuf> {
    ensure_log_dir(component)?;
    let path = log_file_path(component, timestamp);
    output::init_with_mirror(&path, max_level, append, false)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn logs_root_from_manifest_path() {
        let manifest = Path::new("/workspace/Renderide/crates/logger");
        let root = logs_root_with(manifest, None).expect("resolve logs root");
        assert_eq!(root, PathBuf::from("/workspace/Renderide/logs"));
    }

    #[test]
    fn logs_root_env_override_wins() {
        let manifest = Path::new("/workspace/Renderide/crates/logger");
        let root = logs_root_with(manifest, Some(OsStr::new("/tmp/custom_logs")))
            .expect("resolve logs root");
        assert_eq!(root, PathBuf::from("/tmp/custom_logs"));
    }

    #[test]
    fn logs_root_with_env_override_takes_precedence_over_short_manifest() {
        let manifest = Path::new("/logger");
        let root =
            logs_root_with(manifest, Some(OsStr::new("/tmp/override_logs"))).expect("env override");
        assert_eq!(root, PathBuf::from("/tmp/override_logs"));
    }

    #[test]
    fn log_component_subdirs() {
        assert_eq!(LogComponent::Bootstrapper.subdir(), "bootstrapper");
        assert_eq!(LogComponent::Host.subdir(), "host");
        assert_eq!(LogComponent::Renderer.subdir(), "renderer");
    }

    #[test]
    fn log_component_display_matches_subdir() {
        assert_eq!(format!("{}", LogComponent::Bootstrapper), "bootstrapper");
        assert_eq!(format!("{}", LogComponent::Host), "host");
        assert_eq!(format!("{}", LogComponent::Renderer), "renderer");
    }

    #[test]
    fn log_file_path_layout() {
        let manifest = Path::new("/r/Renderide/crates/logger");
        let expected = logs_root_with(manifest, None)
            .expect("resolve logs root")
            .join("renderer")
            .join("2026-04-05_12-00-00.log");
        let got = logs_root_with(manifest, None)
            .expect("resolve logs root")
            .join(LogComponent::Renderer.subdir())
            .join("2026-04-05_12-00-00.log");
        assert_eq!(got, expected);
    }

    #[test]
    fn log_file_path_appends_dot_log_suffix() {
        let p = log_file_path(LogComponent::Host, "ts");
        assert!(p.to_string_lossy().ends_with("ts.log"));
    }

    #[test]
    fn log_dir_for_each_component_distinct() {
        let manifest = Path::new("/z/Renderide/crates/logger");
        let root = logs_root_with(manifest, None).expect("root");
        let a = root.join(LogComponent::Bootstrapper.subdir());
        let b = root.join(LogComponent::Host.subdir());
        let c = root.join(LogComponent::Renderer.subdir());
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
    }

    #[test]
    fn logs_root_err_on_short_manifest_path() {
        let manifest = Path::new("/logger");
        let err = logs_root_with(manifest, None).expect_err("short path");
        assert!(matches!(err, LogsRootError::ManifestPathTooShort { .. }));
    }

    #[test]
    fn logs_root_manifest_path_too_short_preserves_manifest_path() {
        let manifest = PathBuf::from("logger");
        let err = logs_root_with(&manifest, None).unwrap_err();
        assert!(matches!(
            err,
            LogsRootError::ManifestPathTooShort { manifest_dir } if manifest_dir == manifest
        ));
    }

    #[test]
    fn ensure_log_dir_creates_directory_using_env_override() {
        let _guard = ENV_LOCK.lock().expect("env lock");
        let dir = tempfile::tempdir().expect("tempdir");
        let prev = std::env::var_os(LOGS_ROOT_ENV);
        std::env::set_var(LOGS_ROOT_ENV, dir.path().as_os_str());
        let result = ensure_log_dir(LogComponent::Renderer);
        if let Some(p) = prev {
            std::env::set_var(LOGS_ROOT_ENV, p);
        } else {
            std::env::remove_var(LOGS_ROOT_ENV);
        }
        let path = result.expect("ensure_log_dir");
        assert!(path.is_dir());
        assert!(path.ends_with("renderer"));
    }
}
