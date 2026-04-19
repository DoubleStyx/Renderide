//! Resolves `Renderide/logs/<component>/` and helpers such as [`init_for`].

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use crate::level::LogLevel;
use crate::output;

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

/// Resolves where all Renderide logs live, for use in tests without touching process environment.
///
/// If `override_root` is [`Some`], that path is used as the logs root (`RENDERIDE_LOGS_ROOT`).
/// Otherwise `manifest_dir` must be `.../Renderide/crates/logger`; [`Path::ancestors`] yields
/// `logger` → `crates` → `Renderide`, so index `2` is the repository root.
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
/// environment variable `RENDERIDE_LOGS_ROOT` is set, that path is used instead (no subdirectory
/// insertion; you get exactly that folder as the root for all components).
pub fn logs_root() -> PathBuf {
    logs_root_with(
        Path::new(env!("CARGO_MANIFEST_DIR")),
        std::env::var_os("RENDERIDE_LOGS_ROOT").as_deref(),
    )
    .unwrap_or_else(|e| {
        eprintln!("Renderide logger: {e}; using fallback logs directory");
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

    /// [`LogsRootError::ManifestPathTooShort`] must echo the path that failed resolution (see also
    /// crate-level tests in `lib.rs`).
    #[test]
    fn logs_root_manifest_path_too_short_preserves_manifest_path() {
        let manifest = PathBuf::from("logger");
        let err = logs_root_with(&manifest, None).unwrap_err();
        assert!(matches!(
            err,
            LogsRootError::ManifestPathTooShort { manifest_dir } if manifest_dir == manifest
        ));
    }
}
