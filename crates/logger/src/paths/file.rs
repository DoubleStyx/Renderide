//! Component log directories, timestamp sanitization, and log file path construction.

use std::io;
use std::path::{Path, PathBuf};

use super::component::LogComponent;
use super::root;

/// `logs_root()` joined with [`LogComponent::subdir`].
pub fn log_dir_for(component: LogComponent) -> PathBuf {
    root::logs_root().join(component.subdir())
}

/// Full path to a timestamped log file: `<logs>/<component>/<timestamp>.log`.
///
/// The `timestamp` is sanitized via `sanitize_timestamp` before being joined to the log
/// directory: any character outside `[A-Za-z0-9_-]` is replaced with `_` so that a caller
/// passing path-like input (e.g. `"../etc/passwd"`) cannot escape the component log
/// directory or write to a different file extension. Empty or fully-stripped timestamps fall
/// back to `"invalid"` so the result is always a single, well-formed filename.
pub fn log_file_path(component: LogComponent, timestamp: &str) -> PathBuf {
    log_file_path_at_root(&root::logs_root(), component, timestamp)
}

/// Full path to a timestamped log file under an already-selected `root`.
pub(super) fn log_file_path_at_root(
    root: &Path,
    component: LogComponent,
    timestamp: &str,
) -> PathBuf {
    let safe = sanitize_timestamp(timestamp);
    root.join(component.subdir()).join(format!("{safe}.log"))
}

/// Adds path context to an [`io::Error`] while preserving the source error kind.
pub(super) fn io_with_path_context(action: &str, path: &Path, source: io::Error) -> io::Error {
    io::Error::new(
        source.kind(),
        format!("{action} {}: {source}", path.display()),
    )
}

/// Ensures the log directory for `component` exists under `root`.
pub(super) fn ensure_log_dir_at(root: &Path, component: LogComponent) -> io::Result<PathBuf> {
    let dir = root.join(component.subdir());
    std::fs::create_dir_all(&dir)
        .map_err(|source| io_with_path_context("failed to create log directory", &dir, source))?;
    Ok(dir)
}

/// Replaces every character outside `[A-Za-z0-9_-]` with `_`; empty input becomes `"invalid"`.
///
/// This is a defense-in-depth guard for [`log_file_path`]: every current caller produces
/// timestamps via [`crate::log_filename_timestamp`] (already in the safe alphabet), but the
/// public signature accepts arbitrary `&str` and we do not want a future caller -- or
/// attacker-influenced input -- to slip a `..` segment or `/` into the joined path.
pub(super) fn sanitize_timestamp(timestamp: &str) -> String {
    let mut out = String::with_capacity(timestamp.len());
    for c in timestamp.chars() {
        if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
            out.push(c);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        out.push_str("invalid");
    }
    out
}
