//! Runtime log root discovery, explicit-root overrides, and selected-root caching.

use std::env;
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Environment variable that overrides the default Renderide logs root directory.
pub(super) const LOGS_ROOT_ENV: &str = "RENDERIDE_LOGS_ROOT";

/// Lowercase application directory name used for Unix-style per-user and temp paths.
const APP_DIR_NAME: &str = "renderide";

/// Titlecase application directory name used by platform conventions that prefer display names.
#[cfg(any(target_os = "macos", target_os = "windows"))]
const USER_DIR_NAME: &str = "Renderide";

/// Logs root chosen by the first successful [`crate::init_for`] call.
static SELECTED_LOGS_ROOT: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Failure to resolve a default Renderide logs root.
#[derive(Debug, thiserror::Error)]
pub enum LogsRootError {
    /// Compatibility variant preserved for callers that matched the old manifest-path failure.
    #[error(
        "logger manifest path did not resolve to a Renderide workspace root; got {manifest_dir:?}"
    )]
    ManifestPathTooShort {
        /// Path that failed to resolve to a workspace root.
        manifest_dir: PathBuf,
    },
    /// No runtime fallback root was available.
    #[error("no Renderide log root candidate was available")]
    NoCandidates,
}

/// Resolves where all Renderide logs live, for use in tests without touching process environment.
///
/// If `override_root` is [`Some`], that path is used as the logs root (same role as the
/// `RENDERIDE_LOGS_ROOT` environment variable). Otherwise `start` and its ancestors are searched
/// for a Renderide workspace root; if none is found, the per-user and temporary fallbacks are used.
pub fn logs_root_with(
    start: &Path,
    override_root: Option<&OsStr>,
) -> Result<PathBuf, LogsRootError> {
    default_logs_root_candidates(
        &[start.to_path_buf()],
        override_root.and_then(non_empty_path),
        per_user_logs_root(),
        None,
        temp_logs_root(),
    )
    .into_iter()
    .next()
    .ok_or(LogsRootError::NoCandidates)
}

/// Root directory containing per-component folders (`bootstrapper`, `host`, `renderer`,
/// `renderer-test`).
///
/// If logging has already been initialized through [`crate::init_for`], this returns the selected
/// root from that successful initialization. Otherwise the root is chosen at runtime: an explicit
/// `RENDERIDE_LOGS_ROOT`, a discovered checkout `logs` directory, a per-user logs directory, an
/// executable-adjacent `logs` directory, then a temp-directory fallback.
pub fn logs_root() -> PathBuf {
    selected_logs_root()
        .or_else(|| log_root_candidates().into_iter().next())
        .unwrap_or_else(temp_logs_root)
}

/// Returns the cached root selected by a previous successful [`crate::init_for`] call.
pub(super) fn selected_logs_root() -> Option<PathBuf> {
    SELECTED_LOGS_ROOT.lock().ok().and_then(|root| root.clone())
}

/// Caches the root used by the first successful [`crate::init_for`] call.
pub(super) fn remember_selected_logs_root(root: &Path) {
    if let Ok(mut selected) = SELECTED_LOGS_ROOT.lock()
        && selected.is_none()
    {
        *selected = Some(root.to_path_buf());
    }
}

/// Returns whether root fallback should stop at the explicit environment override.
pub(super) fn strict_explicit_root_active() -> bool {
    selected_logs_root().is_none() && explicit_logs_root().is_some()
}

/// Resolves the runtime candidate list in priority order.
pub(super) fn log_root_candidates() -> Vec<PathBuf> {
    if let Some(root) = selected_logs_root() {
        return vec![root];
    }
    default_logs_root_candidates(
        &runtime_start_paths(),
        explicit_logs_root(),
        per_user_logs_root(),
        binary_output_dir(),
        temp_logs_root(),
    )
}

/// Treats an empty OS string as no path.
fn non_empty_path(path: &OsStr) -> Option<PathBuf> {
    if path.is_empty() {
        None
    } else {
        Some(PathBuf::from(path))
    }
}

/// Reads the explicit logs-root environment override.
fn explicit_logs_root() -> Option<PathBuf> {
    env::var_os(LOGS_ROOT_ENV)
        .as_deref()
        .and_then(non_empty_path)
}

/// Returns plausible start paths for checkout discovery.
fn runtime_start_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Ok(exe) = env::current_exe()
        && let Some(parent) = exe.parent()
    {
        push_unique(&mut paths, parent.to_path_buf());
    }
    if let Ok(cwd) = env::current_dir() {
        push_unique(&mut paths, cwd);
    }
    paths
}

/// Returns the current executable directory when available.
fn binary_output_dir() -> Option<PathBuf> {
    env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(Path::to_path_buf))
}

/// Finds the Renderide workspace root by walking ancestors from `start`.
fn find_renderide_workspace_root(start: &Path) -> Option<PathBuf> {
    let mut current = start.to_path_buf();
    loop {
        let cargo = current.join("Cargo.toml");
        let logger = current.join("crates/logger/Cargo.toml");
        let renderer = current.join("crates/renderide/Cargo.toml");
        if cargo.is_file() && logger.is_file() && renderer.is_file() {
            return Some(current);
        }
        if !current.pop() {
            return None;
        }
    }
}

/// Appends `path` to `out` unless it is already present.
fn push_unique(out: &mut Vec<PathBuf>, path: PathBuf) {
    if !out.iter().any(|candidate| candidate == &path) {
        out.push(path);
    }
}

/// Builds the ordered logs-root fallback list from explicit inputs.
pub(super) fn default_logs_root_candidates(
    start_paths: &[PathBuf],
    explicit_root: Option<PathBuf>,
    user_root: Option<PathBuf>,
    exe_dir: Option<PathBuf>,
    temp_root: PathBuf,
) -> Vec<PathBuf> {
    if let Some(root) = explicit_root {
        return vec![root];
    }

    let mut roots = Vec::new();
    for start in start_paths {
        if let Some(workspace) = find_renderide_workspace_root(start) {
            push_unique(&mut roots, workspace.join("logs"));
        }
    }
    if let Some(root) = user_root {
        push_unique(&mut roots, root);
    }
    if let Some(dir) = exe_dir {
        push_unique(&mut roots, dir.join("logs"));
    }
    push_unique(&mut roots, temp_root);
    roots
}

/// Returns the temp-directory fallback logs root.
fn temp_logs_root() -> PathBuf {
    env::temp_dir().join(APP_DIR_NAME).join("logs")
}

/// Returns the platform-specific per-user logs root when available.
fn per_user_logs_root() -> Option<PathBuf> {
    per_user_logs_root_with(|key| env::var_os(key))
}

/// Returns the platform-specific per-user logs root using an injected environment reader.
pub(super) fn per_user_logs_root_with(
    mut get_env: impl FnMut(&str) -> Option<OsString>,
) -> Option<PathBuf> {
    per_user_logs_root_for_platform(&mut get_env)
}

/// Resolves the Linux per-user logs root.
#[cfg(target_os = "linux")]
fn per_user_logs_root_for_platform(
    get_env: &mut impl FnMut(&str) -> Option<OsString>,
) -> Option<PathBuf> {
    if let Some(root) = get_env("XDG_STATE_HOME")
        .as_deref()
        .and_then(non_empty_path)
    {
        Some(root.join(APP_DIR_NAME).join("logs"))
    } else {
        get_env("HOME")
            .as_deref()
            .and_then(non_empty_path)
            .map(|home| {
                home.join(".local")
                    .join("state")
                    .join(APP_DIR_NAME)
                    .join("logs")
            })
    }
}

/// Resolves the macOS per-user logs root.
#[cfg(target_os = "macos")]
fn per_user_logs_root_for_platform(
    get_env: &mut impl FnMut(&str) -> Option<OsString>,
) -> Option<PathBuf> {
    get_env("HOME")
        .as_deref()
        .and_then(non_empty_path)
        .map(|home| home.join("Library").join("Logs").join(USER_DIR_NAME))
}

/// Resolves the Windows per-user logs root.
#[cfg(target_os = "windows")]
fn per_user_logs_root_for_platform(
    get_env: &mut impl FnMut(&str) -> Option<OsString>,
) -> Option<PathBuf> {
    get_env("LOCALAPPDATA")
        .as_deref()
        .and_then(non_empty_path)
        .map(|root| root.join(USER_DIR_NAME).join("logs"))
}

/// Resolves a generic Unix per-user logs root for non-Linux, non-macOS targets.
#[cfg(all(unix, not(any(target_os = "linux", target_os = "macos"))))]
fn per_user_logs_root_for_platform(
    get_env: &mut impl FnMut(&str) -> Option<OsString>,
) -> Option<PathBuf> {
    get_env("HOME")
        .as_deref()
        .and_then(non_empty_path)
        .map(|home| {
            home.join(".local")
                .join("state")
                .join(APP_DIR_NAME)
                .join("logs")
        })
}

/// Returns no per-user logs root on targets without a supported user-directory convention.
#[cfg(not(any(unix, target_os = "windows")))]
fn per_user_logs_root_for_platform(
    _get_env: &mut impl FnMut(&str) -> Option<OsString>,
) -> Option<PathBuf> {
    None
}
