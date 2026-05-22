//! Resolves the runtime logs root, applies the `RENDERIDE_LOGS_ROOT` override, and wires
//! [`init_for`] to the global file sink in [`crate::output`].

use std::env;
use std::ffi::{OsStr, OsString};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::level::LogLevel;
use crate::output;

/// Environment variable that overrides the default Renderide logs root directory.
pub const LOGS_ROOT_ENV: &str = "RENDERIDE_LOGS_ROOT";

const APP_DIR_NAME: &str = "renderide";
#[cfg(any(target_os = "macos", target_os = "windows"))]
const USER_DIR_NAME: &str = "Renderide";

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

/// Which part of the system produces a log stream under [`logs_root`] / `<component>/`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LogComponent {
    /// Bootstrapper process (Rust).
    Bootstrapper,
    /// Host process output captured by the bootstrapper (stdout/stderr into one file).
    Host,
    /// Renderer process (Rust).
    Renderer,
    /// Renderer integration-test harness process (Rust).
    RendererTest,
}

impl LogComponent {
    /// Subdirectory name under `logs/` for this component.
    pub const fn subdir(self) -> &'static str {
        match self {
            Self::Bootstrapper => "bootstrapper",
            Self::Host => "host",
            Self::Renderer => "renderer",
            Self::RendererTest => "renderer-test",
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
/// If logging has already been initialized through [`init_for`], this returns the selected root
/// from that successful initialization. Otherwise the root is chosen at runtime: an explicit
/// `RENDERIDE_LOGS_ROOT`, a discovered checkout `logs` directory, a per-user logs directory, an
/// executable-adjacent `logs` directory, then a temp-directory fallback.
pub fn logs_root() -> PathBuf {
    selected_logs_root()
        .or_else(|| log_root_candidates().into_iter().next())
        .unwrap_or_else(temp_logs_root)
}

/// `logs_root()` joined with [`LogComponent::subdir`].
pub fn log_dir_for(component: LogComponent) -> PathBuf {
    logs_root().join(component.subdir())
}

/// Full path to a timestamped log file: `<logs>/<component>/<timestamp>.log`.
///
/// The `timestamp` is sanitized via `sanitize_timestamp` before being joined to the log
/// directory: any character outside `[A-Za-z0-9_-]` is replaced with `_` so that a caller
/// passing path-like input (e.g. `"../etc/passwd"`) cannot escape the component log
/// directory or write to a different file extension. Empty or fully-stripped timestamps fall
/// back to `"invalid"` so the result is always a single, well-formed filename.
pub fn log_file_path(component: LogComponent, timestamp: &str) -> PathBuf {
    let safe = sanitize_timestamp(timestamp);
    log_dir_for(component).join(format!("{safe}.log"))
}

fn selected_logs_root() -> Option<PathBuf> {
    SELECTED_LOGS_ROOT.lock().ok().and_then(|root| root.clone())
}

fn remember_selected_logs_root(root: &Path) {
    if let Ok(mut selected) = SELECTED_LOGS_ROOT.lock()
        && selected.is_none()
    {
        *selected = Some(root.to_path_buf());
    }
}

fn non_empty_path(path: &OsStr) -> Option<PathBuf> {
    if path.is_empty() {
        None
    } else {
        Some(PathBuf::from(path))
    }
}

fn explicit_logs_root() -> Option<PathBuf> {
    env::var_os(LOGS_ROOT_ENV)
        .as_deref()
        .and_then(non_empty_path)
}

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

fn binary_output_dir() -> Option<PathBuf> {
    env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(Path::to_path_buf))
}

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

fn push_unique(out: &mut Vec<PathBuf>, path: PathBuf) {
    if !out.iter().any(|candidate| candidate == &path) {
        out.push(path);
    }
}

fn default_logs_root_candidates(
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

fn log_root_candidates() -> Vec<PathBuf> {
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

fn temp_logs_root() -> PathBuf {
    env::temp_dir().join(APP_DIR_NAME).join("logs")
}

fn per_user_logs_root() -> Option<PathBuf> {
    per_user_logs_root_with(|key| env::var_os(key))
}

fn per_user_logs_root_with(mut get_env: impl FnMut(&str) -> Option<OsString>) -> Option<PathBuf> {
    per_user_logs_root_for_platform(&mut get_env)
}

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

#[cfg(target_os = "macos")]
fn per_user_logs_root_for_platform(
    get_env: &mut impl FnMut(&str) -> Option<OsString>,
) -> Option<PathBuf> {
    get_env("HOME")
        .as_deref()
        .and_then(non_empty_path)
        .map(|home| home.join("Library").join("Logs").join(USER_DIR_NAME))
}

#[cfg(target_os = "windows")]
fn per_user_logs_root_for_platform(
    get_env: &mut impl FnMut(&str) -> Option<OsString>,
) -> Option<PathBuf> {
    get_env("LOCALAPPDATA")
        .as_deref()
        .and_then(non_empty_path)
        .map(|root| root.join(USER_DIR_NAME).join("logs"))
}

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

#[cfg(not(any(unix, target_os = "windows")))]
fn per_user_logs_root_for_platform(
    _get_env: &mut impl FnMut(&str) -> Option<OsString>,
) -> Option<PathBuf> {
    None
}

fn io_with_path_context(action: &str, path: &Path, source: io::Error) -> io::Error {
    io::Error::new(
        source.kind(),
        format!("{action} {}: {source}", path.display()),
    )
}

fn ensure_log_dir_at(root: &Path, component: LogComponent) -> io::Result<PathBuf> {
    let dir = root.join(component.subdir());
    std::fs::create_dir_all(&dir)
        .map_err(|source| io_with_path_context("failed to create log directory", &dir, source))?;
    Ok(dir)
}

fn init_for_root(
    root: &Path,
    component: LogComponent,
    timestamp: &str,
    max_level: LogLevel,
    append: bool,
) -> io::Result<PathBuf> {
    let dir = ensure_log_dir_at(root, component)?;
    let safe = sanitize_timestamp(timestamp);
    let path = dir.join(format!("{safe}.log"));
    output::init_with_mirror(&path, max_level, append, false)
        .map_err(|source| io_with_path_context("failed to open log file", &path, source))?;
    Ok(path)
}

/// Replaces every character outside `[A-Za-z0-9_-]` with `_`; empty input becomes `"invalid"`.
///
/// This is a defense-in-depth guard for [`log_file_path`]: every current caller produces
/// timestamps via [`crate::log_filename_timestamp`] (already in the safe alphabet), but the
/// public signature accepts arbitrary `&str` and we do not want a future caller -- or
/// attacker-influenced input -- to slip a `..` segment or `/` into the joined path.
fn sanitize_timestamp(timestamp: &str) -> String {
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

/// Ensures `<logs>/<component>/` exists.
pub fn ensure_log_dir(component: LogComponent) -> io::Result<PathBuf> {
    let strict = selected_logs_root().is_none() && explicit_logs_root().is_some();
    let mut last_error = None;
    for root in log_root_candidates() {
        match ensure_log_dir_at(&root, component) {
            Ok(path) => return Ok(path),
            Err(error) if strict => return Err(error),
            Err(error) => last_error = Some(error),
        }
    }
    Err(last_error.unwrap_or_else(|| io::Error::other("no Renderide log root candidate available")))
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
) -> io::Result<PathBuf> {
    let strict = selected_logs_root().is_none() && explicit_logs_root().is_some();
    let mut last_error = None;
    for root in log_root_candidates() {
        match init_for_root(&root, component, timestamp, max_level, append) {
            Ok(path) => {
                remember_selected_logs_root(&root);
                return Ok(path);
            }
            Err(error) if strict => return Err(error),
            Err(error) => last_error = Some(error),
        }
    }
    Err(last_error.unwrap_or_else(|| io::Error::other("no Renderide log root candidate available")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::{OsStr, OsString};
    use std::fs;
    use std::sync::{Mutex, MutexGuard};

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// RAII guard that restores `RENDERIDE_LOGS_ROOT` to its prior value when dropped, even on
    /// panic. Holds the [`ENV_LOCK`] mutex for the lifetime of the override so concurrent tests
    /// cannot observe each other's value.
    struct LogsRootOverride<'lock> {
        /// Mutex guard kept alive so the env-var window cannot overlap another test.
        _guard: MutexGuard<'lock, ()>,
        /// The value that was set in the environment before the override, restored on drop.
        prev: Option<OsString>,
    }

    impl Drop for LogsRootOverride<'_> {
        fn drop(&mut self) {
            // SAFETY: env mutation in test; serialized via the ENV_LOCK guard held by `_guard`.
            unsafe {
                match self.prev.take() {
                    Some(p) => env::set_var(LOGS_ROOT_ENV, p),
                    None => env::remove_var(LOGS_ROOT_ENV),
                }
            }
        }
    }

    /// Sets `RENDERIDE_LOGS_ROOT` to `root` under the [`ENV_LOCK`] mutex, returning a guard that
    /// restores the prior value on drop. Use this for any test that mutates the env var so the
    /// restoration runs even if the test panics.
    fn with_logs_root_override(root: &Path) -> LogsRootOverride<'static> {
        let guard = ENV_LOCK.lock().expect("env lock");
        let prev = env::var_os(LOGS_ROOT_ENV);
        // SAFETY: env mutation in test; serialized via the ENV_LOCK guard held above.
        unsafe {
            env::set_var(LOGS_ROOT_ENV, root.as_os_str());
        }
        LogsRootOverride {
            _guard: guard,
            prev,
        }
    }

    fn make_workspace_root() -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::write(dir.path().join("Cargo.toml"), "[workspace]\n").expect("workspace manifest");
        fs::create_dir_all(dir.path().join("crates/logger")).expect("logger crate dir");
        fs::write(
            dir.path().join("crates/logger/Cargo.toml"),
            "[package]\nname = \"logger\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
        )
        .expect("logger manifest");
        fs::create_dir_all(dir.path().join("crates/renderide")).expect("renderide crate dir");
        fs::write(
            dir.path().join("crates/renderide/Cargo.toml"),
            "[package]\nname = \"renderide\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
        )
        .expect("renderide manifest");
        dir
    }

    #[test]
    fn logs_root_from_workspace_path() {
        let workspace = make_workspace_root();
        let manifest = workspace.path().join("crates/logger");
        let root = logs_root_with(&manifest, None).expect("resolve logs root");
        assert_eq!(root, workspace.path().join("logs"));
    }

    #[test]
    fn logs_root_env_override_wins() {
        let manifest = Path::new("/workspace/Renderide/crates/logger");
        let root = logs_root_with(manifest, Some(Path::new("/tmp/custom_logs").as_os_str()))
            .expect("resolve logs root");
        assert_eq!(root, PathBuf::from("/tmp/custom_logs"));
    }

    #[test]
    fn logs_root_with_env_override_takes_precedence_over_missing_workspace() {
        let manifest = Path::new("/logger");
        let root = logs_root_with(manifest, Some(Path::new("/tmp/override_logs").as_os_str()))
            .expect("env override");
        assert_eq!(root, PathBuf::from("/tmp/override_logs"));
    }

    #[test]
    fn default_candidates_keep_workspace_before_user_logs() {
        let workspace = make_workspace_root();
        let user_root = PathBuf::from("/user/renderide/logs");
        let exe_dir = PathBuf::from("/install/bin");
        let temp_root = PathBuf::from("/tmp/renderide/logs");

        let roots = default_logs_root_candidates(
            &[workspace.path().join("target/release")],
            None,
            Some(user_root.clone()),
            Some(exe_dir.clone()),
            temp_root.clone(),
        );

        assert_eq!(roots[0], workspace.path().join("logs"));
        assert_eq!(roots[1], user_root);
        assert_eq!(roots[2], exe_dir.join("logs"));
        assert_eq!(roots[3], temp_root);
    }

    #[test]
    fn default_candidates_use_strict_explicit_root_only() {
        let workspace = make_workspace_root();
        let explicit = PathBuf::from("/explicit/logs");

        let roots = default_logs_root_candidates(
            &[workspace.path().to_path_buf()],
            Some(explicit.clone()),
            Some(PathBuf::from("/user/logs")),
            Some(PathBuf::from("/exe")),
            PathBuf::from("/tmp/logs"),
        );

        assert_eq!(roots, vec![explicit]);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn per_user_logs_root_prefers_xdg_state_home_on_linux() {
        let root = per_user_logs_root_with(|key| match key {
            "XDG_STATE_HOME" => Some(OsString::from("/state")),
            "HOME" => Some(OsString::from("/home/user")),
            _ => None,
        })
        .expect("user logs root");

        assert_eq!(root, PathBuf::from("/state/renderide/logs"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn per_user_logs_root_falls_back_to_home_on_linux() {
        let root = per_user_logs_root_with(|key| match key {
            "HOME" => Some(OsString::from("/home/user")),
            _ => None,
        })
        .expect("user logs root");

        assert_eq!(
            root,
            PathBuf::from("/home/user/.local/state/renderide/logs")
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn per_user_logs_root_uses_library_logs_on_macos() {
        let root = per_user_logs_root_with(|key| match key {
            "HOME" => Some(OsString::from("/Users/user")),
            _ => None,
        })
        .expect("user logs root");

        assert_eq!(root, PathBuf::from("/Users/user/Library/Logs/Renderide"));
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn per_user_logs_root_uses_local_app_data_on_windows() {
        let root = per_user_logs_root_with(|key| match key {
            "LOCALAPPDATA" => Some(OsString::from(r"C:\Users\user\AppData\Local")),
            _ => None,
        })
        .expect("user logs root");

        assert_eq!(
            root,
            PathBuf::from(r"C:\Users\user\AppData\Local\Renderide\logs")
        );
    }

    #[test]
    fn log_component_subdirs() {
        assert_eq!(LogComponent::Bootstrapper.subdir(), "bootstrapper");
        assert_eq!(LogComponent::Host.subdir(), "host");
        assert_eq!(LogComponent::Renderer.subdir(), "renderer");
        assert_eq!(LogComponent::RendererTest.subdir(), "renderer-test");
    }

    #[test]
    fn log_component_display_matches_subdir() {
        assert_eq!(format!("{}", LogComponent::Bootstrapper), "bootstrapper");
        assert_eq!(format!("{}", LogComponent::Host), "host");
        assert_eq!(format!("{}", LogComponent::Renderer), "renderer");
        assert_eq!(format!("{}", LogComponent::RendererTest), "renderer-test");
    }

    #[test]
    fn log_file_path_layout() {
        let dir = tempfile::tempdir().expect("tempdir");
        let _override = with_logs_root_override(dir.path());
        let expected = dir.path().join("renderer").join("2026-04-05_12-00-00.log");
        assert_eq!(
            log_file_path(LogComponent::Renderer, "2026-04-05_12-00-00"),
            expected
        );
    }

    #[test]
    fn log_file_path_appends_dot_log_suffix() {
        let p = log_file_path(LogComponent::Host, "ts");
        assert!(p.to_string_lossy().ends_with("ts.log"));
    }

    #[test]
    fn log_file_path_sanitizes_path_traversal_attempts() {
        let p = log_file_path(LogComponent::Host, "../etc/passwd");
        let s = p.to_string_lossy();
        assert!(!s.contains(".."), "must not pass `..` through: {s}");
        assert!(!s.contains("/etc/"), "must not pass `/` through: {s}");
        assert!(s.ends_with(".log"));
        // Component directory is preserved (use path components; Windows uses `\\` not `/`).
        assert!(
            p.iter().any(|c| c == OsStr::new("host")),
            "missing component dir: {p:?}"
        );
    }

    #[test]
    fn log_file_path_empty_timestamp_falls_back_to_invalid() {
        let p = log_file_path(LogComponent::Host, "");
        assert!(p.to_string_lossy().ends_with("invalid.log"));
    }

    #[test]
    fn sanitize_timestamp_preserves_safe_alphabet() {
        assert_eq!(
            sanitize_timestamp("2026-04-25_12-30-00"),
            "2026-04-25_12-30-00"
        );
    }

    #[test]
    fn sanitize_timestamp_replaces_unsafe_characters() {
        assert_eq!(sanitize_timestamp("a/b\\c.d"), "a_b_c_d");
    }

    #[test]
    fn log_dir_for_each_component_distinct() {
        let dir = tempfile::tempdir().expect("tempdir");
        let _override = with_logs_root_override(dir.path());
        let root = dir.path();
        let a = root.join(LogComponent::Bootstrapper.subdir());
        let b = root.join(LogComponent::Host.subdir());
        let c = root.join(LogComponent::Renderer.subdir());
        let d = root.join(LogComponent::RendererTest.subdir());
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
        assert_ne!(a, d);
        assert_ne!(b, d);
        assert_ne!(c, d);
    }

    #[test]
    fn default_candidates_fall_back_to_temp_without_workspace_or_user_root() {
        let temp_root = PathBuf::from("/tmp/renderide/logs");
        let roots = default_logs_root_candidates(&[], None, None, None, temp_root.clone());
        assert_eq!(roots, vec![temp_root]);
    }

    #[test]
    fn ensure_log_dir_creates_directory_using_env_override() {
        let dir = tempfile::tempdir().expect("tempdir");
        let _override = with_logs_root_override(dir.path());
        let path = ensure_log_dir(LogComponent::Renderer).expect("ensure_log_dir");
        assert!(path.is_dir());
        assert!(path.ends_with("renderer"));
    }

    #[test]
    fn sanitize_timestamp_replaces_each_individually_unsafe_char() {
        for unsafe_char in ['\n', '\t', ' ', '"', '\'', '/', '\\', '.', ':', ';'] {
            let input = format!("a{unsafe_char}b");
            let got = sanitize_timestamp(&input);
            assert_eq!(got, "a_b", "input {input:?} produced {got:?}");
        }
    }

    #[test]
    fn sanitize_timestamp_replaces_each_consecutive_unsafe_char_one_to_one() {
        // The contract is per-char replacement (no run collapsing), so three unsafe characters in
        // a row become three underscores -- important so different inputs cannot accidentally
        // collide on the same sanitized filename.
        assert_eq!(sanitize_timestamp("a///b"), "a___b");
        assert_eq!(sanitize_timestamp(".../"), "____");
    }

    #[test]
    fn sanitize_timestamp_empty_string_returns_invalid_fallback() {
        assert_eq!(sanitize_timestamp(""), "invalid");
    }

    #[test]
    fn ensure_log_dir_is_idempotent_for_already_existing_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        let _override = with_logs_root_override(dir.path());

        let p1 = ensure_log_dir(LogComponent::Bootstrapper).expect("first call");
        let p2 = ensure_log_dir(LogComponent::Bootstrapper).expect("second call must also succeed");
        assert_eq!(p1, p2);
        assert!(p2.is_dir());
    }

    /// Verifies that non-ASCII characters (Greek, emoji) are all replaced with `_` while the
    /// safe alphabet around them survives. Sanitization works at the `char` level, so a
    /// multi-byte codepoint becomes a single `_` (not multiple).
    #[test]
    fn sanitize_timestamp_replaces_unicode_with_underscores() {
        let got = sanitize_timestamp("ts-π-🚀-2026");
        assert_eq!(got, "ts-_-_-2026", "unexpected sanitized form: {got:?}");
    }

    /// Verifies [`log_file_path`] cannot escape the component directory under an env-overridden
    /// logs root, even when given a hostile timestamp containing `..` and path separators.
    #[test]
    fn log_file_path_stays_inside_component_dir_for_malicious_timestamp() {
        let dir = tempfile::tempdir().expect("tempdir");
        let _override = with_logs_root_override(dir.path());

        let p = log_file_path(LogComponent::Renderer, "../escape");

        let component_dir = dir.path().join("renderer");
        assert_eq!(
            p.parent().expect("parent"),
            component_dir.as_path(),
            "expected file directly under component dir: {p:?}"
        );

        let stem = p.file_stem().and_then(|s| s.to_str()).expect("file stem");
        assert!(!stem.contains(".."), "stem must not contain `..`: {stem:?}");
        assert!(!stem.contains('/'), "stem must not contain `/`: {stem:?}");
        assert!(!stem.contains('\\'), "stem must not contain `\\`: {stem:?}");
        assert!(p.to_string_lossy().ends_with(".log"));
    }
}
