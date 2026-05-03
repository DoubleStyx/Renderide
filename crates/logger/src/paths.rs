//! Resolves `Renderide/logs/<component>/`, applies the `RENDERIDE_LOGS_ROOT` override, and wires
//! [`init_for`] to the global file sink in [`crate::output`].

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use crate::level::LogLevel;
use crate::output;

/// Environment variable that overrides the default `Renderide/logs` root directory.
pub const LOGS_ROOT_ENV: &str = "RENDERIDE_LOGS_ROOT";

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
/// `RENDERIDE_LOGS_ROOT` environment variable). Otherwise `manifest_dir` must be
/// `.../Renderide/crates/logger`;
/// [`Path::ancestors`] yields `logger` -> `crates` -> `Renderide`, so index `2` is the repository root.
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

/// Root directory containing per-component folders (`bootstrapper`, `host`, `renderer`,
/// `renderer-test`).
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
        // Can't route through the logger -- this is the logger bootstrap path.
        #[expect(
            clippy::print_stderr,
            reason = "logger not yet initialized at bootstrap"
        )]
        {
            eprintln!("Renderide logger: {e}; using fallback logs directory");
        };
        std::env::current_dir().map_or_else(|_| PathBuf::from("logs"), |p| p.join("logs"))
    })
}

/// `logs_root()` joined with [`LogComponent::subdir`].
pub fn log_dir_for(component: LogComponent) -> PathBuf {
    logs_root().join(component.subdir())
}

/// Full path to a timestamped log file: `<logs>/<component>/<timestamp>.log`.
///
/// The `timestamp` is sanitized via [`sanitize_timestamp`] before being joined to the log
/// directory: any character outside `[A-Za-z0-9_-]` is replaced with `_` so that a caller
/// passing path-like input (e.g. `"../etc/passwd"`) cannot escape the component log
/// directory or write to a different file extension. Empty or fully-stripped timestamps fall
/// back to `"invalid"` so the result is always a single, well-formed filename.
pub fn log_file_path(component: LogComponent, timestamp: &str) -> PathBuf {
    let safe = sanitize_timestamp(timestamp);
    log_dir_for(component).join(format!("{safe}.log"))
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
    use std::ffi::{OsStr, OsString};
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
                    Some(p) => std::env::set_var(LOGS_ROOT_ENV, p),
                    None => std::env::remove_var(LOGS_ROOT_ENV),
                }
            }
        }
    }

    /// Sets `RENDERIDE_LOGS_ROOT` to `root` under the [`ENV_LOCK`] mutex, returning a guard that
    /// restores the prior value on drop. Use this for any test that mutates the env var so the
    /// restoration runs even if the test panics.
    fn with_logs_root_override(root: &Path) -> LogsRootOverride<'static> {
        let guard = ENV_LOCK.lock().expect("env lock");
        let prev = std::env::var_os(LOGS_ROOT_ENV);
        // SAFETY: env mutation in test; serialized via the ENV_LOCK guard held above.
        unsafe {
            std::env::set_var(LOGS_ROOT_ENV, root.as_os_str());
        }
        LogsRootOverride {
            _guard: guard,
            prev,
        }
    }

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
    fn log_file_path_sanitizes_path_traversal_attempts() {
        let p = log_file_path(LogComponent::Host, "../etc/passwd");
        let s = p.to_string_lossy();
        assert!(!s.contains(".."), "must not pass `..` through: {s}");
        assert!(!s.contains("/etc/"), "must not pass `/` through: {s}");
        assert!(s.ends_with(".log"));
        // Component directory is preserved (use path components; Windows uses `\\` not `/`).
        assert!(
            p.iter().any(|c| c == std::ffi::OsStr::new("host")),
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
        let manifest = Path::new("/z/Renderide/crates/logger");
        let root = logs_root_with(manifest, None).expect("root");
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
