//! Standard log layout helpers and `init_for` wiring.

mod component;
mod file;
mod root;

use std::io;
use std::path::{Path, PathBuf};

use crate::level::LogLevel;
use crate::output;

pub use component::LogComponent;
pub use file::{log_dir_for, log_file_path};
pub use root::{LogsRootError, logs_root, logs_root_with};

use file::{ensure_log_dir_at, io_with_path_context, log_file_path_at_root};
use root::{log_root_candidates, remember_selected_logs_root, strict_explicit_root_active};

#[cfg(test)]
use file::sanitize_timestamp;
#[cfg(test)]
use root::{LOGS_ROOT_ENV, default_logs_root_candidates, per_user_logs_root_with};

/// Applies `attempt` to runtime log-root candidates until one succeeds.
fn with_first_available_log_root<T>(
    mut attempt: impl FnMut(&Path) -> io::Result<T>,
) -> io::Result<(PathBuf, T)> {
    let strict = strict_explicit_root_active();
    let mut last_error = None;
    for root in log_root_candidates() {
        match attempt(&root) {
            Ok(value) => return Ok((root, value)),
            Err(error) if strict => return Err(error),
            Err(error) => last_error = Some(error),
        }
    }
    Err(last_error.unwrap_or_else(|| io::Error::other("no Renderide log root candidate available")))
}

/// Initializes logging under a specific root and returns the opened log path.
fn init_for_root(
    root: &Path,
    component: LogComponent,
    timestamp: &str,
    max_level: LogLevel,
    append: bool,
) -> io::Result<PathBuf> {
    ensure_log_dir_at(root, component)?;
    let path = log_file_path_at_root(root, component, timestamp);
    output::init_with_mirror(&path, max_level, append, false)
        .map_err(|source| io_with_path_context("failed to open log file", &path, source))?;
    Ok(path)
}

/// Ensures `<logs>/<component>/` exists.
pub fn ensure_log_dir(component: LogComponent) -> io::Result<PathBuf> {
    with_first_available_log_root(|root| ensure_log_dir_at(root, component)).map(|(_, path)| path)
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
    let (root, path) = with_first_available_log_root(|root| {
        init_for_root(root, component, timestamp, max_level, append)
    })?;
    remember_selected_logs_root(&root);
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::ffi::{OsStr, OsString};
    use std::fs;
    use std::sync::{Mutex, MutexGuard};

    /// Serializes process environment changes across logger path tests.
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

    /// Creates a temporary directory with enough Cargo metadata to look like a Renderide checkout.
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

    /// Verifies workspace discovery resolves from a crate directory to the checkout logs root.
    #[test]
    fn logs_root_from_workspace_path() {
        let workspace = make_workspace_root();
        let manifest = workspace.path().join("crates/logger");
        let root = logs_root_with(&manifest, None).expect("resolve logs root");
        assert_eq!(root, workspace.path().join("logs"));
    }

    /// Verifies an explicit logs root overrides workspace discovery.
    #[test]
    fn logs_root_env_override_wins() {
        let manifest = Path::new("/workspace/Renderide/crates/logger");
        let root = logs_root_with(manifest, Some(Path::new("/tmp/custom_logs").as_os_str()))
            .expect("resolve logs root");
        assert_eq!(root, PathBuf::from("/tmp/custom_logs"));
    }

    /// Verifies an explicit logs root works even when no checkout can be found.
    #[test]
    fn logs_root_with_env_override_takes_precedence_over_missing_workspace() {
        let manifest = Path::new("/logger");
        let root = logs_root_with(manifest, Some(Path::new("/tmp/override_logs").as_os_str()))
            .expect("env override");
        assert_eq!(root, PathBuf::from("/tmp/override_logs"));
    }

    /// Verifies root candidates keep the intended workspace, user, executable, temp ordering.
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

    /// Verifies an explicit root disables all lower-priority fallbacks.
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

    /// Verifies Linux prefers XDG state home for per-user logs.
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

    /// Verifies Linux falls back to `$HOME/.local/state`.
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

    /// Verifies macOS follows the `Library/Logs` convention.
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

    /// Verifies Windows follows the `LOCALAPPDATA` convention.
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

    /// Verifies each component maps to its stable subdirectory.
    #[test]
    fn log_component_subdirs() {
        assert_eq!(LogComponent::Bootstrapper.subdir(), "bootstrapper");
        assert_eq!(LogComponent::Host.subdir(), "host");
        assert_eq!(LogComponent::Renderer.subdir(), "renderer");
        assert_eq!(LogComponent::RendererTest.subdir(), "renderer-test");
    }

    /// Verifies display formatting matches the component subdirectory.
    #[test]
    fn log_component_display_matches_subdir() {
        assert_eq!(format!("{}", LogComponent::Bootstrapper), "bootstrapper");
        assert_eq!(format!("{}", LogComponent::Host), "host");
        assert_eq!(format!("{}", LogComponent::Renderer), "renderer");
        assert_eq!(format!("{}", LogComponent::RendererTest), "renderer-test");
    }

    /// Verifies timestamped log files are placed under the component directory.
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

    /// Verifies log file paths always use the `.log` suffix.
    #[test]
    fn log_file_path_appends_dot_log_suffix() {
        let p = log_file_path(LogComponent::Host, "ts");
        assert!(p.to_string_lossy().ends_with("ts.log"));
    }

    /// Verifies hostile timestamp input cannot produce path traversal.
    #[test]
    fn log_file_path_sanitizes_path_traversal_attempts() {
        let p = log_file_path(LogComponent::Host, "../etc/passwd");
        let s = p.to_string_lossy();
        assert!(!s.contains(".."), "must not pass `..` through: {s}");
        assert!(!s.contains("/etc/"), "must not pass `/` through: {s}");
        assert!(s.ends_with(".log"));
        assert!(
            p.iter().any(|c| c == OsStr::new("host")),
            "missing component dir: {p:?}"
        );
    }

    /// Verifies empty timestamp input falls back to a stable placeholder.
    #[test]
    fn log_file_path_empty_timestamp_falls_back_to_invalid() {
        let p = log_file_path(LogComponent::Host, "");
        assert!(p.to_string_lossy().ends_with("invalid.log"));
    }

    /// Verifies timestamp sanitization preserves the filename-safe alphabet.
    #[test]
    fn sanitize_timestamp_preserves_safe_alphabet() {
        assert_eq!(
            sanitize_timestamp("2026-04-25_12-30-00"),
            "2026-04-25_12-30-00"
        );
    }

    /// Verifies timestamp sanitization replaces unsafe characters one-for-one.
    #[test]
    fn sanitize_timestamp_replaces_unsafe_characters() {
        assert_eq!(sanitize_timestamp("a/b\\c.d"), "a_b_c_d");
    }

    /// Verifies each component has a distinct log directory.
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

    /// Verifies root candidates always include the temp fallback.
    #[test]
    fn default_candidates_fall_back_to_temp_without_workspace_or_user_root() {
        let temp_root = PathBuf::from("/tmp/renderide/logs");
        let roots = default_logs_root_candidates(&[], None, None, None, temp_root.clone());
        assert_eq!(roots, vec![temp_root]);
    }

    /// Verifies `ensure_log_dir` creates the requested component directory.
    #[test]
    fn ensure_log_dir_creates_directory_using_env_override() {
        let dir = tempfile::tempdir().expect("tempdir");
        let _override = with_logs_root_override(dir.path());
        let path = ensure_log_dir(LogComponent::Renderer).expect("ensure_log_dir");
        assert!(path.is_dir());
        assert!(path.ends_with("renderer"));
    }

    /// Verifies each known unsafe character is replaced independently.
    #[test]
    fn sanitize_timestamp_replaces_each_individually_unsafe_char() {
        for unsafe_char in ['\n', '\t', ' ', '"', '\'', '/', '\\', '.', ':', ';'] {
            let input = format!("a{unsafe_char}b");
            let got = sanitize_timestamp(&input);
            assert_eq!(got, "a_b", "input {input:?} produced {got:?}");
        }
    }

    /// Verifies consecutive unsafe characters are not collapsed.
    #[test]
    fn sanitize_timestamp_replaces_each_consecutive_unsafe_char_one_to_one() {
        assert_eq!(sanitize_timestamp("a///b"), "a___b");
        assert_eq!(sanitize_timestamp(".../"), "____");
    }

    /// Verifies empty strings sanitize to the fallback stem.
    #[test]
    fn sanitize_timestamp_empty_string_returns_invalid_fallback() {
        assert_eq!(sanitize_timestamp(""), "invalid");
    }

    /// Verifies repeated directory creation is harmless and stable.
    #[test]
    fn ensure_log_dir_is_idempotent_for_already_existing_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        let _override = with_logs_root_override(dir.path());

        let p1 = ensure_log_dir(LogComponent::Bootstrapper).expect("first call");
        let p2 = ensure_log_dir(LogComponent::Bootstrapper).expect("second call must also succeed");
        assert_eq!(p1, p2);
        assert!(p2.is_dir());
    }

    /// Verifies non-ASCII codepoints are replaced as whole characters while adjacent safe
    /// characters survive.
    #[test]
    fn sanitize_timestamp_replaces_unicode_with_underscores() {
        let input = format!("ts-{}-{}-2026", '\u{03c0}', '\u{1f680}');
        let got = sanitize_timestamp(&input);
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
