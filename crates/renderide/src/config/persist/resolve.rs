//! Locate `config.toml`: `RENDERIDE_CONFIG`, then the per-user config directory.

use std::path::{Path, PathBuf};

#[cfg(test)]
use std::sync::atomic::{AtomicBool, Ordering};

/// How the config file path was chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConfigSource {
    /// `RENDERIDE_CONFIG` pointed at an existing file.
    Env,
    /// File loaded from the per-user config directory.
    Search,
    /// No existing file; defaults were written to the save path on first load.
    Generated,
    /// A previous-layout `config.toml` was migrated into the per-user config directory.
    Migrated,
    /// No file found; caller uses defaults only.
    None,
}

/// Result of resolving a config path (whether or not a file was read).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigResolveOutcome {
    /// Every path checked, in order (`RENDERIDE_CONFIG` first when set, then user config path,
    /// then any previous-layout candidates inspected by migration).
    pub attempted_paths: Vec<PathBuf>,
    /// First existing regular file used for config content (`config.toml`).
    pub loaded_path: Option<PathBuf>,
    /// How the effective config path was chosen (env, user dir, generated, migrated, or none).
    pub source: ConfigSource,
}

/// Canonical on-disk config file (TOML).
pub const FILE_NAME_TOML: &str = "config.toml";
/// Suffix appended to a previous-layout `config.toml` after migration.
pub const LEGACY_MIGRATED_SUFFIX: &str = ".migrated";
const ENV_OVERRIDE: &str = "RENDERIDE_CONFIG";
const APPLICATION_DIR: &str = "Renderide";

#[cfg(test)]
pub(crate) static TEST_USER_CONFIG_OVERRIDE: parking_lot::Mutex<Option<PathBuf>> =
    parking_lot::Mutex::new(None);

/// Returns the per-user renderer config path.
///
/// Platform bases come from [`directories::BaseDirs::config_dir`], then Renderide appends
/// `Renderide/config.toml`.
pub fn user_config_path() -> Option<PathBuf> {
    #[cfg(test)]
    {
        let override_value = TEST_USER_CONFIG_OVERRIDE.lock().clone();
        if let Some(p) = override_value {
            return Some(p);
        }
    }
    directories::BaseDirs::new()
        .map(|dirs| dirs.config_dir().join(APPLICATION_DIR).join(FILE_NAME_TOML))
}

/// Returns `true` if `RENDERIDE_CONFIG` is set to a non-empty value (explicit user path).
pub fn renderide_config_env_nonempty() -> bool {
    match std::env::var(ENV_OVERRIDE) {
        Ok(s) => !s.trim().is_empty(),
        Err(_) => false,
    }
}

fn push_unique(out: &mut Vec<PathBuf>, p: PathBuf) {
    if !out.iter().any(|x| x == &p) {
        out.push(p);
    }
}

/// Records that `config.toml` was created at `path` on first load (see [`super::load::load_renderer_settings`]).
pub fn apply_generated_config(outcome: &mut ConfigResolveOutcome, path: PathBuf) {
    push_unique(&mut outcome.attempted_paths, path.clone());
    outcome.loaded_path = Some(path);
    outcome.source = ConfigSource::Generated;
}

/// Records that a previous-layout `config.toml` was migrated into the user config directory.
pub fn apply_migrated_config(outcome: &mut ConfigResolveOutcome, path: PathBuf) {
    push_unique(&mut outcome.attempted_paths, path.clone());
    outcome.loaded_path = Some(path);
    outcome.source = ConfigSource::Migrated;
}

/// Records a path inspected outside normal config resolution.
pub(super) fn record_attempted_path(outcome: &mut ConfigResolveOutcome, path: PathBuf) {
    push_unique(&mut outcome.attempted_paths, path);
}

/// Resolves the config file path. If `RENDERIDE_CONFIG` is set but missing, logs a warning and
/// continues with the user config directory.
pub fn resolve_config_path() -> ConfigResolveOutcome {
    let mut attempted_paths = Vec::new();

    if let Ok(raw) = std::env::var(ENV_OVERRIDE) {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            let p = PathBuf::from(trimmed);
            push_unique(&mut attempted_paths, p.clone());
            if p.is_file() {
                return ConfigResolveOutcome {
                    attempted_paths,
                    loaded_path: Some(p),
                    source: ConfigSource::Env,
                };
            }
            logger::warn!(
                "{ENV_OVERRIDE}={} does not exist or is not a file; falling back to the user config directory",
                p.display()
            );
        }
    }

    if let Some(p) = user_config_path() {
        push_unique(&mut attempted_paths, p.clone());
        if p.is_file() {
            return ConfigResolveOutcome {
                attempted_paths,
                loaded_path: Some(p),
                source: ConfigSource::Search,
            };
        }
    }

    ConfigResolveOutcome {
        attempted_paths,
        loaded_path: None,
        source: ConfigSource::None,
    }
}

/// Reads the file at `path` if it exists.
pub fn read_config_file(path: &Path) -> std::io::Result<String> {
    std::fs::read_to_string(path)
}

/// Picks the path used when persisting settings from the UI or [`crate::config::save_renderer_settings`].
///
/// - If a file was loaded ([`ConfigResolveOutcome::loaded_path`]), that path is used.
/// - Otherwise: use the per-user config path from [`user_config_path`].
/// - If no user config path is available, fall back to `current_dir()/config.toml`.
pub fn resolve_save_path(resolve: &ConfigResolveOutcome) -> PathBuf {
    if let Some(p) = resolve.loaded_path.clone() {
        return p;
    }

    if let Some(p) = user_config_path() {
        return p;
    }

    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(FILE_NAME_TOML)
}

/// Best-effort writable check used before creating `config.toml`.
pub(crate) fn is_dir_writable(dir: &Path) -> bool {
    if !dir.is_dir() {
        return false;
    }
    let probe = dir.join(".renderide_write_probe");
    match std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&probe)
    {
        Ok(_) => {
            let _ = std::fs::remove_file(&probe);
            true
        }
        Err(_) => false,
    }
}

// TODO(remove-after-1.0): Previous-layout config migration. Earlier builds dropped `config.toml`
// next to the binary, at the workspace root, or in the cwd. Keep these helpers only for the
// one-shot migration path in `super::load`.

/// Walks `start` and its ancestors looking for a directory that contains both `Cargo.toml` and
/// `crates/renderide/Cargo.toml`, identifying the Renderide workspace root.
pub fn find_renderide_workspace_root(start: &Path) -> Option<PathBuf> {
    let mut cur = start.to_path_buf();
    loop {
        let cargo = cur.join("Cargo.toml");
        let renderide_crate = cur.join("crates/renderide/Cargo.toml");
        if cargo.is_file() && renderide_crate.is_file() {
            return Some(cur);
        }
        if !cur.pop() {
            break;
        }
    }
    None
}

/// When set by unit tests, [`legacy_discover_workspace_roots`] returns this list instead of
/// scanning cwd/exe.
#[cfg(test)]
pub(crate) static TEST_WORKSPACE_ROOTS_OVERRIDE: parking_lot::Mutex<Option<Vec<PathBuf>>> =
    parking_lot::Mutex::new(None);

/// When true, [`legacy_search_candidates`] stops after binary and workspace candidates.
#[cfg(test)]
pub(crate) static TEST_EXTRA_SEARCH_CANDIDATES_DISABLED: AtomicBool = AtomicBool::new(false);

/// When set by unit tests, [`legacy_binary_output_dir`] returns this path instead of
/// `current_exe()`'s parent.
#[cfg(test)]
pub(crate) static TEST_BINARY_DIR_OVERRIDE: parking_lot::Mutex<Option<PathBuf>> =
    parking_lot::Mutex::new(None);

/// Directory containing the running executable (`target/debug/`, install folder, etc.).
fn legacy_binary_output_dir() -> Option<PathBuf> {
    #[cfg(test)]
    {
        let override_value = TEST_BINARY_DIR_OVERRIDE.lock().clone();
        if let Some(p) = override_value {
            return Some(p);
        }
    }
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
}

fn legacy_discover_workspace_roots() -> Vec<PathBuf> {
    #[cfg(test)]
    {
        let override_value = TEST_WORKSPACE_ROOTS_OVERRIDE.lock().clone();
        if let Some(v) = override_value {
            return v;
        }
    }
    let mut v = Vec::new();
    if let Ok(cwd) = std::env::current_dir()
        && let Some(r) = find_renderide_workspace_root(&cwd)
    {
        v.push(r);
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
        && let Some(r) = find_renderide_workspace_root(dir)
        && !v.iter().any(|x| x == &r)
    {
        v.push(r);
    }
    v
}

fn push_toml_candidate(out: &mut Vec<PathBuf>, dir: &Path) {
    push_unique(out, dir.join(FILE_NAME_TOML));
}

/// Returns previous-layout config paths inspected during one-shot migration.
pub(super) fn legacy_search_candidates() -> Vec<PathBuf> {
    let mut v = Vec::new();

    if let Some(dir) = legacy_binary_output_dir() {
        push_toml_candidate(&mut v, dir.as_path());
        if let Some(parent) = dir.parent() {
            push_toml_candidate(&mut v, parent);
        }
    }

    for root in legacy_discover_workspace_roots() {
        push_toml_candidate(&mut v, root.as_path());
    }

    #[cfg(test)]
    {
        if TEST_EXTRA_SEARCH_CANDIDATES_DISABLED.load(Ordering::Relaxed) {
            return v;
        }
    }

    if let Ok(cwd) = std::env::current_dir() {
        push_toml_candidate(&mut v, cwd.as_path());
        if let Some(p1) = cwd.parent()
            && let Some(p2) = p1.parent()
        {
            push_toml_candidate(&mut v, p2);
        }
    }

    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    static RESOLVE_STATE_TEST_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

    /// Restores test resolution overrides when dropped so other tests see normal resolution.
    struct UserConfigOverride {
        old_cwd: PathBuf,
    }

    impl UserConfigOverride {
        /// Routes [`user_config_path`] at the provided path and disables cwd migration crawling.
        fn new(user_config: PathBuf) -> Self {
            *TEST_USER_CONFIG_OVERRIDE.lock() = Some(user_config);
            TEST_EXTRA_SEARCH_CANDIDATES_DISABLED.store(true, Ordering::Relaxed);
            *TEST_WORKSPACE_ROOTS_OVERRIDE.lock() = Some(Vec::new());
            *TEST_BINARY_DIR_OVERRIDE.lock() = Some(PathBuf::from("/nonexistent_renderide_test"));
            let old_cwd = std::env::current_dir().expect("cwd");
            Self { old_cwd }
        }

        /// Routes the user config path while allowing a previous-layout binary-dir candidate.
        fn new_with_legacy_binary_dir(user_config: PathBuf, binary_dir: PathBuf) -> Self {
            *TEST_USER_CONFIG_OVERRIDE.lock() = Some(user_config);
            TEST_EXTRA_SEARCH_CANDIDATES_DISABLED.store(true, Ordering::Relaxed);
            *TEST_WORKSPACE_ROOTS_OVERRIDE.lock() = Some(Vec::new());
            *TEST_BINARY_DIR_OVERRIDE.lock() = Some(binary_dir);
            let old_cwd = std::env::current_dir().expect("cwd");
            Self { old_cwd }
        }
    }

    impl Drop for UserConfigOverride {
        fn drop(&mut self) {
            let _ = std::env::set_current_dir(&self.old_cwd);
            TEST_EXTRA_SEARCH_CANDIDATES_DISABLED.store(false, Ordering::Relaxed);
            *TEST_USER_CONFIG_OVERRIDE.lock() = None;
            *TEST_WORKSPACE_ROOTS_OVERRIDE.lock() = None;
            *TEST_BINARY_DIR_OVERRIDE.lock() = None;
        }
    }

    #[test]
    fn find_workspace_root_from_nested() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        fs::write(root.join("Cargo.toml"), "[workspace]\n").unwrap();
        fs::create_dir_all(root.join("crates/renderide")).unwrap();
        fs::write(
            root.join("crates/renderide/Cargo.toml"),
            "[package]\nname = \"renderide\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )
        .unwrap();
        let nested = root.join("crates/renderide/src");
        fs::create_dir_all(&nested).unwrap();
        assert_eq!(
            find_renderide_workspace_root(&nested).as_deref(),
            Some(root)
        );
    }

    #[test]
    fn is_dir_writable_detects_writable_tempdir_and_rejects_non_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert!(is_dir_writable(dir.path()));
        let file = dir.path().join("not_a_dir.txt");
        fs::write(&file, b"x").unwrap();
        assert!(!is_dir_writable(&file));
        assert!(!is_dir_writable(&dir.path().join("does/not/exist")));
    }

    #[test]
    fn find_workspace_root_negative_without_renderide_crate() {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::write(dir.path().join("Cargo.toml"), "[workspace]\n").unwrap();
        assert!(find_renderide_workspace_root(dir.path()).is_none());
    }

    #[test]
    fn apply_generated_config_updates_outcome() {
        let mut outcome = ConfigResolveOutcome {
            attempted_paths: vec![],
            loaded_path: None,
            source: ConfigSource::None,
        };
        let p = PathBuf::from("/tmp/renderide_test_apply_generated/config.toml");
        apply_generated_config(&mut outcome, p.clone());
        assert_eq!(outcome.loaded_path, Some(p));
        assert_eq!(outcome.source, ConfigSource::Generated);
    }

    #[test]
    fn apply_migrated_config_updates_outcome() {
        let mut outcome = ConfigResolveOutcome {
            attempted_paths: vec![],
            loaded_path: None,
            source: ConfigSource::None,
        };
        let p = PathBuf::from("/tmp/renderide_test_apply_migrated/config.toml");
        apply_migrated_config(&mut outcome, p.clone());
        assert_eq!(outcome.loaded_path, Some(p));
        assert_eq!(outcome.source, ConfigSource::Migrated);
    }

    #[test]
    fn save_path_defaults_to_user_config_path() {
        let _state_guard = RESOLVE_STATE_TEST_LOCK.lock();
        let dir = tempfile::tempdir().expect("tempdir");
        let user_config = dir.path().join("user-config").join(FILE_NAME_TOML);
        let _iso = UserConfigOverride::new(user_config.clone());

        let resolve = ConfigResolveOutcome {
            attempted_paths: vec![],
            loaded_path: None,
            source: ConfigSource::None,
        };

        assert_eq!(resolve_save_path(&resolve), user_config);
    }

    #[test]
    fn resolve_config_path_env_file_wins_over_user_config() {
        let _state_guard = RESOLVE_STATE_TEST_LOCK.lock();
        let _env_guard = crate::config::CONFIG_ENV_TEST_LOCK.lock().expect("lock");
        let dir = tempfile::tempdir().expect("tempdir");
        let env_path = dir.path().join("env-config.toml");
        let user_config = dir.path().join("user-config").join(FILE_NAME_TOML);
        fs::create_dir_all(user_config.parent().expect("user parent")).expect("user parent");
        fs::write(&env_path, "config_version = \"0.0.0\"\n").expect("env config");
        fs::write(&user_config, "config_version = \"0.0.0\"\n").expect("user config");
        let _iso = UserConfigOverride::new(user_config);
        // SAFETY: env mutation in test; serialized via CONFIG_ENV_TEST_LOCK.
        unsafe {
            std::env::set_var(ENV_OVERRIDE, &env_path);
        }

        let resolve = resolve_config_path();

        assert_eq!(resolve.source, ConfigSource::Env);
        assert_eq!(resolve.loaded_path, Some(env_path));
        // SAFETY: env mutation in test; serialized via CONFIG_ENV_TEST_LOCK.
        unsafe {
            std::env::remove_var(ENV_OVERRIDE);
        }
    }

    #[test]
    fn load_creates_default_config_in_user_dir() {
        let _state_guard = RESOLVE_STATE_TEST_LOCK.lock();
        let _env_guard = crate::config::CONFIG_ENV_TEST_LOCK.lock().expect("lock");
        // SAFETY: env mutation in test; serialized via CONFIG_ENV_TEST_LOCK.
        unsafe {
            std::env::remove_var(ENV_OVERRIDE);
        }

        let dir = tempfile::tempdir().expect("tempdir");
        let user_config = dir.path().join("user-config").join(FILE_NAME_TOML);

        let _iso = UserConfigOverride::new(user_config.clone());
        std::env::set_current_dir(dir.path()).expect("set cwd");

        let load = crate::config::load_renderer_settings(crate::config::ConfigFilePolicy::Load);
        assert!(
            user_config.is_file(),
            "expected generated config at {}",
            user_config.display()
        );
        assert_eq!(load.resolve.loaded_path, Some(user_config.clone()));
        assert_eq!(load.resolve.source, ConfigSource::Generated);
        assert_eq!(
            load.settings,
            crate::config::RendererSettings::from_defaults()
        );
        assert_eq!(load.save_path, user_config);
    }

    #[test]
    fn invalid_env_override_blocks_default_creation_and_migration() {
        let _state_guard = RESOLVE_STATE_TEST_LOCK.lock();
        let _env_guard = crate::config::CONFIG_ENV_TEST_LOCK.lock().expect("lock");

        let dir = tempfile::tempdir().expect("tempdir");
        let legacy_dir = dir.path().join("legacy");
        fs::create_dir_all(&legacy_dir).expect("legacy dir");
        let legacy_path = legacy_dir.join(FILE_NAME_TOML);
        fs::write(&legacy_path, "config_version = \"0.0.0\"\n").expect("legacy config");
        let missing_env_path = dir.path().join("missing-env.toml");
        let user_config = dir.path().join("user-config").join(FILE_NAME_TOML);
        let _iso = UserConfigOverride::new_with_legacy_binary_dir(user_config.clone(), legacy_dir);
        std::env::set_current_dir(dir.path()).expect("set cwd");
        // SAFETY: env mutation in test; serialized via CONFIG_ENV_TEST_LOCK.
        unsafe {
            std::env::set_var(ENV_OVERRIDE, &missing_env_path);
        }

        let load = crate::config::load_renderer_settings(crate::config::ConfigFilePolicy::Load);

        assert_eq!(load.resolve.source, ConfigSource::None);
        assert!(load.resolve.loaded_path.is_none());
        assert!(!user_config.exists(), "user config should not be created");
        assert!(legacy_path.exists(), "legacy config should not be migrated");
        // SAFETY: env mutation in test; serialized via CONFIG_ENV_TEST_LOCK.
        unsafe {
            std::env::remove_var(ENV_OVERRIDE);
        }
    }

    #[test]
    fn existing_user_config_prevents_legacy_migration() {
        let _state_guard = RESOLVE_STATE_TEST_LOCK.lock();
        let _env_guard = crate::config::CONFIG_ENV_TEST_LOCK.lock().expect("lock");
        // SAFETY: env mutation in test; serialized via CONFIG_ENV_TEST_LOCK.
        unsafe {
            std::env::remove_var(ENV_OVERRIDE);
        }

        let dir = tempfile::tempdir().expect("tempdir");
        let legacy_dir = dir.path().join("legacy");
        fs::create_dir_all(&legacy_dir).expect("legacy dir");
        let legacy_path = legacy_dir.join(FILE_NAME_TOML);
        let version = crate::config::RendererSettings::CURRENT_CONFIG_VERSION;
        fs::write(
            &legacy_path,
            format!("config_version = \"{version}\"\n\n[display]\nfocused_fps = 137\n"),
        )
        .expect("legacy config");

        let user_config = dir.path().join("user-config").join(FILE_NAME_TOML);
        fs::create_dir_all(user_config.parent().expect("user parent")).expect("user parent");
        fs::write(
            &user_config,
            format!("config_version = \"{version}\"\n\n[display]\nfocused_fps = 91\n"),
        )
        .expect("user config");
        let _iso = UserConfigOverride::new_with_legacy_binary_dir(user_config.clone(), legacy_dir);
        std::env::set_current_dir(dir.path()).expect("set cwd");

        let load = crate::config::load_renderer_settings(crate::config::ConfigFilePolicy::Load);

        assert_eq!(load.resolve.source, ConfigSource::Search);
        assert_eq!(load.resolve.loaded_path, Some(user_config));
        assert_eq!(load.settings.display.focused_fps_cap, 91);
        assert!(
            legacy_path.exists(),
            "legacy config should remain untouched"
        );
    }

    #[test]
    fn load_migrates_legacy_config_from_binary_dir() {
        let _state_guard = RESOLVE_STATE_TEST_LOCK.lock();
        let _env_guard = crate::config::CONFIG_ENV_TEST_LOCK.lock().expect("lock");
        // SAFETY: env mutation in test; serialized via CONFIG_ENV_TEST_LOCK.
        unsafe {
            std::env::remove_var(ENV_OVERRIDE);
        }

        let dir = tempfile::tempdir().expect("tempdir");
        let legacy_dir = dir.path().join("legacy-binary-dir");
        fs::create_dir_all(&legacy_dir).expect("legacy dir");
        let legacy_path = legacy_dir.join(FILE_NAME_TOML);
        let version = crate::config::RendererSettings::CURRENT_CONFIG_VERSION;
        fs::write(
            &legacy_path,
            format!("config_version = \"{version}\"\n\n[display]\nfocused_fps = 137\n"),
        )
        .expect("legacy config");

        let user_config = dir.path().join("user-config").join(FILE_NAME_TOML);
        let _iso =
            UserConfigOverride::new_with_legacy_binary_dir(user_config.clone(), legacy_dir.clone());
        std::env::set_current_dir(dir.path()).expect("set cwd");

        let load = crate::config::load_renderer_settings(crate::config::ConfigFilePolicy::Load);

        assert_eq!(load.resolve.source, ConfigSource::Migrated);
        assert_eq!(load.resolve.loaded_path, Some(user_config.clone()));
        assert!(user_config.is_file(), "user config should exist");
        let migrated_contents = fs::read_to_string(&user_config).expect("read user config");
        assert!(
            migrated_contents.contains("focused_fps = 137"),
            "migrated content should match legacy:\n{migrated_contents}"
        );
        assert!(!legacy_path.exists(), "legacy file should be tombstoned");
        let tombstone = legacy_dir.join(format!("{FILE_NAME_TOML}{LEGACY_MIGRATED_SUFFIX}"));
        assert!(tombstone.is_file(), "expected tombstone");
        assert_eq!(load.settings.display.focused_fps_cap, 137);
    }
}
