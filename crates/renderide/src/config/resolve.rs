//! Locate `config.ini`: `RENDERIDE_CONFIG`, then standard search paths.

use std::path::{Path, PathBuf};

/// How the INI path was chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConfigSource {
    /// `RENDERIDE_CONFIG` pointed at an existing file.
    Env,
    /// First hit among exe-adjacent / cwd searches.
    Search,
    /// No existing file; defaults were written to the save path on first load.
    Generated,
    /// No file found; caller uses defaults only.
    None,
}

/// Result of resolving a config path (whether or not a file was read).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigResolveOutcome {
    /// Every path checked, in order (`RENDERIDE_CONFIG` first when set, then search candidates).
    pub attempted_paths: Vec<PathBuf>,
    /// First existing regular file used for INI content.
    pub loaded_path: Option<PathBuf>,
    pub source: ConfigSource,
}

const FILE_NAME: &str = "config.ini";
const ENV_OVERRIDE: &str = "RENDERIDE_CONFIG";

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

/// When set by unit tests, [`discover_workspace_roots`] returns this list instead of scanning cwd/exe.
#[cfg(test)]
pub(crate) static TEST_WORKSPACE_ROOTS_OVERRIDE: std::sync::Mutex<Option<Vec<PathBuf>>> =
    std::sync::Mutex::new(None);

/// When true, [`search_candidates`] returns only workspace-root paths (used to isolate tests from the real repo).
#[cfg(test)]
pub(crate) static TEST_EXTRA_SEARCH_CANDIDATES_DISABLED: std::sync::Mutex<bool> =
    std::sync::Mutex::new(false);

fn discover_workspace_roots() -> Vec<PathBuf> {
    #[cfg(test)]
    {
        if let Ok(g) = TEST_WORKSPACE_ROOTS_OVERRIDE.lock() {
            if let Some(v) = g.clone() {
                return v;
            }
        }
    }
    let mut v = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        if let Some(r) = find_renderide_workspace_root(&cwd) {
            v.push(r);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            if let Some(r) = find_renderide_workspace_root(dir) {
                if !v.iter().any(|x| x == &r) {
                    v.push(r);
                }
            }
        }
    }
    v
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

/// Records that `config.ini` was created at `path` on first load (see [`super::settings::load_renderer_settings`]).
pub fn apply_generated_config(outcome: &mut ConfigResolveOutcome, path: PathBuf) {
    push_unique(&mut outcome.attempted_paths, path.clone());
    outcome.loaded_path = Some(path);
    outcome.source = ConfigSource::Generated;
}

fn search_candidates() -> Vec<PathBuf> {
    let mut v = Vec::new();

    for root in discover_workspace_roots() {
        v.push(root.join(FILE_NAME));
    }

    #[cfg(test)]
    {
        if let Ok(g) = TEST_EXTRA_SEARCH_CANDIDATES_DISABLED.lock() {
            if *g {
                return v;
            }
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            v.push(dir.join(FILE_NAME));
            if let Some(parent) = dir.parent() {
                v.push(parent.join(FILE_NAME));
            }
        }
    }

    if let Ok(cwd) = std::env::current_dir() {
        v.push(cwd.join(FILE_NAME));
        if let Some(p1) = cwd.parent() {
            if let Some(p2) = p1.parent() {
                v.push(p2.join(FILE_NAME));
            }
        }
    }

    v
}

/// Resolves the config file path. If `RENDERIDE_CONFIG` is set but missing, logs a warning and
/// continues with the search list.
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
                "{ENV_OVERRIDE}={} does not exist or is not a file; trying default locations",
                p.display()
            );
        }
    }

    for p in search_candidates() {
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
/// - Otherwise: prefer a discovered workspace root `config.ini` when that directory is writable;
///   then `current_dir()/config.ini` when the directory exists and is writable; else the first
///   path in the same search order as [`resolve_config_path`] whose parent exists and is writable.
pub fn resolve_save_path(resolve: &ConfigResolveOutcome) -> PathBuf {
    if let Some(p) = resolve.loaded_path.clone() {
        return p;
    }

    for root in discover_workspace_roots() {
        if is_dir_writable(root.as_path()) {
            return root.join(FILE_NAME);
        }
    }

    if let Ok(cwd) = std::env::current_dir() {
        let p = cwd.join(FILE_NAME);
        if is_dir_writable(cwd.as_path()) {
            return p;
        }
    }

    for p in search_candidates() {
        if let Some(parent) = p.parent() {
            if parent.as_os_str().is_empty() {
                continue;
            }
            if is_dir_writable(parent) {
                return p;
            }
        }
    }

    // Last resort: cwd join even if we could not verify writability (save may fail at runtime).
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(FILE_NAME)
}

/// Best-effort writable check for choosing where to create `config.ini`.
pub(crate) fn is_dir_writable(dir: &Path) -> bool {
    if !dir.is_dir() {
        return false;
    }
    // Best-effort: create a probe file (same approach as `access` is not fully portable for ACLs).
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::Mutex;

    static CWD_TEST_LOCK: Mutex<()> = Mutex::new(());

    /// Restores test search overrides when dropped so other tests see normal resolution.
    struct TestSearchIsolation {
        old_cwd: PathBuf,
    }

    impl TestSearchIsolation {
        fn new(temp_root: PathBuf) -> Self {
            *TEST_EXTRA_SEARCH_CANDIDATES_DISABLED.lock().unwrap() = true;
            *TEST_WORKSPACE_ROOTS_OVERRIDE.lock().unwrap() = Some(vec![temp_root]);
            let old_cwd = std::env::current_dir().expect("cwd");
            Self { old_cwd }
        }
    }

    impl Drop for TestSearchIsolation {
        fn drop(&mut self) {
            let _ = std::env::set_current_dir(&self.old_cwd);
            *TEST_EXTRA_SEARCH_CANDIDATES_DISABLED.lock().unwrap() = false;
            *TEST_WORKSPACE_ROOTS_OVERRIDE.lock().unwrap() = None;
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
        let p = PathBuf::from("/tmp/renderide_test_apply_generated/config.ini");
        apply_generated_config(&mut outcome, p.clone());
        assert_eq!(outcome.loaded_path, Some(p));
        assert_eq!(outcome.source, ConfigSource::Generated);
    }

    #[test]
    fn load_creates_default_config_in_workspace() {
        let _guard = CWD_TEST_LOCK.lock().expect("lock");
        std::env::remove_var(ENV_OVERRIDE);

        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_path_buf();
        fs::write(root.join("Cargo.toml"), "[workspace]\n").unwrap();
        fs::create_dir_all(root.join("crates/renderide")).unwrap();
        fs::write(
            root.join("crates/renderide/Cargo.toml"),
            "[package]\nname = \"renderide\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )
        .unwrap();

        let _iso = TestSearchIsolation::new(root.clone());
        std::env::set_current_dir(&root).expect("set cwd");

        let load = crate::config::load_renderer_settings();
        let path = root.join(FILE_NAME);
        assert!(
            path.is_file(),
            "expected generated config at {}",
            path.display()
        );
        assert_eq!(load.resolve.loaded_path, Some(path.clone()));
        assert_eq!(load.resolve.source, ConfigSource::Generated);
        assert_eq!(
            load.settings,
            crate::config::RendererSettings::from_defaults()
        );
        assert_eq!(load.save_path, path);
    }
}
