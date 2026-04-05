//! Locate `config.ini`: `RENDERIDE_CONFIG`, then standard search paths.

use std::path::{Path, PathBuf};

/// How the INI path was chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConfigSource {
    /// `RENDERIDE_CONFIG` pointed at an existing file.
    Env,
    /// First hit among exe-adjacent / cwd searches.
    Search,
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

fn push_unique(out: &mut Vec<PathBuf>, p: PathBuf) {
    if !out.iter().any(|x| x == &p) {
        out.push(p);
    }
}

fn search_candidates() -> Vec<PathBuf> {
    let mut v = Vec::new();

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
