use std::path::{Path, PathBuf};

use super::parser::build_manifest;
use super::types::{Manifest, ManifestError};

/// Environment override: set to a directory containing `actions.toml` and `bindings/` to bypass search.
const ENV_XR_ASSETS: &str = "RENDERIDE_XR_ASSETS";
/// Canonical action manifest filename inside the XR assets directory.
const ACTIONS_FILE: &str = "actions.toml";
/// Canonical bindings subdirectory inside the XR assets directory.
const BINDINGS_DIR: &str = "bindings";
/// Directory name containing the XR actions manifest and bindings
const XR_DIR: &str = "xr";

/// Enumerates directories that might contain `actions.toml` plus `bindings/`.
fn xr_assets_search_candidates() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    let push_unique = |v: &mut Vec<PathBuf>, p: PathBuf| {
        if !v.iter().any(|x| x == &p) {
            v.push(p);
        }
    };

    if let Ok(raw) = std::env::var(ENV_XR_ASSETS) {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            push_unique(&mut out, PathBuf::from(trimmed));
        }
    }

    if let Ok(cwd) = std::env::current_dir() {
        if let Some(root) = crate::config::find_renderide_workspace_root(&cwd) {
            push_unique(&mut out, root.join("crates/renderide/assets/xr"));
        }
        push_unique(&mut out, cwd.join(XR_DIR));
    }

    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        if let Some(root) = crate::config::find_renderide_workspace_root(dir) {
            push_unique(&mut out, root.join("crates/renderide/assets/xr"));
        }
        push_unique(&mut out, dir.join(XR_DIR));
        if let Some(parent) = dir.parent() {
            push_unique(&mut out, parent.join(XR_DIR));
        }
    }

    #[cfg(target_family = "unix")]
    {
        push_unique(&mut out, PathBuf::from("/usr/share/renderide").join(XR_DIR));
    }

    out
}

/// Chosen XR assets directory.
#[derive(Clone, Debug)]
pub struct XrAssetsLocation {
    /// Directory containing `actions.toml` and `bindings/`.
    pub root: PathBuf,
}

/// Locates the XR assets directory, returning the first candidate that has an `actions.toml`.
pub fn resolve_xr_assets_dir() -> Result<XrAssetsLocation, ManifestError> {
    let attempted = xr_assets_search_candidates();
    for candidate in &attempted {
        if candidate.join(ACTIONS_FILE).is_file() {
            return Ok(XrAssetsLocation {
                root: candidate.clone(),
            });
        }
    }
    Err(ManifestError::ActionsManifestMissing {
        searched: attempted.iter().map(|p| p.display().to_string()).collect(),
    })
}

/// Reads one file from disk, wrapping io errors with the path for diagnostics.
fn read_file(path: &Path) -> Result<String, ManifestError> {
    std::fs::read_to_string(path).map_err(|e| ManifestError::Io {
        path: path.display().to_string(),
        source: e,
    })
}

/// Lists every `.toml` file inside a `bindings/` directory, sorted for deterministic diagnostics.
fn list_binding_files(dir: &Path) -> Result<Vec<PathBuf>, ManifestError> {
    if !dir.is_dir() {
        return Err(ManifestError::BindingsDirMissing {
            path: dir.display().to_string(),
        });
    }
    let mut files = Vec::new();
    let entries = std::fs::read_dir(dir).map_err(|e| ManifestError::Io {
        path: dir.display().to_string(),
        source: e,
    })?;
    for entry in entries {
        let entry = entry.map_err(|e| ManifestError::Io {
            path: dir.display().to_string(),
            source: e,
        })?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("toml") {
            files.push(path);
        }
    }
    if files.is_empty() {
        return Err(ManifestError::BindingsDirMissing {
            path: dir.display().to_string(),
        });
    }
    files.sort();
    Ok(files)
}

/// Reads and validates the action + binding manifests from the resolved XR assets directory.
pub fn load_manifest() -> Result<(Manifest, XrAssetsLocation), ManifestError> {
    let location = resolve_xr_assets_dir()?;
    let actions_path = location.root.join(ACTIONS_FILE);
    let actions_src = read_file(&actions_path)?;

    let bindings_dir = location.root.join(BINDINGS_DIR);
    let binding_paths = list_binding_files(&bindings_dir)?;

    let mut sources: Vec<(String, String)> = Vec::with_capacity(binding_paths.len());
    for path in &binding_paths {
        let src = read_file(path)?;
        sources.push((path.display().to_string(), src));
    }

    let source_refs: Vec<(&str, &str)> = sources
        .iter()
        .map(|(label, src)| (label.as_str(), src.as_str()))
        .collect();

    let manifest = build_manifest(&actions_src, &source_refs)?;
    Ok((manifest, location))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn list_binding_files_rejects_missing_directory() {
        let root = tempfile::tempdir().expect("tempdir");
        let missing = root.path().join("bindings");

        assert!(matches!(
            list_binding_files(&missing),
            Err(ManifestError::BindingsDirMissing { .. })
        ));
    }

    #[test]
    fn list_binding_files_rejects_empty_directory() {
        let root = tempfile::tempdir().expect("tempdir");
        let bindings = root.path().join("bindings");
        fs::create_dir_all(&bindings).expect("create bindings");

        assert!(matches!(
            list_binding_files(&bindings),
            Err(ManifestError::BindingsDirMissing { .. })
        ));
    }

    #[test]
    fn list_binding_files_returns_sorted_toml_paths_only() {
        let root = tempfile::tempdir().expect("tempdir");
        let bindings = root.path().join("bindings");
        fs::create_dir_all(&bindings).expect("create bindings");
        fs::write(bindings.join("z.toml"), "").expect("write z");
        fs::write(bindings.join("a.toml"), "").expect("write a");
        fs::write(bindings.join("ignored.txt"), "").expect("write txt");

        let files = list_binding_files(&bindings).expect("binding files");
        let names: Vec<_> = files
            .iter()
            .map(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .expect("file name")
            })
            .collect();

        assert_eq!(names, ["a.toml", "z.toml"]);
    }
}
