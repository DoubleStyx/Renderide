//! Resonite install detection and discovery ordering.

use std::env;
use std::path::{Path, PathBuf};

use super::steam;

/// Steam app folder name for Resonite.
pub const RESONITE_APP_NAME: &str = "Resonite";
/// Windows host launcher (native / Wine).
pub const RENDERITE_HOST_EXE: &str = "Renderite.Host.exe";
/// Host assembly for `dotnet` launch.
pub const RENDERITE_HOST_DLL: &str = "Renderite.Host.dll";

/// Returns true if `dir` looks like a Resonite root (host exe or host DLL present).
pub fn is_resonite_install_dir(dir: &Path) -> bool {
    dir.join(RENDERITE_HOST_EXE).exists() || dir.join(RENDERITE_HOST_DLL).exists()
}

/// Candidate Resonite roots in discovery order (env, Steam layout, library folders).
fn resonite_candidate_dirs() -> impl Iterator<Item = PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    if let Ok(dir) = env::var("RESONITE_DIR") {
        out.push(PathBuf::from(dir));
    }
    if let Ok(steam_root) = env::var("STEAM_PATH") {
        out.push(
            PathBuf::from(steam_root)
                .join("steamapps")
                .join("common")
                .join(RESONITE_APP_NAME),
        );
    }
    let bases = steam::base_paths();
    for steam_base in &bases {
        out.push(
            steam_base
                .join("steamapps")
                .join("common")
                .join(RESONITE_APP_NAME),
        );
    }
    for steam_base in &bases {
        for lib_path in steam::library_paths_from_vdf(steam_base) {
            out.push(
                lib_path
                    .join("steamapps")
                    .join("common")
                    .join(RESONITE_APP_NAME),
            );
        }
    }
    out.into_iter()
}

/// Finds the Resonite installation directory.
///
/// Order: `RESONITE_DIR`, `STEAM_PATH` + `steamapps/common/Resonite`, platform Steam roots,
/// then libraries from `libraryfolders.vdf`.
pub fn find_resonite_dir() -> Option<PathBuf> {
    resonite_candidate_dirs().find(|p| is_resonite_install_dir(p))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn is_resonite_install_dir_requires_host_artifact() {
        let tmp = env::temp_dir().join(format!("bootstrapper_paths_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        assert!(!is_resonite_install_dir(&tmp));
        fs::write(tmp.join(RENDERITE_HOST_DLL), b"").unwrap();
        assert!(is_resonite_install_dir(&tmp));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn find_resonite_dir_env_override() {
        let _g = crate::test_env::lock_process_env();
        let _snap = crate::test_env::EnvSnapshot::capture(&["RESONITE_DIR"]);
        let tmp = env::temp_dir().join(format!("bootstrapper_resonite_env_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        fs::write(tmp.join(RENDERITE_HOST_DLL), b"").unwrap();
        // SAFETY: env mutation in test; serialized via the process env test lock.
        unsafe {
            env::set_var("RESONITE_DIR", &tmp);
        }
        let got = find_resonite_dir();
        assert_eq!(got, Some(tmp.clone()));
        let _ = fs::remove_dir_all(&tmp);
    }
}
