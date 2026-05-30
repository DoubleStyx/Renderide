//! Resonite install detection and discovery ordering.

use std::env;
use std::io;
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

/// Candidate Resonite roots in discovery order after explicit path checks.
fn discovered_resonite_candidate_dirs() -> impl Iterator<Item = PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
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

/// Returns the first valid Resonite path from the current environment.
fn env_resonite_dir() -> Option<PathBuf> {
    env::var_os("RESONITE_DIR")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

/// Validates a user-supplied Resonite path and returns an actionable error when it is invalid.
fn validate_explicit_resonite_dir(path: &Path, source: &str) -> io::Result<PathBuf> {
    if is_resonite_install_dir(path) {
        return Ok(path.to_path_buf());
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        format!(
            "{source} points to {}, but that directory does not contain {RENDERITE_HOST_DLL} or {RENDERITE_HOST_EXE}",
            path.display()
        ),
    ))
}

/// Resolves the Resonite installation directory.
///
/// Order: CLI `--resonite-dir` / `-ResoniteDir`, `RESONITE_DIR`, `STEAM_PATH` +
/// `steamapps/common/Resonite`, platform Steam roots, then libraries from `libraryfolders.vdf`.
/// Explicit CLI and environment paths fail with an actionable error when invalid.
pub fn resolve_resonite_dir(explicit_dir: Option<&Path>) -> io::Result<PathBuf> {
    if let Some(path) = explicit_dir {
        return validate_explicit_resonite_dir(path, "--resonite-dir");
    }
    if let Some(path) = env_resonite_dir() {
        return validate_explicit_resonite_dir(&path, "RESONITE_DIR");
    }
    discovered_resonite_candidate_dirs()
        .find(|p| is_resonite_install_dir(p))
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                "Could not find Resonite installation. Set --resonite-dir or RESONITE_DIR, or ensure Steam has Resonite installed.",
            )
        })
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
    fn resolve_resonite_dir_env_override() {
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
        let got = resolve_resonite_dir(None).expect("env path should resolve");
        assert_eq!(got, tmp);
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn resolve_resonite_dir_cli_override_wins_over_env() {
        let _g = crate::test_env::lock_process_env();
        let _snap = crate::test_env::EnvSnapshot::capture(&["RESONITE_DIR"]);
        let base =
            env::temp_dir().join(format!("bootstrapper_resonite_cli_{}", std::process::id()));
        let cli = base.join("cli");
        let env_dir = base.join("env");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&cli).unwrap();
        fs::create_dir_all(&env_dir).unwrap();
        fs::write(cli.join(RENDERITE_HOST_DLL), b"").unwrap();
        fs::write(env_dir.join(RENDERITE_HOST_DLL), b"").unwrap();
        // SAFETY: env mutation in test; serialized via the process env test lock.
        unsafe {
            env::set_var("RESONITE_DIR", &env_dir);
        }

        let got = resolve_resonite_dir(Some(&cli)).expect("cli path should resolve");

        assert_eq!(got, cli);
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn resolve_resonite_dir_rejects_invalid_cli_override() {
        let tmp = env::temp_dir().join(format!(
            "bootstrapper_resonite_bad_cli_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();

        let err = resolve_resonite_dir(Some(&tmp)).expect_err("invalid explicit path");

        assert_eq!(err.kind(), io::ErrorKind::NotFound);
        assert!(err.to_string().contains("--resonite-dir"));
        assert!(err.to_string().contains(RENDERITE_HOST_DLL));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn resolve_resonite_dir_rejects_invalid_env_override() {
        let _g = crate::test_env::lock_process_env();
        let _snap = crate::test_env::EnvSnapshot::capture(&["RESONITE_DIR"]);
        let tmp = env::temp_dir().join(format!(
            "bootstrapper_resonite_bad_env_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        // SAFETY: env mutation in test; serialized via the process env test lock.
        unsafe {
            env::set_var("RESONITE_DIR", &tmp);
        }

        let err = resolve_resonite_dir(None).expect_err("invalid env path");

        assert_eq!(err.kind(), io::ErrorKind::NotFound);
        assert!(err.to_string().contains("RESONITE_DIR"));
        let _ = fs::remove_dir_all(&tmp);
    }
}
