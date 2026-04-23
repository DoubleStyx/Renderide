//! Resonite installation and `dotnet` discovery (Steam, env vars, registry on Windows).

use std::env;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// Steam app folder name for Resonite.
pub(crate) const RESONITE_APP_NAME: &str = "Resonite";
/// Windows host launcher (native / Wine).
pub(crate) const RENDERITE_HOST_EXE: &str = "Renderite.Host.exe";
/// Host assembly for `dotnet` launch.
pub(crate) const RENDERITE_HOST_DLL: &str = "Renderite.Host.dll";

/// Returns true if `dir` looks like a Resonite root (host exe or host DLL present).
pub(crate) fn is_resonite_install_dir(dir: &Path) -> bool {
    dir.join(RENDERITE_HOST_EXE).exists() || dir.join(RENDERITE_HOST_DLL).exists()
}

/// Prefers bundled `dotnet-runtime` under the Resonite folder (`dotnet.exe` then `dotnet` on Windows),
/// else `dotnet` on `PATH`.
pub(crate) fn find_dotnet_for_host(resonite_dir: &Path) -> PathBuf {
    let candidates: Vec<PathBuf> = {
        #[cfg(windows)]
        {
            vec![
                resonite_dir.join("dotnet-runtime").join("dotnet.exe"),
                resonite_dir.join("dotnet-runtime").join("dotnet"),
            ]
        }
        #[cfg(not(windows))]
        {
            vec![resonite_dir.join("dotnet-runtime").join("dotnet")]
        }
    };
    for c in candidates {
        if c.exists() {
            return c;
        }
    }
    PathBuf::from("dotnet")
}

/// Extracts `"path"` values from Steam's `libraryfolders.vdf` under `steam_base`.
fn parse_libraryfolders_vdf(steam_base: &Path) -> Vec<PathBuf> {
    let vdf_path = steam_base.join("steamapps").join("libraryfolders.vdf");
    let Ok(file) = fs::File::open(&vdf_path) else {
        return Vec::new();
    };
    let mut paths = Vec::new();
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        if let Some(idx) = line.find("\"path\"") {
            let rest = line[idx + 6..].trim_start_matches(['\t', ' ']);
            if let Some(start) = rest.find('"') {
                let inner = &rest[start + 1..];
                if let Some(end) = inner.find('"') {
                    paths.push(PathBuf::from(&inner[..end]));
                }
            }
        }
    }
    paths
}

/// Candidate Resonite roots in discovery order (env, Steam layout, library folders).
fn resonite_candidate_dirs() -> impl Iterator<Item = PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    if let Ok(dir) = env::var("RESONITE_DIR") {
        out.push(PathBuf::from(dir));
    }
    if let Ok(steam) = env::var("STEAM_PATH") {
        out.push(
            PathBuf::from(steam)
                .join("steamapps")
                .join("common")
                .join(RESONITE_APP_NAME),
        );
    }
    let bases = steam_base_paths();
    for steam_base in &bases {
        out.push(
            steam_base
                .join("steamapps")
                .join("common")
                .join(RESONITE_APP_NAME),
        );
    }
    for steam_base in &bases {
        for lib_path in parse_libraryfolders_vdf(steam_base) {
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
pub(crate) fn find_resonite_dir() -> Option<PathBuf> {
    resonite_candidate_dirs().find(|p| is_resonite_install_dir(p))
}

/// Returns likely Steam installation roots for the current platform (env vars and registry on Windows).
fn steam_base_paths() -> Vec<PathBuf> {
    #[cfg(windows)]
    {
        let mut bases = Vec::new();
        if let Ok(steam) = env::var("STEAM_PATH") {
            bases.push(PathBuf::from(steam));
        }
        if let Ok(path) = steam_path_from_registry() {
            if !bases.iter().any(|b| b == &path) {
                bases.push(path);
            }
        }
        for env_var in ["ProgramFiles(x86)", "ProgramFiles"] {
            if let Ok(pf) = env::var(env_var) {
                let steam = PathBuf::from(pf).join("Steam");
                if !bases.contains(&steam) {
                    bases.push(steam);
                }
            }
        }
        if let Ok(local) = env::var("LOCALAPPDATA") {
            let steam = PathBuf::from(local).join("Steam");
            if !bases.contains(&steam) {
                bases.push(steam);
            }
        }
        bases
    }

    #[cfg(target_os = "macos")]
    {
        let Some(home) = env::var_os("HOME").map(PathBuf::from) else {
            return Vec::new();
        };
        vec![
            home.join("Library")
                .join("Application Support")
                .join("Steam"),
            home.join(".steam").join("steam"),
            home.join(".local").join("share").join("Steam"),
        ]
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let Some(home) = env::var_os("HOME").map(PathBuf::from) else {
            return Vec::new();
        };
        vec![
            home.join(".local").join("share").join("Steam"),
            home.join(".steam").join("steam"),
        ]
    }

    #[cfg(not(any(windows, unix)))]
    {
        compile_error!("bootstrapper paths require unix or windows");
    }
}

/// Reads the Steam install path from `HKLM\...\Valve\Steam` when `InstallPath` is present.
#[cfg(windows)]
fn steam_path_from_registry() -> Result<PathBuf, std::io::Error> {
    use winreg::enums::HKEY_LOCAL_MACHINE;
    use winreg::RegKey;

    let hklm = RegKey::predef(HKEY_LOCAL_MACHINE);
    for key_path in &[r"SOFTWARE\WOW6432Node\Valve\Steam", r"SOFTWARE\Valve\Steam"] {
        if let Ok(steam_key) = hklm.open_subkey(key_path) {
            if let Ok(install_path) = steam_key.get_value::<String, &str>("InstallPath") {
                return Ok(PathBuf::from(install_path));
            }
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "Steam path not found in registry",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn parse_libraryfolders_vdf_extracts_quoted_paths() {
        let tmp = env::temp_dir().join(format!(
            "bootstrapper_libraryfolders_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(tmp.join("steamapps")).unwrap();
        let vdf = tmp.join("steamapps").join("libraryfolders.vdf");
        let mut f = fs::File::create(&vdf).unwrap();
        writeln!(f, r#" "path" "/data/SteamLibrary" "#,).unwrap();
        let paths = parse_libraryfolders_vdf(&tmp);
        assert!(
            paths
                .iter()
                .any(|p| p == Path::new("/data/SteamLibrary")),
            "paths={paths:?}"
        );
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn parse_libraryfolders_vdf_multiple_paths_and_garbage() {
        let tmp = env::temp_dir().join(format!(
            "bootstrapper_libraryfolders_multi_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(tmp.join("steamapps")).unwrap();
        let vdf = tmp.join("steamapps").join("libraryfolders.vdf");
        let mut f = fs::File::create(&vdf).unwrap();
        writeln!(f, "garbage line").unwrap();
        writeln!(f, r#"	"path"		"/first/lib" "#).unwrap();
        writeln!(f, r#" "path" "/Volumes/My Disk/SteamLibrary" "#).unwrap();
        let paths = parse_libraryfolders_vdf(&tmp);
        assert!(paths.iter().any(|p| p == Path::new("/first/lib")));
        assert!(paths
            .iter()
            .any(|p| p == Path::new("/Volumes/My Disk/SteamLibrary")));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[cfg(unix)]
    #[test]
    fn find_dotnet_for_host_prefers_bundled_dotnet() {
        let tmp =
            env::temp_dir().join(format!("bootstrapper_dotnet_unix_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        let bundled = tmp.join("dotnet-runtime").join("dotnet");
        fs::create_dir_all(bundled.parent().unwrap()).unwrap();
        fs::write(&bundled, b"").unwrap();
        assert_eq!(find_dotnet_for_host(&tmp), bundled);
        let _ = fs::remove_dir_all(&tmp);
    }

    #[cfg(windows)]
    #[test]
    fn find_dotnet_for_host_prefers_bundled_exe() {
        let tmp =
            std::env::temp_dir().join(format!("bootstrapper_dotnet_win_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        let bundled_exe = tmp.join("dotnet-runtime").join("dotnet.exe");
        fs::create_dir_all(bundled_exe.parent().unwrap()).unwrap();
        fs::write(&bundled_exe, b"").unwrap();
        assert_eq!(find_dotnet_for_host(&tmp), bundled_exe);
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn find_dotnet_for_host_falls_back_without_bundled() {
        let tmp = env::temp_dir().join(format!(
            "bootstrapper_dotnet_fallback_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        assert_eq!(find_dotnet_for_host(&tmp), PathBuf::from("dotnet"));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn is_resonite_install_dir_requires_host_artifact() {
        let tmp = env::temp_dir().join(format!("bootstrapper_paths_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        assert!(!is_resonite_install_dir(&tmp));
        std::fs::write(tmp.join(RENDERITE_HOST_DLL), b"").unwrap();
        assert!(is_resonite_install_dir(&tmp));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn find_resonite_dir_env_override() {
        let _g = ENV_LOCK.lock().expect("env lock");
        let tmp =
            env::temp_dir().join(format!("bootstrapper_resonite_env_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        fs::write(tmp.join(RENDERITE_HOST_DLL), b"").unwrap();
        env::set_var("RESONITE_DIR", &tmp);
        let got = find_resonite_dir();
        env::remove_var("RESONITE_DIR");
        assert_eq!(got, Some(tmp.clone()));
        let _ = fs::remove_dir_all(&tmp);
    }
}
