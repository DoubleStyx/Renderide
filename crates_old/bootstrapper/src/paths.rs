//! Path discovery for Resonite installation and dotnet.
//! Searches RESONITE_DIR, STEAM_PATH, Steam registry (Windows), and libraryfolders.vdf.

use std::env;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// Steam app name for Resonite.
pub const RESONITE_APP_NAME: &str = "Resonite";
/// Host executable name (Windows/native).
pub const RENDERITE_HOST_EXE: &str = "Renderite.Host.exe";
/// Host DLL for dotnet on Linux.
pub const RENDERITE_HOST_DLL: &str = "Renderite.Host.dll";

/// Finds the dotnet executable to run Renderite.Host.dll.
/// Prefers bundled dotnet-runtime in the Resonite folder (matches Linux behavior).
#[cfg(unix)]
pub fn find_dotnet_for_host(resonite_dir: &Path) -> PathBuf {
    let bundled = resonite_dir.join("dotnet-runtime").join("dotnet");
    if bundled.exists() {
        bundled
    } else {
        PathBuf::from("dotnet")
    }
}

#[cfg(windows)]
pub fn find_dotnet_for_host(resonite_dir: &Path) -> PathBuf {
    let bundled_exe = resonite_dir.join("dotnet-runtime").join("dotnet.exe");
    let bundled = resonite_dir.join("dotnet-runtime").join("dotnet");
    if bundled_exe.exists() {
        bundled_exe
    } else if bundled.exists() {
        bundled
    } else {
        PathBuf::from("dotnet")
    }
}

/// Paths of Steam library folders parsed from libraryfolders.vdf.
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

/// Finds the Resonite installation directory by searching Steam libraries.
/// Checks: RESONITE_DIR env, STEAM_PATH env, platform-specific Steam paths, libraryfolders.vdf.
pub fn find_resonite_dir() -> Option<PathBuf> {
    let host_exe = |dir: &Path| dir.join(RENDERITE_HOST_EXE).exists();

    if let Ok(dir) = env::var("RESONITE_DIR") {
        let path = PathBuf::from(&dir);
        if host_exe(&path) {
            return Some(path);
        }
    }

    if let Ok(steam) = env::var("STEAM_PATH") {
        let path = PathBuf::from(&steam)
            .join("steamapps")
            .join("common")
            .join(RESONITE_APP_NAME);
        if host_exe(&path) {
            return Some(path);
        }
    }

    let steam_bases = steam_base_paths();

    for steam_base in &steam_bases {
        let path = steam_base
            .join("steamapps")
            .join("common")
            .join(RESONITE_APP_NAME);
        if host_exe(&path) {
            return Some(path);
        }
    }

    for steam_base in &steam_bases {
        for lib_path in parse_libraryfolders_vdf(steam_base) {
            let resonite = lib_path
                .join("steamapps")
                .join("common")
                .join(RESONITE_APP_NAME);
            if host_exe(&resonite) {
                return Some(resonite);
            }
        }
    }

    None
}

/// Returns Steam installation base paths to search (platform-specific).
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

    #[cfg(not(windows))]
    {
        let home = match env::var("HOME") {
            Ok(h) => PathBuf::from(h),
            Err(_) => return Vec::new(),
        };
        vec![
            home.join(".local").join("share").join("Steam"),
            home.join(".steam").join("steam"),
        ]
    }
}

#[cfg(windows)]
fn steam_path_from_registry() -> Result<PathBuf, std::io::Error> {
    use winreg::RegKey;
    use winreg::enums::HKEY_LOCAL_MACHINE;

    let hklm = RegKey::predef(HKEY_LOCAL_MACHINE);
    for key_path in &[r"SOFTWARE\WOW6432Node\Valve\Steam", r"SOFTWARE\Valve\Steam"] {
        if let Ok(steam_key) = hklm.open_subkey(key_path) {
            if let Ok(install_path) = steam_key.get_value::<String, _>("InstallPath") {
                return Ok(PathBuf::from(install_path));
            }
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "Steam path not found in registry",
    ))
}
