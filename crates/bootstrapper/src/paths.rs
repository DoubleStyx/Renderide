//! Path discovery for Resonite installation, dotnet, and PID file.
//! Searches RESONITE_DIR, STEAM_PATH, and Steam libraryfolders.vdf.

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

const PID_FILE_NAME: &str = "renderide_bootstrap.pid";

/// Finds the dotnet executable to run Renderite.Host.dll on Linux.
/// Prefers bundled dotnet-runtime/dotnet in the Resonite folder.
pub fn find_dotnet_for_host(resonite_dir: &Path) -> PathBuf {
    let bundled = resonite_dir.join("dotnet-runtime").join("dotnet");
    if bundled.exists() {
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
    for line in BufReader::new(file).lines().filter_map(Result::ok) {
        if let Some(idx) = line.find("\"path\"") {
            let rest = line[idx + 6..].trim_start_matches(|c: char| c == '\t' || c == ' ');
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
/// Checks: RESONITE_DIR env, STEAM_PATH env, home-based defaults, libraryfolders.vdf.
pub fn find_resonite_dir() -> Option<PathBuf> {
    let host_exe = |dir: &Path| dir.join(RENDERITE_HOST_EXE).exists();

    if let Ok(dir) = env::var("RESONITE_DIR") {
        let path = PathBuf::from(&dir);
        if host_exe(&path) {
            return Some(path);
        }
    }

    let home = env::var("HOME").ok()?;
    let home = Path::new(&home);

    if let Ok(steam) = env::var("STEAM_PATH") {
        let path = PathBuf::from(steam)
            .join("steamapps")
            .join("common")
            .join(RESONITE_APP_NAME);
        if host_exe(&path) {
            return Some(path);
        }
    }

    let steam_bases = [
        home.join(".local").join("share").join("Steam"),
        home.join(".steam").join("steam"),
    ];

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

/// PID file path for orphan cleanup. Stored in temp dir so it persists across runs.
pub fn pid_file_path() -> PathBuf {
    env::temp_dir().join(PID_FILE_NAME)
}
