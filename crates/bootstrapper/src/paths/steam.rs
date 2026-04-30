//! Steam introspection: `libraryfolders.vdf` parsing, platform default roots, and (on Windows)
//! registry lookup of `InstallPath`.

use std::env;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// Returns likely Steam installation roots for the current platform (env vars and registry on Windows).
pub fn base_paths() -> Vec<PathBuf> {
    #[cfg(windows)]
    {
        let mut bases = Vec::new();
        if let Ok(steam) = env::var("STEAM_PATH") {
            bases.push(PathBuf::from(steam));
        }
        if let Ok(path) = path_from_registry()
            && !bases.iter().any(|b| b == &path)
        {
            bases.push(path);
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

/// Extracts `"path"` values from Steam's `libraryfolders.vdf` under `steam_base`.
pub fn library_paths_from_vdf(steam_base: &Path) -> Vec<PathBuf> {
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

/// Reads the Steam install path from `HKLM\...\Valve\Steam` when `InstallPath` is present.
#[cfg(windows)]
fn path_from_registry() -> Result<PathBuf, std::io::Error> {
    use winreg::RegKey;
    use winreg::enums::HKEY_LOCAL_MACHINE;

    let hklm = RegKey::predef(HKEY_LOCAL_MACHINE);
    for key_path in &[r"SOFTWARE\WOW6432Node\Valve\Steam", r"SOFTWARE\Valve\Steam"] {
        if let Ok(steam_key) = hklm.open_subkey(key_path)
            && let Ok(install_path) = steam_key.get_value::<String, &str>("InstallPath")
        {
            return Ok(PathBuf::from(install_path));
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

    #[test]
    fn library_paths_from_vdf_extracts_quoted_paths() {
        let tmp = env::temp_dir().join(format!(
            "bootstrapper_libraryfolders_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(tmp.join("steamapps")).unwrap();
        let vdf = tmp.join("steamapps").join("libraryfolders.vdf");
        let mut f = fs::File::create(&vdf).unwrap();
        writeln!(f, r#" "path" "/data/SteamLibrary" "#,).unwrap();
        let paths = library_paths_from_vdf(&tmp);
        assert!(
            paths.iter().any(|p| p == Path::new("/data/SteamLibrary")),
            "paths={paths:?}"
        );
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn library_paths_from_vdf_multiple_paths_and_garbage() {
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
        let paths = library_paths_from_vdf(&tmp);
        assert!(paths.iter().any(|p| p == Path::new("/first/lib")));
        assert!(
            paths
                .iter()
                .any(|p| p == Path::new("/Volumes/My Disk/SteamLibrary"))
        );
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn library_paths_from_vdf_returns_empty_for_file_without_path_keys() {
        let tmp = env::temp_dir().join(format!(
            "bootstrapper_libraryfolders_nopath_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(tmp.join("steamapps")).unwrap();
        let vdf = tmp.join("steamapps").join("libraryfolders.vdf");
        let mut f = fs::File::create(&vdf).unwrap();
        writeln!(f, r#""label" "main""#).unwrap();
        writeln!(f, r#""contentid" "12345""#).unwrap();
        let paths = library_paths_from_vdf(&tmp);
        assert!(
            paths.is_empty(),
            "expected no extracted paths, got {paths:?}"
        );
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn library_paths_from_vdf_returns_empty_when_file_missing() {
        let tmp = env::temp_dir().join(format!(
            "bootstrapper_libraryfolders_missing_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        let paths = library_paths_from_vdf(&tmp);
        assert!(paths.is_empty());
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn library_paths_from_vdf_extracts_unicode_path() {
        let tmp = env::temp_dir().join(format!(
            "bootstrapper_libraryfolders_unicode_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(tmp.join("steamapps")).unwrap();
        let vdf = tmp.join("steamapps").join("libraryfolders.vdf");
        let mut f = fs::File::create(&vdf).unwrap();
        writeln!(f, r#" "path" "/games/Steam ライブラリ" "#).unwrap();
        let paths = library_paths_from_vdf(&tmp);
        assert!(
            paths
                .iter()
                .any(|p| p == Path::new("/games/Steam ライブラリ")),
            "expected unicode path preserved, got {paths:?}"
        );
        let _ = fs::remove_dir_all(&tmp);
    }
}
