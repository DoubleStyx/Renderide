//! Vendored OpenXR loader copying for Windows build artifacts.
//!
//! For `windows` targets, copies **one** `openxr_loader.dll` from
//! `../../third_party/openxr_loader/openxr_loader_windows-*/` matching `CARGO_CFG_TARGET_ARCH`
//! into the same artifact directory Cargo uses for this build: `target/<PROFILE>/` when `TARGET`
//! equals `HOST`, and `target/<TARGET>/<PROFILE>/` when cross-compiling (`--target`).
//! Non-Windows targets skip this (Linux uses the system loader at run time).

use std::fs;
use std::path::{Path, PathBuf};

/// Khronos `openxr_loader_windows-*` subfolder names for each Rust target arch.
mod openxr_win {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/openxr_windows_arch.rs"
    ));
}

/// Picks the lexicographically last `openxr_loader_windows-*` directory so newer SDK versions win.
fn find_latest_openxr_windows_package_dir(third_party_openxr: &Path) -> Option<PathBuf> {
    let rd = fs::read_dir(third_party_openxr).ok()?;
    let mut candidates: Vec<PathBuf> = rd
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.is_dir()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("openxr_loader_windows-"))
        })
        .collect();
    candidates.sort();
    candidates.into_iter().next_back()
}

/// Directory where Cargo places binaries for the current package build (`renderide.exe`, etc.).
///
/// Cargo uses `target/<PROFILE>/` for the default host target and `target/<TARGET>/<PROFILE>/` when
/// `--target` selects a different triple; copying the loader only to `target/<PROFILE>/` misses
/// cross-compiled outputs (e.g. `x86_64-pc-windows-gnu`).
fn cargo_artifact_profile_dir(cargo_target_dir: &Path, profile: &str) -> Option<PathBuf> {
    let target = std::env::var("TARGET").ok()?;
    let host = std::env::var("HOST").ok()?;
    if target == host {
        Some(cargo_target_dir.join(profile))
    } else {
        Some(cargo_target_dir.join(target).join(profile))
    }
}

/// Copies the Khronos `OpenXR` loader DLL next to the build output for Windows targets only.
pub(crate) fn copy_vendored_openxr_loader_windows(manifest_dir: &Path) {
    let Ok(target_os) = std::env::var("CARGO_CFG_TARGET_OS") else {
        return;
    };
    if target_os != "windows" {
        return;
    }

    let Ok(arch) = std::env::var("CARGO_CFG_TARGET_ARCH") else {
        println!("cargo:warning=openxr_loader: CARGO_CFG_TARGET_ARCH unset");
        return;
    };

    let Some(subdir) = openxr_win::khronos_windows_subdir_for_arch(&arch) else {
        println!("cargo:warning=openxr_loader: no vendored Khronos folder for target arch {arch}");
        return;
    };

    let workspace_dir = manifest_dir.join("../..");
    let third_party = workspace_dir.join("third_party/openxr_loader");
    println!("cargo:rerun-if-changed={}", third_party.display());

    let Some(pkg_root) = find_latest_openxr_windows_package_dir(&third_party) else {
        println!(
            "cargo:warning=openxr_loader: no openxr_loader_windows-* under {}",
            third_party.display()
        );
        return;
    };

    let src = pkg_root.join(subdir).join("openxr_loader.dll");
    println!("cargo:rerun-if-changed={}", src.display());

    if !src.exists() {
        println!(
            "cargo:warning=openxr_loader: missing vendored DLL at {}",
            src.display()
        );
        return;
    }

    let cargo_target_dir = std::env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| manifest_dir.join("../../target"));
    let profile = std::env::var("PROFILE").unwrap_or_else(|_| "debug".into());

    let Some(dest_dir) = cargo_artifact_profile_dir(&cargo_target_dir, &profile) else {
        println!("cargo:warning=openxr_loader: TARGET/HOST unset");
        return;
    };
    if let Err(e) = fs::create_dir_all(&dest_dir) {
        println!(
            "cargo:warning=openxr_loader: mkdir {} failed: {e}",
            dest_dir.display()
        );
        return;
    }
    let dest = dest_dir.join("openxr_loader.dll");
    if let Err(e) = fs::copy(&src, &dest) {
        println!(
            "cargo:warning=openxr_loader: copy {} -> {} failed: {e}",
            src.display(),
            dest.display()
        );
    }
}
