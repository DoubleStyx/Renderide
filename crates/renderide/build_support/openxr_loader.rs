//! Vendored OpenXR loader copying for build artifacts.
//!
//! For `windows` targets, copies one `openxr_loader.dll` from
//! `../../third_party/openxr_loader/openxr_loader_windows-*/` matching `CARGO_CFG_TARGET_ARCH`.
//! For `macos` targets, copies one `libopenxr_loader.dylib` from
//! `../../third_party/openxr_loader/openxr_loader_macos-*/`.
//!
//! The destination is derived from `OUT_DIR`, not `PROFILE`, so custom profiles that inherit from
//! `dev` place the loader next to the binary under `target/<profile-dir>/`.
//! Unsupported targets skip this (Linux uses the system loader at run time).

use std::fs;
use std::path::{Path, PathBuf};

use super::artifacts::artifact_dir_from_out_dir;

/// Khronos `openxr_loader_windows-*` subfolder names for each Rust target arch.
mod openxr_win {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/build_support/openxr_windows_arch.rs"
    ));
}

/// Returns the vendored loader filename for the requested target OS.
pub(crate) fn openxr_loader_library_filename(target_os: &str) -> Option<&'static str> {
    match target_os {
        "windows" => Some("openxr_loader.dll"),
        "macos" => Some("libopenxr_loader.dylib"),
        _ => None,
    }
}

/// Returns the vendored loader package directory prefix for the requested target OS.
fn openxr_loader_package_prefix(target_os: &str) -> Option<&'static str> {
    match target_os {
        "windows" => Some("openxr_loader_windows-"),
        "macos" => Some("openxr_loader_macos-"),
        _ => None,
    }
}

/// Picks the lexicographically last matching package directory so newer SDK versions win.
fn find_latest_openxr_package_dir(
    third_party_openxr: &Path,
    package_prefix: &str,
) -> Option<PathBuf> {
    let rd = fs::read_dir(third_party_openxr).ok()?;
    let mut candidates: Vec<PathBuf> = rd
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.is_dir()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with(package_prefix))
        })
        .collect();
    candidates.sort();
    candidates.into_iter().next_back()
}

/// Resolves the source path for a vendored OpenXR loader package.
pub(crate) fn vendored_openxr_loader_source(
    third_party_openxr: &Path,
    target_os: &str,
    arch: Option<&str>,
) -> Option<PathBuf> {
    let package_prefix = openxr_loader_package_prefix(target_os)?;
    let library_filename = openxr_loader_library_filename(target_os)?;
    let pkg_root = find_latest_openxr_package_dir(third_party_openxr, package_prefix)?;

    if target_os == "windows" {
        let subdir = openxr_win::khronos_windows_subdir_for_arch(arch?)?;
        Some(pkg_root.join(subdir).join(library_filename))
    } else {
        Some(pkg_root.join(library_filename))
    }
}

/// Resolves the destination path next to the build artifact for the requested target OS.
pub(crate) fn openxr_loader_destination_path(out_dir: &Path, target_os: &str) -> Option<PathBuf> {
    let library_filename = openxr_loader_library_filename(target_os)?;
    Some(artifact_dir_from_out_dir(out_dir)?.join(library_filename))
}

/// Copies the Khronos `OpenXR` loader next to the build output for supported vendored targets.
pub fn copy_vendored_openxr_loader(manifest_dir: &Path, out_dir: &Path) {
    let Ok(target_os) = std::env::var("CARGO_CFG_TARGET_OS") else {
        return;
    };
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").ok();
    copy_vendored_openxr_loader_for_target(manifest_dir, out_dir, &target_os, arch.as_deref());
}

/// Copies the vendored loader for an explicit target OS and architecture.
///
/// Returns `true` when a loader was copied. Unsupported targets and missing package files return
/// `false` after emitting Cargo warnings where appropriate.
pub(crate) fn copy_vendored_openxr_loader_for_target(
    manifest_dir: &Path,
    out_dir: &Path,
    target_os: &str,
    arch: Option<&str>,
) -> bool {
    if openxr_loader_library_filename(target_os).is_none() {
        return false;
    }
    let workspace_dir = manifest_dir.join("../..");
    let third_party = workspace_dir.join("third_party/openxr_loader");
    println!("cargo:rerun-if-changed={}", third_party.display());

    if target_os == "windows" && arch.is_none() {
        println!("cargo:warning=openxr_loader: CARGO_CFG_TARGET_ARCH unset");
        return false;
    }

    let Some(src) = vendored_openxr_loader_source(&third_party, target_os, arch) else {
        let package_prefix = openxr_loader_package_prefix(target_os).unwrap_or("openxr_loader_");
        println!(
            "cargo:warning=openxr_loader: no usable {package_prefix}* package under {}",
            third_party.display()
        );
        return false;
    };
    println!("cargo:rerun-if-changed={}", src.display());

    if !src.exists() {
        println!(
            "cargo:warning=openxr_loader: missing vendored loader at {}",
            src.display()
        );
        return false;
    }

    let Some(dest) = openxr_loader_destination_path(out_dir, target_os) else {
        println!("cargo:warning=openxr_loader: cannot derive artifact dir from OUT_DIR");
        return false;
    };
    let Some(dest_dir) = dest.parent() else {
        println!(
            "cargo:warning=openxr_loader: destination {} has no parent",
            dest.display()
        );
        return false;
    };
    if let Err(e) = fs::create_dir_all(dest_dir) {
        println!(
            "cargo:warning=openxr_loader: mkdir {} failed: {e}",
            dest_dir.display()
        );
        return false;
    }
    if let Err(e) = fs::copy(&src, &dest) {
        println!(
            "cargo:warning=openxr_loader: copy {} -> {} failed: {e}",
            src.display(),
            dest.display()
        );
        return false;
    }
    true
}
