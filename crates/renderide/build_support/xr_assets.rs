//! XR action manifest and binding-table copying for build artifacts.

use std::fs;
use std::path::{Path, PathBuf};

/// Derives the Cargo artifact profile directory from `OUT_DIR`.
///
/// Cargo always sets `OUT_DIR = .../target/<profile-dir>/build/<pkg>-<hash>/out`; walking up
/// three components recovers `.../target/<profile-dir>/` even when `PROFILE` is `debug` for a
/// custom profile that inherits from `dev` (like this workspace's `dev-fast`).
fn artifact_dir_from_out_dir(out_dir: &Path) -> Option<PathBuf> {
    out_dir.ancestors().nth(3).map(std::path::Path::to_path_buf)
}

/// Copies the XR action manifest and per-profile binding tables into the artifact directory so the
/// runtime can load them alongside the binary (same convention as `config.toml`).
///
/// Source files live at `crates/renderide/assets/xr/` and are mirrored to
/// `target/<profile-dir>/xr/` with `actions.toml` at the root and `bindings/*.toml` below.
/// `cargo:rerun-if-changed` is emitted for the source directory so TOML edits trigger a rebuild
/// copy.
pub(crate) fn copy_xr_assets_to_artifact_dir(manifest_dir: &Path, out_dir: &Path) {
    let src_root = manifest_dir.join("assets/xr");
    println!("cargo:rerun-if-changed={}", src_root.display());
    if !src_root.is_dir() {
        return;
    }

    let Some(dest_root_parent) = artifact_dir_from_out_dir(out_dir) else {
        println!("cargo:warning=xr_assets: cannot derive artifact dir from OUT_DIR");
        return;
    };
    let dest_root = dest_root_parent.join("xr");
    let dest_bindings = dest_root.join("bindings");
    if let Err(e) = fs::create_dir_all(&dest_bindings) {
        println!(
            "cargo:warning=xr_assets: mkdir {} failed: {e}",
            dest_bindings.display()
        );
        return;
    }

    let src_actions = src_root.join("actions.toml");
    let dest_actions = dest_root.join("actions.toml");
    if let Err(e) = fs::copy(&src_actions, &dest_actions) {
        println!(
            "cargo:warning=xr_assets: copy {} -> {} failed: {e}",
            src_actions.display(),
            dest_actions.display()
        );
    }

    let src_bindings = src_root.join("bindings");
    let Ok(entries) = fs::read_dir(&src_bindings) else {
        println!(
            "cargo:warning=xr_assets: read_dir {} failed",
            src_bindings.display()
        );
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("toml") {
            continue;
        }
        let Some(file_name) = path.file_name() else {
            continue;
        };
        let dest = dest_bindings.join(file_name);
        if let Err(e) = fs::copy(&path, &dest) {
            println!(
                "cargo:warning=xr_assets: copy {} -> {} failed: {e}",
                path.display(),
                dest.display()
            );
        }
    }
}
