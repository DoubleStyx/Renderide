//! Copying assets such as skybox meshes for build artifacts.

use std::fs;
use std::path::Path;

use super::artifacts::artifact_dir_from_out_dir;

/// Copies the assets into the artifact directory so the
/// runtime can load them alongside the binary (same convention as `config.toml`).
///
/// Source files live at `crates/renderide/assets/models/` and are mirrored to `target/<profile-dir>/models/`.
/// `cargo:rerun-if-changed` is emitted for the source directory so new assets trigger a rebuild copy.
pub fn copy_assets_to_artifact_dir(manifest_dir: &Path, out_dir: &Path) {
    let src_root = manifest_dir.join("assets");
    println!("cargo:rerun-if-changed={}", src_root.display());
    if !src_root.is_dir() {
        return;
    }

    let Some(dest_root_parent) = artifact_dir_from_out_dir(out_dir) else {
        println!("cargo:warning=assets: cannot derive artifact dir from OUT_DIR");
        return;
    };
    let dest_root = dest_root_parent.join("assets");
    if let Err(e) = fs::create_dir_all(&dest_root) {
        println!(
            "cargo:warning=assets: mkdir {} failed: {e}",
            dest_root.display()
        );
        return;
    }

    let Ok(entries) = fs::read_dir(&src_root) else {
        println!(
            "cargo:warning=assets: read_dir {} failed",
            src_root.display()
        );
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(file_name) = path.file_name() else {
            continue;
        };
        if path.is_dir() && file_name.to_str().is_some_and(is_copied_asset_dir) {
            copy_asset_dir_to_artifact_dir(path.as_path(), &out_dir.join(file_name));
        }
    }
}

fn copy_asset_dir_to_artifact_dir(src_dir: &Path, out_dir: &Path) {
    let Ok(entries) = fs::read_dir(src_dir) else {
        println!(
            "cargo:warning=assets: read_dir {} failed",
            src_dir.display()
        );
        return;
    };
    if let Err(e) = fs::create_dir_all(out_dir) {
        println!(
            "cargo:warning=assets: mkdir {} failed: {e}",
            out_dir.display()
        );
        return;
    }
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(file_name) = path.file_name() else {
            continue;
        };
        if path.is_dir() {
            copy_asset_dir_to_artifact_dir(path.as_path(), &out_dir.join(file_name));
            continue;
        }
        if !path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(is_supported_extension)
        {
            continue;
        }
        let dest = out_dir.join(file_name);
        if let Err(e) = fs::copy(&path, &dest) {
            println!(
                "cargo:warning=assets: copy {} -> {} failed: {e}",
                path.display(),
                dest.display()
            );
        }
    }
}

const ASSET_DIRS: [&str; 1] = ["models"];

const SUPPORTED_EXTENSIONS: [&str; 2] = ["glb", "gltf"];

fn is_copied_asset_dir(dir: &str) -> bool {
    ASSET_DIRS
        .iter()
        .any(|supported| supported.eq_ignore_ascii_case(dir))
}

fn is_supported_extension(ext: &str) -> bool {
    SUPPORTED_EXTENSIONS
        .iter()
        .any(|supported| supported.eq_ignore_ascii_case(ext))
}
