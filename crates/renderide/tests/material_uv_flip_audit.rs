//! Audit: every WGSL material that samples a host-uploaded `@group(1)` 2D texture must rely on
//! the unified Unity-orientation convention -- sampled storage is V=0 bottom (Unity), mesh UVs
//! are also V=0 bottom, so material shaders apply no V flip and use the plain `apply_st` helper.
//!
//! This guard exists to prevent the previous per-binding `_<Tex>_StorageVInverted` flag (and the
//! `apply_st_for_storage` / `flip_v_for_storage` helpers) from creeping back in. Cubemap orientation
//! is a separate concern: `cubemap_storage_dir` and `_<Cube>_StorageVInverted` for cubemap bindings
//! remain in `projection360.wgsl` and `skybox_projection360.wgsl` and are not flagged.

use std::path::{Path, PathBuf};

const FORBIDDEN_2D_HELPERS: &[&str] = &[
    "apply_st_for_storage(",
    "flip_v_for_storage(",
    "uvu::flip_v(",
];

fn materials_dir() -> PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest).join("shaders/materials")
}

fn modules_dir() -> PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest).join("shaders/modules")
}

fn wgsl_files_in(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("wgsl") {
            files.push(path);
        }
    }
    Ok(files)
}

fn declared_storage_inverted_fields(src: &str) -> Vec<String> {
    let mut names = Vec::new();
    for line in src.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("//") || !trimmed.contains("_StorageVInverted") {
            continue;
        }
        let Some((name, ty)) = trimmed.split_once(':') else {
            continue;
        };
        if !ty.trim_start().starts_with("f32") {
            continue;
        }
        let name = name.trim().trim_end_matches(',').to_owned();
        names.push(name);
    }
    names.sort();
    names.dedup();
    names
}

#[test]
fn material_2d_textures_have_no_storage_inverted_field() -> Result<(), Box<dyn std::error::Error>> {
    let mut offenders: Vec<String> = Vec::new();
    for dir in [materials_dir(), modules_dir()] {
        for path in wgsl_files_in(&dir)? {
            let src = std::fs::read_to_string(&path)?;
            for field in declared_storage_inverted_fields(&src) {
                if !field.contains("Cube") {
                    offenders.push(format!(
                        "{} declares Texture2D field {field}",
                        path.file_name().unwrap().to_string_lossy()
                    ));
                }
            }
        }
    }
    if !offenders.is_empty() {
        offenders.sort();
        panic!(
            "Texture2D bindings must not carry _StorageVInverted; sampled textures use Unity (V=0 bottom) convention:\n  - {}",
            offenders.join("\n  - ")
        );
    }
    Ok(())
}

#[test]
fn no_storage_aware_uv_helpers_remain() -> Result<(), Box<dyn std::error::Error>> {
    let mut offenders: Vec<String> = Vec::new();
    for dir in [materials_dir(), modules_dir()] {
        for path in wgsl_files_in(&dir)? {
            let src = std::fs::read_to_string(&path)?;
            for helper in FORBIDDEN_2D_HELPERS {
                if src.contains(helper) {
                    offenders.push(format!(
                        "{} contains forbidden helper {helper}",
                        path.file_name().unwrap().to_string_lossy()
                    ));
                }
            }
        }
    }
    if !offenders.is_empty() {
        offenders.sort();
        panic!(
            "storage-aware UV helpers must be replaced with plain `apply_st`:\n  - {}",
            offenders.join("\n  - ")
        );
    }
    Ok(())
}
