//! Source audits for clustered-forward shader list lookup invariants.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Returns the renderide crate directory.
fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Returns all WGSL files directly under `relative_dir`.
fn wgsl_files(relative_dir: &str) -> io::Result<Vec<PathBuf>> {
    let dir = manifest_dir().join(relative_dir);
    fs::read_dir(dir)?
        .filter_map(|entry| match entry {
            Ok(entry) => {
                let path = entry.path();
                path.extension()
                    .is_some_and(|ext| ext == "wgsl")
                    .then_some(Ok(path))
            }
            Err(err) => Some(Err(err)),
        })
        .collect()
}

/// Returns true when `path` is allowed to read raw clustered-light storage.
fn raw_cluster_storage_reader_allowed(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| matches!(name, "pbs_cluster.wgsl" | "globals.wgsl"))
}

/// Returns true when a shader source reads raw clustered-light list storage.
fn contains_raw_cluster_storage_read(src: &str) -> bool {
    [
        "rg::cluster_light_counts[",
        "cluster_light_counts[",
        "rg::cluster_light_indices[",
        "cluster_light_indices[",
    ]
    .iter()
    .any(|needle| src.contains(needle))
}

/// Materials and lighting modules must read clustered lists through `pcls` helpers.
#[test]
fn clustered_light_storage_uses_shared_helpers() -> io::Result<()> {
    let mut offenders = Vec::new();
    for dir in ["shaders/source/materials", "shaders/source/modules"] {
        for path in wgsl_files(dir)? {
            if raw_cluster_storage_reader_allowed(&path) {
                continue;
            }
            let src = fs::read_to_string(&path)?;
            if contains_raw_cluster_storage_read(&src) {
                offenders.push(path);
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "clustered-light list storage must be read through pcls helper functions:\n  {}",
        offenders
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join("\n  ")
    );
    Ok(())
}
