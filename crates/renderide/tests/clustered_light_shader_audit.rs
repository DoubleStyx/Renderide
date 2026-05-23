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
        .is_some_and(|name| matches!(name, "cluster.wgsl" | "globals.wgsl"))
}

/// Returns true when a shader source reads raw clustered-light list storage.
fn contains_raw_cluster_storage_read(src: &str) -> bool {
    [
        "rg::cluster_light_counts[",
        "cluster_light_counts[",
        "rg::cluster_light_ranges[",
        "cluster_light_ranges[",
        "rg::cluster_light_indices[",
        "cluster_light_indices[",
    ]
    .iter()
    .any(|needle| src.contains(needle))
}

/// Returns the clustered-light compute shader source.
fn clustered_light_compute_src() -> io::Result<String> {
    fs::read_to_string(manifest_dir().join("shaders/passes/compute/clustered_light.wgsl"))
}

/// Materials and lighting modules must read clustered lists through `pcls` helpers.
#[test]
fn clustered_light_storage_uses_shared_helpers() -> io::Result<()> {
    let mut offenders = Vec::new();
    for dir in ["shaders/materials", "shaders/modules"] {
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

/// Spotlight assignment must use froxel-sphere cone culling after the range broad phase.
#[test]
fn clustered_light_spotlights_use_conservative_cone_sphere_culling() -> io::Result<()> {
    let src = clustered_light_compute_src()?;
    for needle in [
        "fn spotlight_cone_intersects_sphere",
        "sphere_aabb_intersect(apex, range, aabb_min, aabb_max)",
        "aabb_bounding_sphere_radius",
        "SPOT_CULL_WIDE_COS_HALF",
        "SPOT_CULL_MIN_COS_HALF",
    ] {
        assert!(
            src.contains(needle),
            "clustered-light compute shader is missing spotlight culling invariant `{needle}`"
        );
    }
    Ok(())
}
