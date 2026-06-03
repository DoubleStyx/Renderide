//! Exercises the bootstrapper build-script cleanup helper under `cargo test`.

#![allow(
    clippy::print_stdout,
    reason = "the included build-script helper emits Cargo warnings through println!"
)]

use std::fs;
use std::path::{Path, PathBuf};

#[path = "../build_support/stale_bootstrapper_artifacts.rs"]
mod stale_bootstrapper_artifacts;

type TestResult = Result<(), Box<dyn std::error::Error>>;

#[test]
fn artifact_dir_from_out_dir_handles_profiles_and_cross_targets() {
    for (out_dir, expected) in [
        ("target/debug/build/bootstrapper-abc/out", "target/debug"),
        (
            "target/dev-fast/build/bootstrapper-abc/out",
            "target/dev-fast",
        ),
        (
            "target/x86_64-pc-windows-msvc/release/build/bootstrapper-abc/out",
            "target/x86_64-pc-windows-msvc/release",
        ),
    ] {
        assert_eq!(
            stale_bootstrapper_artifacts::artifact_dir_from_out_dir(Path::new(out_dir)),
            Some(PathBuf::from(expected))
        );
    }
}

#[test]
fn cleanup_removes_exact_top_level_stale_binaries_only() -> TestResult {
    let temp = tempfile::tempdir()?;
    let artifact_dir = temp.path().join("target/dev-fast");
    let out_dir = artifact_dir.join("build/bootstrapper-abc/out");
    fs::create_dir_all(&out_dir)?;
    fs::create_dir_all(artifact_dir.join("deps"))?;

    fs::write(artifact_dir.join("bootstrapper"), b"old unix launcher")?;
    fs::write(
        artifact_dir.join("bootstrapper.exe"),
        b"old windows launcher",
    )?;
    fs::write(artifact_dir.join("bootstrapper.d"), b"depinfo")?;
    fs::write(artifact_dir.join("renderide"), b"current launcher")?;
    fs::write(
        artifact_dir.join("deps/bootstrapper-hash"),
        b"internal cargo artifact",
    )?;

    stale_bootstrapper_artifacts::remove_stale_bootstrapper_artifacts(&out_dir);

    assert!(!artifact_dir.join("bootstrapper").exists());
    assert!(!artifact_dir.join("bootstrapper.exe").exists());
    assert!(artifact_dir.join("bootstrapper.d").is_file());
    assert!(artifact_dir.join("renderide").is_file());
    assert!(artifact_dir.join("deps/bootstrapper-hash").is_file());

    Ok(())
}

#[test]
fn cleanup_leaves_directories_named_like_stale_artifacts() -> TestResult {
    let temp = tempfile::tempdir()?;
    let artifact_dir = temp.path().join("target/debug");
    let out_dir = artifact_dir.join("build/bootstrapper-abc/out");
    fs::create_dir_all(&out_dir)?;
    fs::create_dir_all(artifact_dir.join("bootstrapper"))?;

    stale_bootstrapper_artifacts::remove_stale_bootstrapper_artifacts(&out_dir);

    assert!(artifact_dir.join("bootstrapper").is_dir());

    Ok(())
}
