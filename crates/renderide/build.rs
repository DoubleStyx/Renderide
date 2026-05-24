//! Build-script entry point for shader composition and generated runtime data.

use std::path::PathBuf;

mod build_support;

use build_support::git;
use build_support::openxr_loader::copy_vendored_openxr_loader;
use build_support::shader::{self, BuildError};
use build_support::skybox_mesh::emit_procedural_skybox_mesh;
use build_support::xr_assets::copy_xr_assets_to_artifact_dir;

/// Runs the build script.
fn main() {
    if let Err(e) = run() {
        #[expect(
            clippy::print_stderr,
            reason = "build script: errors route to cargo stderr"
        )]
        {
            eprintln!("renderide build.rs: {e:#}");
        };
        std::process::exit(1);
    }
}

/// Coordinates build-time asset copying and shader composition.
fn run() -> Result<(), BuildError> {
    let manifest_dir = PathBuf::from(shader::env_var("CARGO_MANIFEST_DIR")?);
    let out_dir = PathBuf::from(shader::env_var("OUT_DIR")?);

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=build_support");

    git::emit_rerun_if_changed(&manifest_dir);
    let commit = git::current_commit_short(&manifest_dir).unwrap_or_default();
    println!("cargo:rustc-env=RENDERIDE_GIT_COMMIT={commit}");

    copy_vendored_openxr_loader(&manifest_dir, &out_dir);
    copy_xr_assets_to_artifact_dir(&manifest_dir, &out_dir);
    emit_procedural_skybox_mesh(&manifest_dir, &out_dir)?;
    shader::compile(&manifest_dir, &out_dir)
}
