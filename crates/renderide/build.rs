//! Build-script entry point for shader composition and runtime asset copying.

use std::path::PathBuf;

mod build_support;

use build_support::openxr_loader::copy_vendored_openxr_loader_windows;
use build_support::shader::{self, BuildError};
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
        }
        std::process::exit(1);
    }
}

/// Coordinates build-time asset copying and shader composition.
fn run() -> Result<(), BuildError> {
    let manifest_dir = PathBuf::from(shader::env_var("CARGO_MANIFEST_DIR")?);
    let out_dir = PathBuf::from(shader::env_var("OUT_DIR")?);

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=build_support");

    copy_vendored_openxr_loader_windows(&manifest_dir);
    copy_xr_assets_to_artifact_dir(&manifest_dir, &out_dir);
    shader::compile(&manifest_dir, &out_dir)
}
