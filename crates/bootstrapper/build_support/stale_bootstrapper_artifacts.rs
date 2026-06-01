//! Cleanup for stale pre-rename launcher artifacts in Cargo output directories.

use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

const STALE_BOOTSTRAPPER_ARTIFACTS: [&str; 2] = ["bootstrapper", "bootstrapper.exe"];

/// Derives the Cargo artifact profile directory from `OUT_DIR`.
///
/// Cargo sets `OUT_DIR` to `target/<profile>/build/<pkg>-<hash>/out`; walking up three
/// components recovers `target/<profile>`. Cross-target builds already include the target triple
/// in `OUT_DIR`, so the same derivation yields `target/<triple>/<profile>`.
pub(crate) fn artifact_dir_from_out_dir(out_dir: &Path) -> Option<PathBuf> {
    out_dir.ancestors().nth(3).map(Path::to_path_buf)
}

/// Removes stale top-level `bootstrapper` launcher artifacts for the current Cargo output dir.
pub(crate) fn remove_stale_bootstrapper_artifacts(out_dir: &Path) {
    let Some(artifact_dir) = artifact_dir_from_out_dir(out_dir) else {
        cargo_warning(format_args!(
            "bootstrapper cleanup: cannot derive artifact dir from OUT_DIR {}",
            out_dir.display()
        ));
        return;
    };

    for artifact_name in STALE_BOOTSTRAPPER_ARTIFACTS {
        remove_stale_artifact(&artifact_dir.join(artifact_name));
    }
}

fn remove_stale_artifact(path: &Path) {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            let file_type = metadata.file_type();
            if file_type.is_dir() {
                cargo_warning(format_args!(
                    "bootstrapper cleanup: leaving directory {}",
                    path.display()
                ));
                return;
            }
        }
        Err(err) if err.kind() == ErrorKind::NotFound => return,
        Err(err) => {
            cargo_warning(format_args!(
                "bootstrapper cleanup: cannot inspect stale artifact {}: {}",
                path.display(),
                err
            ));
            return;
        }
    }

    match fs::remove_file(path) {
        Ok(()) => {}
        Err(err) if err.kind() == ErrorKind::NotFound => {}
        Err(err) => {
            cargo_warning(format_args!(
                "bootstrapper cleanup: failed to remove stale artifact {}: {}",
                path.display(),
                err
            ));
        }
    }
}

fn cargo_warning(args: std::fmt::Arguments<'_>) {
    println!("cargo:warning={args}");
}
