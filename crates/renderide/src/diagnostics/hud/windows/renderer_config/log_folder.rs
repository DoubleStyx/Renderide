//! Platform-specific log-folder opener used by the renderer-config HUD.

use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};

#[derive(Debug, thiserror::Error)]
pub(in crate::diagnostics::hud::windows::renderer_config) enum OpenLogFolderError {
    /// Spawning the platform opener failed.
    #[error("failed to spawn {program} for {path}: {source}")]
    Spawn {
        /// Program selected for the current platform.
        program: &'static str,
        /// Log-folder path passed to the opener.
        path: PathBuf,
        /// Underlying process-spawn error.
        #[source]
        source: io::Error,
    },
    /// The platform opener exited unsuccessfully.
    #[error("{program} failed for {path} with status {status}")]
    ExitStatus {
        /// Program selected for the current platform.
        program: &'static str,
        /// Log-folder path passed to the opener.
        path: PathBuf,
        /// Non-success process exit status.
        status: ExitStatus,
    },
}

/// Returns the platform program used to reveal a folder in the system file manager.
pub(in crate::diagnostics::hud::windows::renderer_config) fn log_folder_opener_program()
-> &'static str {
    if cfg!(target_os = "windows") {
        "explorer"
    } else if cfg!(target_os = "macos") {
        "open"
    } else {
        "xdg-open"
    }
}

/// Opens `path` in the platform file manager.
pub(in crate::diagnostics::hud::windows::renderer_config) fn open_log_folder(
    path: &Path,
) -> Result<(), OpenLogFolderError> {
    let program = log_folder_opener_program();
    let status =
        Command::new(program)
            .arg(path)
            .status()
            .map_err(|source| OpenLogFolderError::Spawn {
                program,
                path: path.to_path_buf(),
                source,
            })?;
    if status.success() {
        Ok(())
    } else {
        Err(OpenLogFolderError::ExitStatus {
            program,
            path: path.to_path_buf(),
            status,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::log_folder_opener_program;

    #[test]
    fn log_folder_opener_program_matches_platform() {
        let expected = if cfg!(target_os = "windows") {
            "explorer"
        } else if cfg!(target_os = "macos") {
            "open"
        } else {
            "xdg-open"
        };

        assert_eq!(log_folder_opener_program(), expected);
    }
}
