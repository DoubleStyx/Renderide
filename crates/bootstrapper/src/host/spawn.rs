//! Host command construction, spawning, and child lifetime registration.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

use crate::child_lifetime::ChildLifetimeGroup;
use crate::config::ResoBootConfig;
use crate::paths;

use super::runtime_config::{
    prepare_native_host_runtime_config, strip_windows_desktop_from_runtime_config,
};

/// Configures stdio pipes and working directory for a Host launch.
fn apply_host_stdio(cmd: &mut Command, cwd: &Path) {
    cmd.current_dir(cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
}

/// Prepares the command, spawns, and registers the child with `lifetime`.
fn finish_spawn(mut cmd: Command, lifetime: &ChildLifetimeGroup) -> std::io::Result<Child> {
    logger::info!(
        "Host spawn command: program={:?} cwd={:?}",
        cmd.get_program(),
        cmd.get_current_dir()
    );
    lifetime.prepare_command(&mut cmd);
    let child = cmd.spawn()?;
    lifetime.register_spawned(&child)?;
    Ok(child)
}

/// Spawns the Renderite Host and registers it with `lifetime`.
pub fn spawn_host(
    config: &ResoBootConfig,
    args: &[String],
    lifetime: &ChildLifetimeGroup,
) -> std::io::Result<Child> {
    if config.is_wine {
        logger::info!("Detected Wine; altering startup sequence accordingly.");
        strip_windows_desktop_from_runtime_config(&config.runtime_config);
        logger::info!("Starting LinuxBootstrap.sh via `start` to run the main program.");
        let mut cmd = Command::new("start");
        cmd.args(["/b", "/unix", "./LinuxBootstrap.sh"]).args(args);
        apply_host_stdio(&mut cmd, &config.current_directory);
        finish_spawn(cmd, lifetime)
    } else {
        let resonite_dir = paths::find_resonite_dir().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Could not find Resonite installation. Set RESONITE_DIR or ensure Steam has Resonite installed.",
            )
        })?;
        logger::info!("Resonite dir: {:?}", resonite_dir);
        prepare_native_host_runtime_config(&resonite_dir);

        let dotnet = paths::find_dotnet_for_host(&resonite_dir);
        let host_dll: PathBuf = resonite_dir.join(paths::RENDERITE_HOST_DLL);
        if !host_dll.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "Renderite.Host.dll not found at {}. Install Resonite with Renderite.",
                    host_dll.display()
                ),
            ));
        }

        logger::info!(
            "Starting Renderite.Host via dotnet at {:?} with {:?}",
            dotnet,
            host_dll
        );
        let mut cmd = Command::new(&dotnet);
        cmd.arg(&host_dll).args(args);
        apply_host_stdio(&mut cmd, &resonite_dir);
        finish_spawn(cmd, lifetime)
    }
}
