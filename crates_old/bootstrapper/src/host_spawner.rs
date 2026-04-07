//! Host process spawning for Wine (LinuxBootstrap.sh) and native Linux (dotnet).
//! Handles Wine runtime config stripping and output drain threads.

use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};

use serde_json::Value;

use crate::config::ResoBootConfig;
use crate::paths;
use crate::process_lifetime::ChildLifetimeGroup;

/// Removes Microsoft.WindowsDesktop.App from runtime config for Wine compatibility.
pub fn strip_windows_desktop_from_runtime_config(path: &Path) {
    if !path.exists() {
        return;
    }
    let Ok(contents) = fs::read_to_string(path) else {
        return;
    };
    let Ok(mut json) = serde_json::from_str::<Value>(&contents) else {
        return;
    };
    if let Some(frameworks) = json
        .get_mut("runtimeOptions")
        .and_then(|o| o.get_mut("frameworks"))
        .and_then(|f| f.as_array_mut())
    {
        frameworks.retain(|node| {
            node.get("name").and_then(|n| n.as_str()) != Some("Microsoft.WindowsDesktop.App")
        });
    }
    if let Ok(new_contents) = serde_json::to_string_pretty(&json) {
        let _ = fs::write(path, new_contents);
    }
}

/// Spawns a thread that drains the given reader into the log file with the given prefix.
pub fn spawn_output_drainer(
    log_path: std::path::PathBuf,
    reader: impl Read + Send + 'static,
    prefix: &'static str,
) {
    std::thread::spawn(move || {
        if let Ok(mut file) = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
        {
            let mut buf_reader = BufReader::new(reader);
            let mut line = String::new();
            while buf_reader
                .read_line(&mut line)
                .map(|n| n > 0)
                .unwrap_or(false)
            {
                let _ = writeln!(file, "{} {}", prefix, line.trim_end());
                let _ = file.flush();
                line.clear();
            }
        }
    });
}

/// Spawns the Renderite Host process. Returns the child process or an error.
///
/// `lifetime` ties the child to bootstrapper exit (job object on Windows, parent death signal on Linux).
pub fn spawn_host(
    config: &ResoBootConfig,
    args: &[String],
    lifetime: &ChildLifetimeGroup,
) -> std::io::Result<Child> {
    if config.is_wine {
        logger::info!("Detected Wine; altering startup sequence accordingly.");
        strip_windows_desktop_from_runtime_config(&config.runtime_config);
        logger::info!(
            "Starting LinuxBootstrap.sh to check for dotnet and execute the main program."
        );
        let mut cmd = Command::new("start");
        cmd.args(["/b", "/unix", "./LinuxBootstrap.sh"])
            .args(args)
            .current_dir(&config.current_directory)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        lifetime.prepare_command(&mut cmd);
        let child = cmd.spawn()?;
        lifetime.register_spawned(&child)?;
        Ok(child)
    } else {
        let resonite_dir = paths::find_resonite_dir().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Could not find Resonite installation. Set RESONITE_DIR or ensure Steam has Resonite installed.",
            )
        })?;
        logger::info!("Resonite dir: {:?}", resonite_dir);
        logger::info!("Starting Renderite.Host from {:?}", resonite_dir);

        #[cfg(target_os = "linux")]
        {
            let dotnet = paths::find_dotnet_for_host(&resonite_dir);
            let host_dll = resonite_dir.join(paths::RENDERITE_HOST_DLL);
            logger::info!("Using dotnet at {:?} to run Renderite.Host.dll", dotnet);
            let mut cmd = Command::new(&dotnet);
            cmd.arg(&host_dll)
                .args(args)
                .current_dir(&resonite_dir)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            lifetime.prepare_command(&mut cmd);
            let child = cmd.spawn()?;
            lifetime.register_spawned(&child)?;
            Ok(child)
        }

        #[cfg(not(target_os = "linux"))]
        {
            // On Windows, use bundled dotnet when available (like Linux) to avoid runtime version mismatch.
            let dotnet = paths::find_dotnet_for_host(&resonite_dir);
            let host_dll = resonite_dir.join(paths::RENDERITE_HOST_DLL);
            if !host_dll.exists() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!(
                        "Renderite.Host.dll not found at {:?}. Ensure Resonite with Renderite mod is installed.",
                        host_dll
                    ),
                ));
            }
            logger::info!("Using dotnet at {:?} to run Renderite.Host.dll", dotnet);
            let mut cmd = Command::new(&dotnet);
            cmd.arg(&host_dll)
                .args(args)
                .current_dir(&resonite_dir)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            lifetime.prepare_command(&mut cmd);
            let child = cmd.spawn()?;
            lifetime.register_spawned(&child)?;
            Ok(child)
        }
    }
}
