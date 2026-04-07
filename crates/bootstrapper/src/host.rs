//! Spawning Renderite Host (Wine + `LinuxBootstrap.sh` vs `dotnet Renderite.Host.dll`).

use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};

use serde_json::Value;

use crate::child_lifetime::ChildLifetimeGroup;
use crate::config::ResoBootConfig;
use crate::paths;

/// Removes `Microsoft.WindowsDesktop.App` from `runtimeOptions.frameworks` for Wine compatibility.
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

/// Drains a reader into a log file line-by-line with a prefix.
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
            while buf_reader.read_line(&mut line).is_ok_and(|n| n > 0) {
                let _ = writeln!(file, "{} {}", prefix, line.trim_end());
                let _ = file.flush();
                line.clear();
            }
        }
    });
}

/// Raises Host process priority on Windows (ResoBoot `AboveNormal`).
#[cfg(windows)]
pub fn set_host_above_normal_priority(child: &Child) {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::System::Threading::{SetPriorityClass, ABOVE_NORMAL_PRIORITY_CLASS};

    let handle = child.as_raw_handle() as *mut std::ffi::c_void;
    let rc = unsafe { SetPriorityClass(handle, ABOVE_NORMAL_PRIORITY_CLASS) };
    if rc == 0 {
        logger::warn!(
            "SetPriorityClass failed: {}",
            std::io::Error::last_os_error()
        );
    } else {
        logger::info!("Host process priority set to AboveNormal");
    }
}

#[cfg(not(windows))]
pub fn set_host_above_normal_priority(_child: &Child) {}

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

        let dotnet = paths::find_dotnet_for_host(&resonite_dir);
        let host_dll = resonite_dir.join(paths::RENDERITE_HOST_DLL);
        if !host_dll.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "Renderite.Host.dll not found at {:?}. Install Resonite with Renderite.",
                    host_dll
                ),
            ));
        }

        logger::info!(
            "Starting Renderite.Host via dotnet at {:?} with {:?}",
            dotnet,
            host_dll
        );
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
