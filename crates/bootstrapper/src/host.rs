//! Spawning Renderite Host (Wine + `LinuxBootstrap.sh` vs `dotnet Renderite.Host.dll`).

use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

use serde_json::Value;

use crate::child_lifetime::ChildLifetimeGroup;
use crate::config::ResoBootConfig;
use crate::paths;

/// Removes `Microsoft.WindowsDesktop.App` from `runtimeOptions.frameworks` for Wine compatibility.
pub(crate) fn strip_windows_desktop_from_runtime_config(path: &Path) {
    if !path.exists() {
        return;
    }
    let contents = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            logger::warn!("Could not read runtime config {:?}: {}", path, e);
            return;
        }
    };
    let mut json: Value = match serde_json::from_str(&contents) {
        Ok(j) => j,
        Err(e) => {
            logger::warn!("Could not parse runtime config {:?}: {}", path, e);
            return;
        }
    };
    let stripped_any = if let Some(frameworks) = json
        .get_mut("runtimeOptions")
        .and_then(|o| o.get_mut("frameworks"))
        .and_then(|f| f.as_array_mut())
    {
        let before_len = frameworks.len();
        frameworks.retain(|node| {
            node.get("name").and_then(|n| n.as_str()) != Some("Microsoft.WindowsDesktop.App")
        });
        before_len != frameworks.len()
    } else {
        false
    };
    if !stripped_any {
        return;
    }
    let new_contents = match serde_json::to_string_pretty(&json) {
        Ok(s) => s,
        Err(e) => {
            logger::warn!("Could not serialize runtime config {:?}: {}", path, e);
            return;
        }
    };
    if let Err(e) = fs::write(path, new_contents) {
        logger::warn!("Could not write runtime config {:?}: {}", path, e);
    }
}

/// Drains a reader into a log file line-by-line with a prefix.
pub(crate) fn spawn_output_drainer(
    log_path: PathBuf,
    reader: impl Read + Send + 'static,
    prefix: &'static str,
) {
    std::thread::spawn(move || {
        let mut file = match fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
        {
            Ok(f) => f,
            Err(e) => {
                logger::warn!("Could not open host log {:?} for drainer: {}", log_path, e);
                return;
            }
        };
        let mut buf_reader = BufReader::new(reader);
        let mut line = String::new();
        while buf_reader.read_line(&mut line).is_ok_and(|n| n > 0) {
            let _ = writeln!(file, "{} {}", prefix, line.trim_end());
            let _ = file.flush();
            line.clear();
        }
    });
}

/// Configures stdio pipes and working directory for a Host launch.
fn apply_host_stdio(cmd: &mut Command, cwd: &Path) {
    cmd.current_dir(cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
}

/// Prepares the command, spawns, and registers the child with `lifetime`.
fn finish_spawn(mut cmd: Command, lifetime: &ChildLifetimeGroup) -> std::io::Result<Child> {
    lifetime.prepare_command(&mut cmd);
    let child = cmd.spawn()?;
    lifetime.register_spawned(&child)?;
    Ok(child)
}

/// Raises Host process priority on Windows (ResoBoot `AboveNormal`).
#[cfg(windows)]
pub fn set_host_above_normal_priority(child: &Child) {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::System::Threading::{SetPriorityClass, ABOVE_NORMAL_PRIORITY_CLASS};

    let handle = child.as_raw_handle();
    // SAFETY: `handle` is a valid process handle from `Child` until the child is reaped.
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
pub(crate) fn set_host_above_normal_priority(_child: &Child) {}

/// Spawns the Renderite Host and registers it with `lifetime`.
pub(crate) fn spawn_host(
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

        let dotnet = paths::find_dotnet_for_host(&resonite_dir);
        let host_dll: PathBuf = resonite_dir.join(paths::RENDERITE_HOST_DLL);
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
        cmd.arg(&host_dll).args(args);
        apply_host_stdio(&mut cmd, &resonite_dir);
        finish_spawn(cmd, lifetime)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;
    use std::io::Cursor;

    #[test]
    fn strip_windows_desktop_noop_when_missing_file() {
        let path = std::env::temp_dir().join(format!(
            "bootstrapper_runtime_cfg_missing_{}",
            std::process::id()
        ));
        let _ = fs::remove_file(&path);
        strip_windows_desktop_from_runtime_config(&path);
    }

    #[test]
    fn strip_windows_desktop_removes_desktop_framework() {
        let path = std::env::temp_dir().join(format!(
            "bootstrapper_runtime_cfg_strip_{}",
            std::process::id()
        ));
        let before = json!({
            "runtimeOptions": {
                "frameworks": [
                    {"name": "Microsoft.NETCore.App", "version": "8.0.0"},
                    {"name": "Microsoft.WindowsDesktop.App", "version": "8.0.0"}
                ]
            }
        });
        fs::write(&path, serde_json::to_string_pretty(&before).unwrap()).unwrap();
        strip_windows_desktop_from_runtime_config(&path);
        let after: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        let frameworks = after["runtimeOptions"]["frameworks"].as_array().unwrap();
        assert_eq!(frameworks.len(), 1);
        assert_eq!(
            frameworks[0]["name"].as_str(),
            Some("Microsoft.NETCore.App")
        );
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn strip_windows_desktop_idempotent() {
        let path = std::env::temp_dir().join(format!(
            "bootstrapper_runtime_cfg_idem_{}",
            std::process::id()
        ));
        let before = json!({
            "runtimeOptions": {
                "frameworks": [
                    {"name": "Microsoft.NETCore.App", "version": "8.0.0"},
                    {"name": "Microsoft.WindowsDesktop.App", "version": "8.0.0"}
                ]
            }
        });
        fs::write(&path, serde_json::to_string_pretty(&before).unwrap()).unwrap();
        strip_windows_desktop_from_runtime_config(&path);
        let once = fs::read_to_string(&path).unwrap();
        strip_windows_desktop_from_runtime_config(&path);
        let twice = fs::read_to_string(&path).unwrap();
        assert_eq!(once, twice);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn strip_windows_desktop_no_runtime_options_no_rewrite() {
        let path = std::env::temp_dir().join(format!(
            "bootstrapper_runtime_cfg_noro_{}",
            std::process::id()
        ));
        let before = json!({ "other": 1 });
        fs::write(&path, serde_json::to_string_pretty(&before).unwrap()).unwrap();
        strip_windows_desktop_from_runtime_config(&path);
        let after: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(after, before);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn strip_windows_desktop_empty_frameworks_array() {
        let path = std::env::temp_dir().join(format!(
            "bootstrapper_runtime_cfg_empty_fw_{}",
            std::process::id()
        ));
        let before = json!({ "runtimeOptions": { "frameworks": [] } });
        fs::write(&path, serde_json::to_string_pretty(&before).unwrap()).unwrap();
        strip_windows_desktop_from_runtime_config(&path);
        let after: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(
            after["runtimeOptions"]["frameworks"]
                .as_array()
                .unwrap()
                .len(),
            0
        );
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn strip_windows_desktop_only_desktop_framework_removed() {
        let path = std::env::temp_dir().join(format!(
            "bootstrapper_runtime_cfg_only_desktop_{}",
            std::process::id()
        ));
        let before = json!({
            "runtimeOptions": {
                "frameworks": [
                    {"name": "Microsoft.WindowsDesktop.App", "version": "8.0.0"}
                ]
            }
        });
        fs::write(&path, serde_json::to_string_pretty(&before).unwrap()).unwrap();
        strip_windows_desktop_from_runtime_config(&path);
        let after: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        assert!(after["runtimeOptions"]["frameworks"]
            .as_array()
            .unwrap()
            .is_empty());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn strip_windows_desktop_invalid_json_is_noop() {
        let path = std::env::temp_dir().join(format!(
            "bootstrapper_runtime_cfg_bad_{}",
            std::process::id()
        ));
        fs::write(&path, b"not json").unwrap();
        strip_windows_desktop_from_runtime_config(&path);
        assert_eq!(fs::read_to_string(&path).unwrap(), "not json");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn spawn_output_drainer_writes_prefixed_lines() {
        let log_path =
            std::env::temp_dir().join(format!("bootstrapper_drainer_{}.log", std::process::id()));
        let _ = fs::remove_file(&log_path);
        let input = b"line one\nline two\n";
        spawn_output_drainer(log_path.clone(), Cursor::new(input), "[P]");
        std::thread::sleep(std::time::Duration::from_millis(200));
        let out = fs::read_to_string(&log_path).expect("read log");
        assert!(out.contains("[P] line one"));
        assert!(out.contains("[P] line two"));
        let _ = fs::remove_file(&log_path);
    }
}
