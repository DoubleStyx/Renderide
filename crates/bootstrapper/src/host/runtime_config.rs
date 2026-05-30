//! Runtime configuration adjustments needed before Host startup.

use std::fs;
use std::path::Path;

use serde_json::Value;

/// Runtime configuration file shipped next to `Renderite.Host.dll`.
#[cfg(any(target_os = "linux", target_os = "macos"))]
const RENDERITE_HOST_RUNTIME_CONFIG: &str = "Renderite.Host.runtimeconfig.json";

/// Removes `Microsoft.WindowsDesktop.App` from `runtimeOptions.frameworks` for native Unix compatibility.
pub fn strip_windows_desktop_from_runtime_config(path: &Path) {
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

/// Returns the runtimeconfig path for a native Host launch directory.
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn host_runtime_config_path(resonite_dir: &Path) -> std::path::PathBuf {
    resonite_dir.join(RENDERITE_HOST_RUNTIME_CONFIG)
}

/// Applies runtimeconfig adjustments needed before native Unix Host startup.
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub(super) fn prepare_native_host_runtime_config(resonite_dir: &Path) {
    strip_windows_desktop_from_runtime_config(&host_runtime_config_path(resonite_dir));
}

/// Keeps native Host startup preparation explicit on platforms without runtimeconfig edits.
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(super) fn prepare_native_host_runtime_config(_resonite_dir: &Path) {}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn host_runtime_config_path_uses_host_directory() {
        let root = std::path::PathBuf::from("renderite-host");

        assert_eq!(
            host_runtime_config_path(&root),
            root.join(RENDERITE_HOST_RUNTIME_CONFIG)
        );
    }

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
    fn strip_windows_desktop_non_array_frameworks_is_noop() {
        let path = std::env::temp_dir().join(format!(
            "bootstrapper_runtime_cfg_fw_object_{}",
            std::process::id()
        ));
        let before =
            json!({ "runtimeOptions": { "frameworks": {"name": "Microsoft.WindowsDesktop.App"} } });
        fs::write(&path, serde_json::to_string_pretty(&before).unwrap()).unwrap();

        strip_windows_desktop_from_runtime_config(&path);

        let after: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(after, before);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn strip_windows_desktop_retains_framework_entries_without_name() {
        let path = std::env::temp_dir().join(format!(
            "bootstrapper_runtime_cfg_missing_name_{}",
            std::process::id()
        ));
        let before = json!({
            "runtimeOptions": {
                "frameworks": [
                    {"version": "8.0.0"},
                    {"name": "Microsoft.WindowsDesktop.App", "version": "8.0.0"},
                    {"name": "Microsoft.NETCore.App", "version": "8.0.0"}
                ]
            }
        });
        fs::write(&path, serde_json::to_string_pretty(&before).unwrap()).unwrap();

        strip_windows_desktop_from_runtime_config(&path);

        let after: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        let frameworks = after["runtimeOptions"]["frameworks"].as_array().unwrap();
        assert_eq!(frameworks.len(), 2);
        assert!(frameworks.iter().any(|f| f.get("name").is_none()));
        assert!(frameworks.iter().any(|f| {
            f.get("name").and_then(|name| name.as_str()) == Some("Microsoft.NETCore.App")
        }));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn strip_windows_desktop_removes_all_desktop_framework_entries() {
        let path = std::env::temp_dir().join(format!(
            "bootstrapper_runtime_cfg_multi_desktop_{}",
            std::process::id()
        ));
        let before = json!({
            "runtimeOptions": {
                "frameworks": [
                    {"name": "Microsoft.WindowsDesktop.App", "version": "7.0.0"},
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
        assert!(
            after["runtimeOptions"]["frameworks"]
                .as_array()
                .unwrap()
                .is_empty()
        );
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
}
