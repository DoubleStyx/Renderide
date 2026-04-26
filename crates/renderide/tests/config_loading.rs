//! Integration test: load `RendererSettings` through `renderide::config::load_renderer_settings`
//! using a TOML fixture routed via `RENDERIDE_CONFIG`.
//!
//! `RENDERIDE_CONFIG` is process-global and `cargo` may schedule `#[test]` fns in this binary
//! concurrently, so every case runs serialized inside a single `#[test]` fn that holds an RAII
//! guard restoring the original environment on exit.

#![expect(
    clippy::expect_used,
    reason = "integration test fixtures panic on setup failure"
)]

use std::ffi::OsString;
use std::io::Write;
use std::path::PathBuf;

use renderide::config::{load_renderer_settings, ConfigFilePolicy, ConfigSource};

const CONFIG_VAR: &str = "RENDERIDE_CONFIG";
const GPU_VALIDATION_VAR: &str = "RENDERIDE_GPU_VALIDATION";
const VSYNC_ENV_VAR: &str = "RENDERIDE_RENDERING__VSYNC";

/// RAII guard that restores a set of environment variables to their pre-test state on drop so one
/// test case cannot leak state into another (or into other integration binaries that happen to
/// share the same process-wide env space during a parallel cargo run).
struct EnvGuard {
    saved: Vec<(&'static str, Option<OsString>)>,
}

impl EnvGuard {
    fn capture(vars: &[&'static str]) -> Self {
        let saved = vars
            .iter()
            .map(|name| (*name, std::env::var_os(name)))
            .collect();
        Self { saved }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (name, value) in &self.saved {
            match value {
                Some(v) => std::env::set_var(name, v),
                None => std::env::remove_var(name),
            }
        }
    }
}

fn write_toml(dir: &std::path::Path, body: &str) -> PathBuf {
    let path = dir.join("config.toml");
    let mut f = std::fs::File::create(&path).expect("create fixture file");
    f.write_all(body.as_bytes()).expect("write fixture body");
    path
}

/// All cases run under one `#[test]` so they serialize on process-wide env vars. Each case sets
/// its own env state before calling `load_renderer_settings()` and resets the env vars that would
/// otherwise bleed across cases.
#[test]
fn load_renderer_settings_from_toml_and_env() {
    let _guard = EnvGuard::capture(&[CONFIG_VAR, GPU_VALIDATION_VAR, VSYNC_ENV_VAR]);
    let tmp = tempfile::tempdir().expect("tempdir");

    // Clear overrides that might leak from the host shell.
    std::env::remove_var(GPU_VALIDATION_VAR);
    std::env::remove_var(VSYNC_ENV_VAR);

    // --- Case 1: TOML file values reach the resolved settings ---
    // vsync default is `false`; set it to `true` so the override is observable. focused_fps default
    // is `240`, so 30 is distinct.
    let toml = write_toml(
        tmp.path(),
        "[rendering]\nvsync = true\n[display]\nfocused_fps = 30\n",
    );
    std::env::set_var(CONFIG_VAR, &toml);

    let result = load_renderer_settings(ConfigFilePolicy::Load);
    assert_eq!(
        result.resolve.source,
        ConfigSource::Env,
        "expected Env-sourced resolve when RENDERIDE_CONFIG points at an existing file; got {:?}",
        result.resolve.source
    );
    assert_eq!(
        result.resolve.loaded_path.as_deref(),
        Some(toml.as_path()),
        "resolve.loaded_path must match the fixture"
    );
    assert_eq!(
        result.settings.rendering.vsync,
        renderide::config::VsyncMode::On,
        "TOML vsync=true must deserialize as VsyncMode::On"
    );
    assert_eq!(
        result.settings.display.focused_fps_cap, 30,
        "TOML display.focused_fps (serde rename of focused_fps_cap) should reach the loaded settings"
    );

    // --- Case 2: RENDERIDE_* env override beats the TOML value ---
    std::env::set_var(VSYNC_ENV_VAR, "false");
    let result = load_renderer_settings(ConfigFilePolicy::Load);
    assert_eq!(
        result.settings.rendering.vsync,
        renderide::config::VsyncMode::Off,
        "RENDERIDE_RENDERING__VSYNC=false must override TOML vsync=true to VsyncMode::Off"
    );
    std::env::remove_var(VSYNC_ENV_VAR);

    // --- Case 3: RENDERIDE_GPU_VALIDATION flips the post-extract override ---
    std::env::set_var(GPU_VALIDATION_VAR, "1");
    let result = load_renderer_settings(ConfigFilePolicy::Load);
    assert!(
        result.settings.debug.gpu_validation_layers,
        "RENDERIDE_GPU_VALIDATION=1 must force gpu_validation_layers on"
    );
    std::env::set_var(GPU_VALIDATION_VAR, "0");
    let result = load_renderer_settings(ConfigFilePolicy::Load);
    assert!(
        !result.settings.debug.gpu_validation_layers,
        "RENDERIDE_GPU_VALIDATION=0 must force gpu_validation_layers off"
    );
    std::env::remove_var(GPU_VALIDATION_VAR);

    // --- Case 4: RENDERIDE_CONFIG pointing at a missing file falls through to search / defaults ---
    let missing = tmp.path().join("does_not_exist.toml");
    std::env::set_var(CONFIG_VAR, &missing);
    let result = load_renderer_settings(ConfigFilePolicy::Load);
    // Loader may or may not find a file via the subsequent search; the contract is that
    // RENDERIDE_CONFIG pointing at a non-existent path does not become the effective `loaded_path`.
    assert_ne!(
        result.resolve.loaded_path.as_deref(),
        Some(missing.as_path()),
        "missing RENDERIDE_CONFIG path must never be reported as loaded_path"
    );
    assert!(
        result.resolve.attempted_paths.iter().any(|p| p == &missing),
        "missing RENDERIDE_CONFIG path must appear in attempted_paths: {:?}",
        result.resolve.attempted_paths
    );
}
