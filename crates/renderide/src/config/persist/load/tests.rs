//! Tests for the parent module.

use super::*;
use crate::config::ConfigSource;
use std::ffi::OsString;
use std::io::Write;
use std::path::{Path, PathBuf};
use toml_edit::{DocumentMut, Item};

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
            // SAFETY: env mutation in test; restored while the config env lock is held.
            unsafe {
                match value {
                    Some(v) => std::env::set_var(name, v),
                    None => std::env::remove_var(name),
                }
            }
        }
    }
}

fn write_toml(dir: &Path, body: &str) -> PathBuf {
    let path = dir.join("config.toml");
    let mut file = std::fs::File::create(&path).expect("create fixture file");
    file.write_all(body.as_bytes()).expect("write fixture body");
    path
}

/// Test helper: run the canonical pipeline with an inline TOML string.
fn load_settings_from_toml_str(content: &str) -> Result<RendererSettings, Box<figment::Error>> {
    run_pipeline(Some(content.to_string()))
}

fn document_config_version(document: &DocumentMut) -> &str {
    document
        .get("config_version")
        .and_then(Item::as_value)
        .and_then(|value| value.as_str())
        .expect("config_version")
}

#[test]
fn apply_renderide_gpu_validation_env_overrides_flag() {
    let _guard = crate::config::CONFIG_ENV_TEST_LOCK.lock().expect("lock");
    let mut s = RendererSettings::from_defaults();
    s.debug.gpu_validation_layers = false;
    // SAFETY: env mutation in test; serialized via ENV_LOCK / cargo test single-thread.
    unsafe {
        std::env::set_var("RENDERIDE_GPU_VALIDATION", "1");
    }
    apply_renderide_gpu_validation_env(&mut s);
    assert!(s.debug.gpu_validation_layers);

    s.debug.gpu_validation_layers = true;
    // SAFETY: env mutation in test; serialized via ENV_LOCK / cargo test single-thread.
    unsafe {
        std::env::set_var("RENDERIDE_GPU_VALIDATION", "no");
    }
    apply_renderide_gpu_validation_env(&mut s);
    assert!(!s.debug.gpu_validation_layers);

    // SAFETY: env mutation in test; serialized via ENV_LOCK / cargo test single-thread.
    unsafe {
        std::env::remove_var("RENDERIDE_GPU_VALIDATION");
    }
}

#[test]
fn load_settings_from_toml_merges_renderide_env_nested_key() {
    let _guard = crate::config::CONFIG_ENV_TEST_LOCK.lock().expect("lock");
    // SAFETY: env mutation in test; serialized via ENV_LOCK / cargo test single-thread.
    unsafe {
        std::env::set_var("RENDERIDE_DISPLAY__FOCUSED_FPS", "137");
    }
    let toml = r#"
[display]
focused_fps = 10
"#;
    let s = load_settings_from_toml_str(toml).expect("figment extract");
    assert_eq!(s.display.focused_fps_cap, 137);
    // SAFETY: env mutation in test; serialized via ENV_LOCK / cargo test single-thread.
    unsafe {
        std::env::remove_var("RENDERIDE_DISPLAY__FOCUSED_FPS");
    }
}

#[test]
fn max_local_reflection_probe_env_override_clamps_effective_value() {
    const PROBE_LIMIT_VAR: &str = "RENDERIDE_EXPERIMENTAL__MAX_LOCAL_REFLECTION_PROBES";

    let _guard = crate::config::CONFIG_ENV_TEST_LOCK.lock().expect("lock");
    let _env = EnvGuard::capture(&[PROBE_LIMIT_VAR]);
    // SAFETY: env mutation in test; serialized via CONFIG_ENV_TEST_LOCK and restored by EnvGuard.
    unsafe {
        std::env::set_var(PROBE_LIMIT_VAR, "99");
    }

    let settings = load_settings_from_toml_str("").expect("figment extract");

    assert_eq!(settings.experimental.max_local_reflection_probes, 99);
    assert_eq!(
        settings
            .experimental
            .effective_max_local_reflection_probes(),
        crate::render_contract::MAX_LOCAL_REFLECTION_PROBES
    );
}

#[test]
fn ignore_config_skips_file_and_suppresses_writes() {
    let result = load_renderer_settings(ConfigFilePolicy::Ignore);
    assert_eq!(result.resolve.source, ConfigSource::None);
    assert!(result.resolve.loaded_path.is_none());
    assert!(result.resolve.attempted_paths.is_empty());
    assert!(result.suppress_config_disk_writes);
}

#[test]
fn ignore_config_env_override_still_applies() {
    let _guard = crate::config::CONFIG_ENV_TEST_LOCK.lock().expect("lock");
    // SAFETY: env mutation in test; serialized via ENV_LOCK / cargo test single-thread.
    unsafe {
        std::env::set_var("RENDERIDE_DISPLAY__FOCUSED_FPS", "137");
    }
    let result = load_renderer_settings(ConfigFilePolicy::Ignore);
    assert_eq!(result.settings.display.focused_fps_cap, 137);
    // SAFETY: env mutation in test; serialized via ENV_LOCK / cargo test single-thread.
    unsafe {
        std::env::remove_var("RENDERIDE_DISPLAY__FOCUSED_FPS");
    }
}

#[test]
fn pipeline_layers_apply_in_order() {
    // Defaults -> TOML -> Env -> PostExtract: env overrides TOML, post-extract overrides env.
    let _guard = crate::config::CONFIG_ENV_TEST_LOCK.lock().expect("lock");
    // SAFETY: env mutation in test; serialized via ENV_LOCK / cargo test single-thread.
    unsafe {
        std::env::set_var("RENDERIDE_DISPLAY__FOCUSED_FPS", "200");
    }
    let toml = "[display]\nfocused_fps = 10\n";
    let s = run_pipeline(Some(toml.to_string())).expect("extract");
    assert_eq!(s.display.focused_fps_cap, 200, "env wins over TOML");
    // SAFETY: env mutation in test; serialized via ENV_LOCK / cargo test single-thread.
    unsafe {
        std::env::remove_var("RENDERIDE_DISPLAY__FOCUSED_FPS");
    }
}

#[test]
fn save_path_prefers_loaded() {
    use crate::config::persist::resolve::resolve_save_path;
    use std::path::PathBuf;
    let resolve = ConfigResolveOutcome {
        attempted_paths: vec![],
        loaded_path: Some(PathBuf::from("/tmp/x/config.toml")),
        source: ConfigSource::Search,
    };
    assert_eq!(
        resolve_save_path(&resolve),
        PathBuf::from("/tmp/x/config.toml")
    );
}

/// Regression test: loads a `config.toml` whose enum tokens use older on-disk formats.
/// Removed policy knobs are intentionally still present in the TOML so the full figment
/// pipeline proves stale user configs remain harmless.
#[test]
fn pre_refactor_format_loads_through_figment() {
    let _guard = crate::config::CONFIG_ENV_TEST_LOCK.lock().expect("lock");
    let content = r#"
[display]
focused_fps = 0
unfocused_fps = 0

[rendering]
vsync = "off"
asset_integration_budget_ms = 2
asset_particle_integration_budget_ms = 4
reported_max_texture_size = 0
render_texture_hdr_color = false
texture_vram_budget_mib = 0
msaa = "x8"
scene_color_format = "rgba16_float"
record_parallelism = "PerViewParallel"
cluster_assignment = "auto"
max_frame_latency = 2

[debug]
log_verbose = false
power_preference = "high_performance"
gpu_validation_layers = false
debug_hud_frame_timing = true
debug_hud_enabled = false
debug_hud_transforms = false
debug_hud_textures = false

[post_processing]
enabled = true

[post_processing.bloom]
composite_mode = "energy_conserving"

[post_processing.tonemap]
mode = "aces_fitted"

[watchdog]
action = "log_and_continue"
"#;
    let s = run_pipeline(Some(content.to_string())).expect("figment must accept original tokens");
    use crate::config::types::{
        MsaaSampleCount, PowerPreferenceSetting, SceneColorFormat, TonemapMode, VsyncMode,
        WatchdogAction,
    };
    assert_eq!(s.rendering.vsync, VsyncMode::Off);
    assert_eq!(s.rendering.msaa, MsaaSampleCount::X8);
    assert_eq!(
        s.rendering.scene_color_format,
        SceneColorFormat::Rgba16Float
    );
    assert_eq!(
        s.debug.power_preference,
        PowerPreferenceSetting::HighPerformance
    );
    assert_eq!(s.post_processing.tonemap.mode, TonemapMode::AcesFitted);
    assert_eq!(s.watchdog.action, WatchdogAction::LogAndContinue);
}

#[test]
fn file_pipeline_ignores_unknown_keys_without_drops() {
    let _guard = crate::config::CONFIG_ENV_TEST_LOCK.lock().expect("lock");
    let _env = EnvGuard::capture(&["RENDERIDE_DISPLAY__FOCUSED_FPS"]);
    // SAFETY: env mutation in test; serialized via CONFIG_ENV_TEST_LOCK and restored by EnvGuard.
    unsafe {
        std::env::remove_var("RENDERIDE_DISPLAY__FOCUSED_FPS");
    }
    let content = r#"
[display]
focused_fps = 75
future_display_key = "kept"

[future_renderer]
mode = "future"
"#;

    let load = run_pipeline_tolerating_toml(content).expect("unknown keys should not block load");

    assert_eq!(load.settings.display.focused_fps_cap, 75);
    assert!(
        load.drops.is_empty(),
        "unknown keys should be serde-ignored"
    );
}

#[test]
fn versioned_config_does_not_rerun_auto_exposure_migration() {
    let _guard = crate::config::CONFIG_ENV_TEST_LOCK.lock().expect("lock");
    let content = format!(
        r#"
config_version = "{}"

[post_processing.auto_exposure]
compensation_ev = -3.0
"#,
        RendererSettings::CURRENT_CONFIG_VERSION
    );

    let load = run_pipeline_tolerating_toml(&content).expect("versioned config loads");

    assert!(load.migrated_toml.is_none());
    assert_eq!(
        load.settings.post_processing.auto_exposure.compensation_ev,
        -3.0,
    );
}

#[test]
fn env_compensation_override_wins_without_persisting_to_migrated_file() {
    const CONFIG_VAR: &str = "RENDERIDE_CONFIG";
    const COMPENSATION_VAR: &str = "RENDERIDE_POST_PROCESSING__AUTO_EXPOSURE__COMPENSATION_EV";

    let _lock = crate::config::CONFIG_ENV_TEST_LOCK.lock().expect("lock");
    let _guard = EnvGuard::capture(&[CONFIG_VAR, COMPENSATION_VAR]);
    let tmp = tempfile::tempdir().expect("tempdir");
    let toml = write_toml(
        tmp.path(),
        r#"
[display]
focused_fps = 72
"#,
    );

    // SAFETY: env mutation in test; serialized by CONFIG_ENV_TEST_LOCK.
    unsafe {
        std::env::set_var(CONFIG_VAR, &toml);
        std::env::set_var(COMPENSATION_VAR, "-1.25");
    }

    let result = load_renderer_settings(ConfigFilePolicy::Load);

    assert_eq!(
        result
            .settings
            .post_processing
            .auto_exposure
            .compensation_ev,
        -1.25,
    );
    let text = std::fs::read_to_string(&toml).expect("read migrated file");
    assert!(
        !text.contains("-1.25"),
        "env override should not be persisted:\n{text}"
    );
    let document: DocumentMut = text.parse().expect("persisted config should parse");
    assert_eq!(
        document_config_version(&document),
        RendererSettings::CURRENT_CONFIG_VERSION
    );
    assert!(
        !text.contains("compensation_ev"),
        "env override should not be persisted as file content:\n{text}"
    );
}

#[test]
fn file_pipeline_drops_incompatible_known_value() {
    let _guard = crate::config::CONFIG_ENV_TEST_LOCK.lock().expect("lock");
    let content = r#"
[post_processing.tonemap]
mode = "future_curve"
"#;

    let load = run_pipeline_tolerating_toml(content).expect("future enum token should fall back");

    assert_eq!(
        load.settings.post_processing.tonemap.mode,
        crate::config::TonemapMode::default()
    );
    assert_eq!(load.drops.len(), 1);
    assert_eq!(load.drops[0].path, "post_processing.tonemap.mode");
    assert!(
        load.drops[0].value.contains("future_curve"),
        "drop should report removed value: {:?}",
        load.drops[0]
    );
}

#[test]
fn invalid_toml_suppresses_disk_writes() {
    const CONFIG_VAR: &str = "RENDERIDE_CONFIG";

    let _lock = crate::config::CONFIG_ENV_TEST_LOCK.lock().expect("lock");
    let _guard = EnvGuard::capture(&[CONFIG_VAR]);
    let tmp = tempfile::tempdir().expect("tempdir");
    let toml = write_toml(tmp.path(), "[display\nfocused_fps = 60\n");

    // SAFETY: env mutation in test; serialized by CONFIG_ENV_TEST_LOCK.
    unsafe {
        std::env::set_var(CONFIG_VAR, &toml);
    }

    let result = load_renderer_settings(ConfigFilePolicy::Load);

    assert_eq!(result.resolve.source, ConfigSource::Env);
    assert_eq!(result.resolve.loaded_path.as_deref(), Some(toml.as_path()));
    assert!(result.suppress_config_disk_writes);
}

#[test]
fn incompatible_file_value_does_not_suppress_disk_writes() {
    const CONFIG_VAR: &str = "RENDERIDE_CONFIG";

    let _lock = crate::config::CONFIG_ENV_TEST_LOCK.lock().expect("lock");
    let _guard = EnvGuard::capture(&[CONFIG_VAR]);
    let tmp = tempfile::tempdir().expect("tempdir");
    let toml = write_toml(
        tmp.path(),
        r#"
[post_processing.tonemap]
mode = "future_curve"
"#,
    );

    // SAFETY: env mutation in test; serialized by CONFIG_ENV_TEST_LOCK.
    unsafe {
        std::env::set_var(CONFIG_VAR, &toml);
    }

    let result = load_renderer_settings(ConfigFilePolicy::Load);

    assert_eq!(result.resolve.source, ConfigSource::Env);
    assert_eq!(result.resolve.loaded_path.as_deref(), Some(toml.as_path()));
    assert!(!result.suppress_config_disk_writes);
    assert_eq!(
        result.settings.post_processing.tonemap.mode,
        crate::config::TonemapMode::default()
    );
}

#[test]
fn load_renderer_settings_from_toml_and_env() {
    const CONFIG_VAR: &str = "RENDERIDE_CONFIG";
    const GPU_VALIDATION_VAR: &str = "RENDERIDE_GPU_VALIDATION";
    const GRAPHICS_API_ENV_VAR: &str = "RENDERIDE_RENDERING__GRAPHICS_API";
    const VSYNC_ENV_VAR: &str = "RENDERIDE_RENDERING__VSYNC";

    let _lock = crate::config::CONFIG_ENV_TEST_LOCK.lock().expect("lock");
    let _guard = EnvGuard::capture(&[
        CONFIG_VAR,
        GPU_VALIDATION_VAR,
        GRAPHICS_API_ENV_VAR,
        VSYNC_ENV_VAR,
    ]);
    // SAFETY: env mutation in test; serialized by CONFIG_ENV_TEST_LOCK.
    unsafe {
        std::env::remove_var(GPU_VALIDATION_VAR);
        std::env::remove_var(GRAPHICS_API_ENV_VAR);
        std::env::remove_var(VSYNC_ENV_VAR);
    }

    let tmp = tempfile::tempdir().expect("tempdir");
    let toml = write_toml(
        tmp.path(),
        "[rendering]\nvsync = true\ngraphics_api = \"vulkan\"\n[display]\nfocused_fps = 30\n",
    );

    // SAFETY: env mutation in test; serialized by CONFIG_ENV_TEST_LOCK.
    unsafe {
        std::env::set_var(CONFIG_VAR, &toml);
    }
    let result = load_renderer_settings(ConfigFilePolicy::Load);
    assert_eq!(result.resolve.source, ConfigSource::Env);
    assert_eq!(result.resolve.loaded_path.as_deref(), Some(toml.as_path()));
    assert_eq!(
        result.settings.rendering.vsync,
        crate::config::VsyncMode::On
    );
    assert_eq!(
        result.settings.rendering.graphics_api,
        crate::config::GraphicsApiSetting::Vulkan
    );
    assert_eq!(result.settings.display.focused_fps_cap, 30);

    // SAFETY: env mutation in test; serialized by CONFIG_ENV_TEST_LOCK.
    unsafe {
        std::env::set_var(VSYNC_ENV_VAR, "false");
    }
    let result = load_renderer_settings(ConfigFilePolicy::Load);
    assert_eq!(
        result.settings.rendering.vsync,
        crate::config::VsyncMode::Off
    );
    // SAFETY: env mutation in test; serialized by CONFIG_ENV_TEST_LOCK.
    unsafe {
        std::env::remove_var(VSYNC_ENV_VAR);
        std::env::set_var(GRAPHICS_API_ENV_VAR, "dx12");
    }
    let result = load_renderer_settings(ConfigFilePolicy::Load);
    assert_eq!(
        result.settings.rendering.graphics_api,
        crate::config::GraphicsApiSetting::Dx12
    );
    // SAFETY: env mutation in test; serialized by CONFIG_ENV_TEST_LOCK.
    unsafe {
        std::env::remove_var(GRAPHICS_API_ENV_VAR);
        std::env::set_var(GPU_VALIDATION_VAR, "1");
    }
    let result = load_renderer_settings(ConfigFilePolicy::Load);
    assert!(result.settings.debug.gpu_validation_layers);
    // SAFETY: env mutation in test; serialized by CONFIG_ENV_TEST_LOCK.
    unsafe {
        std::env::set_var(GPU_VALIDATION_VAR, "0");
    }
    let result = load_renderer_settings(ConfigFilePolicy::Load);
    assert!(!result.settings.debug.gpu_validation_layers);

    let missing = tmp.path().join("does_not_exist.toml");
    // SAFETY: env mutation in test; serialized by CONFIG_ENV_TEST_LOCK.
    unsafe {
        std::env::remove_var(GPU_VALIDATION_VAR);
        std::env::set_var(CONFIG_VAR, &missing);
    }
    let result = load_renderer_settings(ConfigFilePolicy::Load);
    assert_ne!(
        result.resolve.loaded_path.as_deref(),
        Some(missing.as_path())
    );
    assert!(result.resolve.attempted_paths.iter().any(|p| p == &missing));
}
