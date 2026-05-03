//! Layered loader for [`super::types::RendererSettings`].
//!
//! The load pipeline is expressed as an explicit ordered [`Vec<ConfigLayer>`] so the precedence
//! chain (defaults -> TOML file -> `RENDERIDE_*` env -> post-extract overrides like
//! `RENDERIDE_GPU_VALIDATION`) is visible in one place. Each layer is one of the variants of
//! [`ConfigLayer`]; pre-extract layers feed the figment merge, post-extract layers run as
//! mutators on the extracted [`super::types::RendererSettings`].
//!
//! [`load_renderer_settings`] is the entry point used by the bootstrap; it builds the canonical
//! pipeline for the requested [`ConfigFilePolicy`] and runs it.

use std::path::PathBuf;

use figment::Figment;
use figment::providers::{Env, Format, Serialized, Toml};

use super::resolve::{
    ConfigResolveOutcome, ConfigSource, apply_generated_config, is_dir_writable, read_config_file,
    renderide_config_env_nonempty, resolve_config_path, resolve_save_path,
};
use super::save::save_renderer_settings;
use super::types::RendererSettings;

/// Controls whether the TOML config file is consulted during startup.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ConfigFilePolicy {
    /// Normal: discover, load, and (if absent) auto-create `config.toml`.
    #[default]
    Load,
    /// Skip all file I/O; use struct defaults plus `RENDERIDE_*` env vars only.
    /// Forces `suppress_config_disk_writes = true`.
    Ignore,
}

/// Full load result: resolved path and save path for persistence.
#[derive(Clone, Debug)]
pub struct ConfigLoadResult {
    /// Effective settings after merge.
    pub settings: RendererSettings,
    /// Path resolution diagnostics.
    pub resolve: ConfigResolveOutcome,
    /// Target file for [`super::save::save_renderer_settings`] and the ImGui config window.
    pub save_path: PathBuf,
    /// When `true`, disk persistence is disabled until restart (Figment extract failed on an
    /// existing file).
    pub suppress_config_disk_writes: bool,
}

/// One step in the [`LoadPipeline`].
///
/// The pre-extract variants ([`Self::Defaults`], [`Self::Toml`], [`Self::EnvPrefixed`]) feed the
/// figment merge; [`Self::PostExtract`] runs as a mutator on the extracted
/// [`RendererSettings`] after extraction. Layers are applied in the order they appear in the
/// pipeline.
pub enum ConfigLayer {
    /// Insert struct defaults via [`Serialized::defaults`].
    Defaults,
    /// Merge an in-memory TOML string. Use this when the resolver located a file on disk and
    /// loaded its contents.
    Toml(String),
    /// Merge `RENDERIDE_*`-style environment variables. `prefix` is the env-var prefix
    /// (typically `"RENDERIDE_"`), `separator` is the nested-key separator
    /// (typically `"__"`).
    EnvPrefixed {
        /// Environment variable prefix (e.g. `"RENDERIDE_"`).
        prefix: &'static str,
        /// Nested-key separator (e.g. `"__"` to map `RENDERIDE_DEBUG__GPU_VALIDATION_LAYERS`
        /// to `debug.gpu_validation_layers`).
        separator: &'static str,
    },
    /// Post-extract mutator. Runs after [`figment::Figment::extract`] succeeds; useful for
    /// special-case env overrides that don't fit the structured `RENDERIDE_*` namespace
    /// (currently only `RENDERIDE_GPU_VALIDATION`).
    PostExtract(fn(&mut RendererSettings)),
}

/// An ordered chain of [`ConfigLayer`] entries. Construct with [`LoadPipeline::new`] then push
/// layers, or build the canonical chain via [`canonical_layers`] / [`load_renderer_settings`].
#[derive(Default)]
pub struct LoadPipeline {
    layers: Vec<ConfigLayer>,
}

impl LoadPipeline {
    /// Empty pipeline (no defaults inserted yet -- the canonical chain always starts with
    /// [`ConfigLayer::Defaults`]).
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a layer to the pipeline.
    pub fn push(&mut self, layer: ConfigLayer) -> &mut Self {
        self.layers.push(layer);
        self
    }

    /// Runs the pipeline: builds the figment from pre-extract layers, extracts
    /// [`RendererSettings`], then runs post-extract mutators in order.
    pub fn extract(self) -> Result<RendererSettings, Box<figment::Error>> {
        let mut figment = Figment::new();
        let mut mutators: Vec<fn(&mut RendererSettings)> = Vec::new();
        for layer in self.layers {
            match layer {
                ConfigLayer::Defaults => {
                    figment = figment.merge(Serialized::defaults(RendererSettings::default()));
                }
                ConfigLayer::Toml(content) => {
                    figment = figment.merge(Toml::string(&content));
                }
                ConfigLayer::EnvPrefixed { prefix, separator } => {
                    figment = figment.merge(Env::prefixed(prefix).split(separator));
                }
                ConfigLayer::PostExtract(f) => {
                    mutators.push(f);
                }
            }
        }
        let mut settings = figment.extract::<RendererSettings>().map_err(Box::new)?;
        for f in mutators {
            f(&mut settings);
        }
        Ok(settings)
    }
}

/// Builds the canonical `RENDERIDE_*` env layering with the post-extract
/// [`apply_renderide_gpu_validation_env`] mutator, optionally including a TOML layer when
/// `toml_content` is provided.
pub fn canonical_layers(toml_content: Option<String>) -> Vec<ConfigLayer> {
    let mut v = Vec::with_capacity(4);
    v.push(ConfigLayer::Defaults);
    if let Some(content) = toml_content {
        v.push(ConfigLayer::Toml(content));
    }
    v.push(ConfigLayer::EnvPrefixed {
        prefix: "RENDERIDE_",
        separator: "__",
    });
    v.push(ConfigLayer::PostExtract(apply_renderide_gpu_validation_env));
    v
}

/// Overrides [`super::types::DebugSettings::gpu_validation_layers`] when
/// `RENDERIDE_GPU_VALIDATION` is set.
///
/// Truthy values (`1`, `true`, `yes`) force validation on; falsey (`0`, `false`, `no`) force
/// off. If unset, the value from config or defaults is unchanged. Wired into the canonical
/// pipeline as a [`ConfigLayer::PostExtract`] entry so the precedence rule lives next to the
/// other layers.
pub fn apply_renderide_gpu_validation_env(settings: &mut RendererSettings) {
    match std::env::var("RENDERIDE_GPU_VALIDATION").as_deref() {
        Ok("1" | "true" | "yes") => settings.debug.gpu_validation_layers = true,
        Ok("0" | "false" | "no") => settings.debug.gpu_validation_layers = false,
        _ => {}
    }
}

/// Resolves `config.toml`, runs the canonical [`LoadPipeline`], and produces a
/// [`ConfigLoadResult`].
///
/// Precedence (top wins): post-extract mutators (`RENDERIDE_GPU_VALIDATION`) -> `RENDERIDE_*`
/// env -> TOML file (skipped under [`ConfigFilePolicy::Ignore`]) -> struct defaults.
///
/// When no file exists and [`renderide_config_env_nonempty`] is false, writes defaults to the
/// save path (see [`super::resolve::resolve_save_path`]) and loads that file. This
/// auto-creation is skipped when `policy` is [`ConfigFilePolicy::Ignore`].
pub fn load_renderer_settings(policy: ConfigFilePolicy) -> ConfigLoadResult {
    if policy == ConfigFilePolicy::Ignore {
        return load_with_ignore_policy();
    }

    let mut resolve = resolve_config_path();
    let mut suppress_config_disk_writes = false;
    let mut settings = initial_settings_from_resolve(&mut suppress_config_disk_writes, &resolve);

    if resolve.loaded_path.is_none() && !renderide_config_env_nonempty() {
        maybe_create_default_config_and_reload(
            &mut resolve,
            &mut settings,
            &mut suppress_config_disk_writes,
        );
    }

    let save_path = resolve_save_path(&resolve);
    logger::trace!("Renderer config will persist to {}", save_path.display());

    ConfigLoadResult {
        settings,
        resolve,
        save_path,
        suppress_config_disk_writes,
    }
}

/// Builds the [`ConfigFilePolicy::Ignore`] result: skip TOML, run defaults+env+overrides only,
/// and force `suppress_config_disk_writes`.
fn load_with_ignore_policy() -> ConfigLoadResult {
    if renderide_config_env_nonempty() {
        logger::warn!(
            "--ignore-config is active; RENDERIDE_CONFIG is also set but the file will be skipped"
        );
    }
    let settings = match run_pipeline(None) {
        Ok(s) => s,
        Err(e) => {
            logger::error!(
                "Renderer config Figment extract failed (--ignore-config, defaults+env): {e:#}"
            );
            RendererSettings::default()
        }
    };
    let resolve = ConfigResolveOutcome {
        attempted_paths: vec![],
        loaded_path: None,
        source: ConfigSource::None,
    };
    let save_path = resolve_save_path(&resolve);
    logger::info!("--ignore-config: skipping TOML file; using struct defaults + RENDERIDE_* env");
    ConfigLoadResult {
        settings,
        resolve,
        save_path,
        suppress_config_disk_writes: true,
    }
}

/// Runs the canonical pipeline with optional TOML content.
fn run_pipeline(toml_content: Option<String>) -> Result<RendererSettings, Box<figment::Error>> {
    let mut pipeline = LoadPipeline::new();
    for layer in canonical_layers(toml_content) {
        pipeline.push(layer);
    }
    pipeline.extract()
}

/// Loads settings from a resolved config path, or defaults plus env when the file is missing or
/// unreadable.
fn initial_settings_from_resolve(
    suppress_config_disk_writes: &mut bool,
    resolve: &ConfigResolveOutcome,
) -> RendererSettings {
    if let Some(path) = resolve.loaded_path.as_ref() {
        logger::info!("Loading renderer config from {}", path.display());
        match read_config_file(path) {
            Ok(content) => match run_pipeline(Some(content)) {
                Ok(s) => s,
                Err(e) => {
                    logger::error!(
                        "Renderer config Figment extract failed for {}: {e:#}",
                        path.display()
                    );
                    *suppress_config_disk_writes = true;
                    RendererSettings::default()
                }
            },
            Err(e) => {
                logger::warn!("Failed to read {}: {e}; using defaults", path.display());
                fallback_to_defaults_plus_env(suppress_config_disk_writes)
            }
        }
    } else {
        logger::info!("Renderer config file not found; using built-in defaults");
        logger::trace!(
            "config search tried {} path(s)",
            resolve.attempted_paths.len()
        );
        fallback_to_defaults_plus_env(suppress_config_disk_writes)
    }
}

/// Runs the pipeline without a TOML layer (defaults + env + post-extract overrides) and falls
/// back to [`RendererSettings::default`] on Figment failure.
fn fallback_to_defaults_plus_env(suppress_config_disk_writes: &mut bool) -> RendererSettings {
    match run_pipeline(None) {
        Ok(s) => s,
        Err(e) => {
            logger::error!("Renderer config Figment extract failed (defaults+env): {e:#}");
            *suppress_config_disk_writes = true;
            RendererSettings::default()
        }
    }
}

/// When no config was loaded and env overrides are empty, writes default `config.toml` and
/// reloads from disk.
fn maybe_create_default_config_and_reload(
    resolve: &mut ConfigResolveOutcome,
    settings: &mut RendererSettings,
    suppress_config_disk_writes: &mut bool,
) {
    let path = resolve_save_path(resolve);
    if path.exists() {
        return;
    }
    let Some(parent) = path.parent() else {
        return;
    };
    if !is_dir_writable(parent) {
        logger::trace!(
            "Not creating default config at {} (directory not writable)",
            path.display()
        );
        return;
    }
    match save_renderer_settings(&path, &RendererSettings::from_defaults()) {
        Ok(()) => {
            logger::info!("Created default renderer config at {}", path.display());
            apply_generated_config(resolve, path.clone());
            match read_config_file(&path) {
                Ok(content) => match run_pipeline(Some(content)) {
                    Ok(s) => {
                        *settings = s;
                    }
                    Err(e) => {
                        logger::error!(
                            "Figment extract failed for newly created {}: {e:#}",
                            path.display()
                        );
                        *suppress_config_disk_writes = true;
                    }
                },
                Err(e) => {
                    logger::warn!(
                        "Failed to read newly created {}: {e}; using defaults",
                        path.display()
                    );
                }
            }
        }
        Err(e) => {
            logger::warn!("Failed to create default config at {}: {e}", path.display());
        }
    }
}

/// Logs [`ConfigLoadResult::resolve`] at trace level for troubleshooting.
pub fn log_config_resolve_trace(resolve: &ConfigResolveOutcome) {
    if resolve.source == ConfigSource::None && !resolve.attempted_paths.is_empty() {
        for p in &resolve.attempted_paths {
            let exists = p.as_path().is_file();
            logger::trace!("  config candidate {} [{}]", p.display(), exists);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::resolve::ConfigSource;

    /// Test helper: run the canonical pipeline with an inline TOML string.
    fn load_settings_from_toml_str(content: &str) -> Result<RendererSettings, Box<figment::Error>> {
        run_pipeline(Some(content.to_string()))
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
        use crate::config::resolve::resolve_save_path;
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

    /// Regression test: loads a `config.toml` whose every enum token uses the original
    /// pre-refactor on-disk format (serde's `snake_case` for the section that had it,
    /// PascalCase for `RecordParallelism` which had no `rename_all`). Goes through the full
    /// figment pipeline rather than a bare `toml::from_str` so it catches the path the
    /// renderer actually exercises at startup.
    #[test]
    fn pre_refactor_format_loads_through_figment() {
        let content = r#"
[display]
focused_fps = 0
unfocused_fps = 0

[rendering]
vsync = "off"
asset_integration_budget_ms = 3
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
        let s =
            run_pipeline(Some(content.to_string())).expect("figment must accept original tokens");
        use crate::config::types::{
            ClusterAssignmentMode, MsaaSampleCount, PowerPreferenceSetting, RecordParallelism,
            SceneColorFormat, TonemapMode, VsyncMode, WatchdogAction,
        };
        assert_eq!(s.rendering.vsync, VsyncMode::Off);
        assert_eq!(s.rendering.msaa, MsaaSampleCount::X8);
        assert_eq!(
            s.rendering.scene_color_format,
            SceneColorFormat::Rgba16Float
        );
        assert_eq!(
            s.rendering.record_parallelism,
            RecordParallelism::PerViewParallel
        );
        assert_eq!(s.rendering.cluster_assignment, ClusterAssignmentMode::Auto);
        assert_eq!(
            s.debug.power_preference,
            PowerPreferenceSetting::HighPerformance
        );
        assert_eq!(s.post_processing.tonemap.mode, TonemapMode::AcesFitted);
        assert_eq!(s.watchdog.action, WatchdogAction::LogAndContinue);
    }
}
