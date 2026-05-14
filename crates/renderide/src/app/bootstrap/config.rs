//! App bootstrap configuration loading and GPU startup settings.

use logger::LogLevel;

use crate::config::{
    ConfigFilePolicy, ConfigLoadResult, GraphicsApiSetting, VsyncMode, load_renderer_settings,
    log_config_resolve_trace,
};
use crate::ipc::get_ignore_config;

/// Fixed swapchain frame latency used for every GPU startup path.
pub(crate) const MAX_FRAME_LATENCY: u32 = 2;

/// Initial GPU/swapchain knobs read once during process bootstrap.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GpuStartupConfig {
    /// Initial vsync preference resolved against surface capabilities by `GpuContext`.
    pub(crate) vsync: VsyncMode,
    /// Initial maximum swapchain frame latency.
    pub(crate) max_frame_latency: u32,
    /// Whether to enable wgpu/Vulkan validation layers at startup.
    pub(crate) gpu_validation_layers: bool,
    /// Adapter ranking preference for desktop/headless GPU selection.
    pub(crate) power_preference: wgpu::PowerPreference,
    /// Startup graphics API preference for desktop/headless GPU selection.
    pub(crate) graphics_api: GraphicsApiSetting,
}

/// App configuration bundle consumed by bootstrap dispatch.
pub(crate) struct AppConfig {
    /// Full renderer config load result.
    pub(crate) load: ConfigLoadResult,
    /// Initial GPU settings distilled from the renderer config.
    pub(crate) gpu: GpuStartupConfig,
}

/// Chooses the process max log level after file logging is initialized.
pub(crate) fn effective_renderer_log_level(cli: Option<LogLevel>, log_verbose: bool) -> LogLevel {
    if let Some(level) = cli {
        level
    } else if log_verbose {
        LogLevel::Trace
    } else {
        LogLevel::Debug
    }
}

/// Loads renderer config, applies log verbosity, and extracts GPU startup settings.
pub(crate) fn load_app_config(log_level_cli: Option<LogLevel>) -> AppConfig {
    let config_file_policy = if get_ignore_config() {
        ConfigFilePolicy::Ignore
    } else {
        ConfigFilePolicy::Load
    };
    let load = load_renderer_settings(config_file_policy);
    logger::set_max_level(effective_renderer_log_level(
        log_level_cli,
        load.settings.debug.log_verbose,
    ));
    log_config_resolve_trace(&load.resolve);

    let gpu = GpuStartupConfig {
        vsync: load.settings.rendering.vsync,
        max_frame_latency: MAX_FRAME_LATENCY,
        gpu_validation_layers: load.settings.debug.gpu_validation_layers,
        power_preference: load.settings.debug.power_preference.to_wgpu(),
        graphics_api: load.settings.rendering.graphics_api,
    };
    log_startup_config_summary(&load, gpu, log_level_cli);

    AppConfig { load, gpu }
}

fn log_startup_config_summary(
    load: &ConfigLoadResult,
    gpu: GpuStartupConfig,
    log_level_cli: Option<LogLevel>,
) {
    let settings = &load.settings;
    logger::info!(
        "Renderer config summary: source={:?} loaded_path={} save_path={} suppress_disk_writes={} log_verbose={} cli_log_level_present={} vsync={:?} graphics_api={} gpu_validation={} power_preference={} msaa={} scene_color={:?} post_processing_enabled={} gtao={} bloom={} auto_exposure={} tonemap={} watchdog_enabled={}",
        load.resolve.source,
        load.resolve
            .loaded_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "<none>".to_string()),
        load.save_path.display(),
        load.suppress_config_disk_writes,
        settings.debug.log_verbose,
        log_level_cli.is_some(),
        gpu.vsync,
        gpu.graphics_api.as_persist_str(),
        gpu.gpu_validation_layers,
        settings.debug.power_preference.persist_str(),
        settings.rendering.msaa.persist_str(),
        settings.rendering.scene_color_format,
        settings.post_processing.enabled,
        settings.post_processing.gtao.enabled,
        settings.post_processing.bloom.enabled,
        settings.post_processing.auto_exposure.enabled,
        settings.post_processing.tonemap.mode.persist_str(),
        settings.watchdog.enabled,
    );
}

#[cfg(test)]
mod tests {
    use super::{
        GpuStartupConfig, MAX_FRAME_LATENCY, effective_renderer_log_level,
        log_startup_config_summary,
    };
    use crate::config::{ConfigLoadResult, ConfigResolveOutcome, ConfigSource, RendererSettings};
    use logger::LogLevel;
    use std::path::PathBuf;

    #[test]
    fn cli_always_overrides_log_verbose() {
        assert_eq!(
            effective_renderer_log_level(Some(LogLevel::Warn), true),
            LogLevel::Warn
        );
    }

    #[test]
    fn no_cli_uses_trace_when_log_verbose() {
        assert_eq!(effective_renderer_log_level(None, true), LogLevel::Trace);
    }

    #[test]
    fn no_cli_uses_debug_when_not_log_verbose() {
        assert_eq!(effective_renderer_log_level(None, false), LogLevel::Debug);
    }

    #[test]
    fn startup_config_summary_accepts_default_settings() {
        let load = ConfigLoadResult {
            settings: RendererSettings::default(),
            resolve: ConfigResolveOutcome {
                attempted_paths: Vec::new(),
                loaded_path: None,
                source: ConfigSource::None,
            },
            save_path: PathBuf::from("config.toml"),
            suppress_config_disk_writes: false,
        };
        let gpu = GpuStartupConfig {
            vsync: load.settings.rendering.vsync,
            max_frame_latency: MAX_FRAME_LATENCY,
            gpu_validation_layers: load.settings.debug.gpu_validation_layers,
            power_preference: load.settings.debug.power_preference.to_wgpu(),
            graphics_api: load.settings.rendering.graphics_api,
        };
        log_startup_config_summary(&load, gpu, None);
    }
}
