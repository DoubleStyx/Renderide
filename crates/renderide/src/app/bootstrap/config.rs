//! App bootstrap configuration loading and GPU startup settings.

use logger::LogLevel;

use crate::config::{
    ConfigFilePolicy, ConfigLoadResult, GraphicsApiSetting, PresentationModeSetting,
    load_renderer_settings, log_config_resolve_trace,
};
use crate::ipc::get_ignore_config;

/// Fixed swapchain frame latency used for every GPU startup path.
pub(crate) const MAX_FRAME_LATENCY: u32 = 2;

/// Initial GPU/swapchain knobs read once during process bootstrap.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GpuStartupConfig {
    /// Initial presentation mode resolved against surface capabilities by `GpuContext`.
    pub(crate) presentation_mode: PresentationModeSetting,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RendererLogLevelResolution {
    level: LogLevel,
    source: RendererLogLevelSource,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RendererLogLevelSource {
    Cli,
    LogVerbose,
    Default,
}

impl RendererLogLevelSource {
    fn as_str(self) -> &'static str {
        match self {
            Self::Cli => "cli",
            Self::LogVerbose => "log_verbose",
            Self::Default => "default",
        }
    }
}

/// Chooses the process max log level after file logging is initialized.
pub(crate) fn effective_renderer_log_level(cli: Option<LogLevel>, log_verbose: bool) -> LogLevel {
    resolve_renderer_log_level(cli, log_verbose).level
}

fn resolve_renderer_log_level(
    cli: Option<LogLevel>,
    log_verbose: bool,
) -> RendererLogLevelResolution {
    if let Some(level) = cli {
        RendererLogLevelResolution {
            level,
            source: RendererLogLevelSource::Cli,
        }
    } else if log_verbose {
        RendererLogLevelResolution {
            level: LogLevel::Trace,
            source: RendererLogLevelSource::LogVerbose,
        }
    } else {
        RendererLogLevelResolution {
            level: LogLevel::Debug,
            source: RendererLogLevelSource::Default,
        }
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
    let resolved_log_level =
        resolve_renderer_log_level(log_level_cli, load.settings.debug.log_verbose);
    logger::set_max_level(resolved_log_level.level);
    log_config_resolve_trace(&load.resolve);

    let gpu = GpuStartupConfig {
        presentation_mode: load.settings.rendering.presentation_mode,
        max_frame_latency: MAX_FRAME_LATENCY,
        gpu_validation_layers: load.settings.debug.gpu_validation_layers,
        power_preference: load.settings.debug.power_preference.to_wgpu(),
        graphics_api: load.settings.rendering.graphics_api,
    };
    log_startup_config_summary(&load, gpu, resolved_log_level);

    AppConfig { load, gpu }
}

fn log_startup_config_summary(
    load: &ConfigLoadResult,
    gpu: GpuStartupConfig,
    log_level: RendererLogLevelResolution,
) {
    let settings = &load.settings;
    logger::info!(
        "Renderer config summary: source={:?} loaded_path={} save_path={} suppress_disk_writes={} log_verbose={} log_level={:?} log_level_source={} presentation_mode={} graphics_api={} gpu_validation={} power_preference={} msaa={} scene_color={:?} post_processing_enabled={} gtao={} bloom={} motion_blur={} auto_exposure={} tonemap={} watchdog_enabled={}",
        load.resolve.source,
        load.resolve
            .loaded_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "<none>".to_string()),
        load.save_path.display(),
        load.suppress_config_disk_writes,
        settings.debug.log_verbose,
        log_level.level,
        log_level.source.as_str(),
        gpu.presentation_mode.persist_str(),
        gpu.graphics_api.as_persist_str(),
        gpu.gpu_validation_layers,
        settings.debug.power_preference.persist_str(),
        settings.rendering.msaa.persist_str(),
        settings.rendering.scene_color_format,
        settings.post_processing.enabled,
        settings.post_processing.gtao.enabled,
        settings.post_processing.bloom.enabled,
        settings.post_processing.motion_blur.enabled,
        settings.post_processing.auto_exposure.enabled,
        settings.post_processing.tonemap.mode.persist_str(),
        settings.watchdog.enabled,
    );
}

#[cfg(test)]
mod tests {
    use super::{
        GpuStartupConfig, MAX_FRAME_LATENCY, RendererLogLevelSource, effective_renderer_log_level,
        log_startup_config_summary, resolve_renderer_log_level,
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
        let resolution = resolve_renderer_log_level(Some(LogLevel::Warn), true);
        assert_eq!(resolution.level, LogLevel::Warn);
        assert_eq!(resolution.source, RendererLogLevelSource::Cli);
        assert_eq!(resolution.source.as_str(), "cli");
    }

    #[test]
    fn no_cli_uses_trace_when_log_verbose() {
        assert_eq!(effective_renderer_log_level(None, true), LogLevel::Trace);
        let resolution = resolve_renderer_log_level(None, true);
        assert_eq!(resolution.level, LogLevel::Trace);
        assert_eq!(resolution.source, RendererLogLevelSource::LogVerbose);
        assert_eq!(resolution.source.as_str(), "log_verbose");
    }

    #[test]
    fn no_cli_uses_debug_when_not_log_verbose() {
        assert_eq!(effective_renderer_log_level(None, false), LogLevel::Debug);
        let resolution = resolve_renderer_log_level(None, false);
        assert_eq!(resolution.level, LogLevel::Debug);
        assert_eq!(resolution.source, RendererLogLevelSource::Default);
        assert_eq!(resolution.source.as_str(), "default");
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
            presentation_mode: load.settings.rendering.presentation_mode,
            max_frame_latency: MAX_FRAME_LATENCY,
            gpu_validation_layers: load.settings.debug.gpu_validation_layers,
            power_preference: load.settings.debug.power_preference.to_wgpu(),
            graphics_api: load.settings.rendering.graphics_api,
        };
        log_startup_config_summary(&load, gpu, resolve_renderer_log_level(None, false));
    }
}
