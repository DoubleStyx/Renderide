//! Process-wide renderer settings merged from defaults and optional `config.ini`.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::parse::{parse_ini_document, IniDocument, ParseWarning};
use super::resolve::{
    apply_generated_config, is_dir_writable, read_config_file, renderide_config_env_nonempty,
    resolve_config_path, resolve_save_path, ConfigResolveOutcome, ConfigSource,
};

/// Display-related caps (future: frame pacing when unfocused). Persisted as `[display]`.
#[derive(Clone, Debug, PartialEq)]
pub struct DisplaySettings {
    /// Target max FPS when the window is focused (0 = uncapped / engine default).
    pub focused_fps_cap: u32,
    /// Target max FPS when unfocused (0 = uncapped / engine default).
    pub unfocused_fps_cap: u32,
}

impl Default for DisplaySettings {
    fn default() -> Self {
        Self {
            focused_fps_cap: 240,
            unfocused_fps_cap: 60,
        }
    }
}

/// Rendering toggles and scalars. Persisted as `[rendering]`.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderingSettings {
    /// Vertical sync via swapchain present mode ([`wgpu::PresentMode::AutoVsync`]).
    pub vsync: bool,
    /// Exposure multiplier for future HDR/tonemap path (stored; not yet applied to passes).
    pub exposure: f32,
}

impl Default for RenderingSettings {
    fn default() -> Self {
        Self {
            vsync: false,
            exposure: 1.0,
        }
    }
}

/// Preferred GPU power mode for future adapter selection (stored; changing at runtime may require
/// re-initialization).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PowerPreferenceSetting {
    /// Maps to [`wgpu::PowerPreference::LowPower`].
    LowPower,
    /// Maps to [`wgpu::PowerPreference::HighPerformance`].
    #[default]
    HighPerformance,
}

impl PowerPreferenceSetting {
    /// All variants for ImGui combo / persistence.
    pub const ALL: [Self; 2] = [Self::LowPower, Self::HighPerformance];

    /// INI token (lowercase).
    pub fn as_ini_str(self) -> &'static str {
        match self {
            Self::LowPower => "low_power",
            Self::HighPerformance => "high_performance",
        }
    }

    /// Parses case-insensitive INI value.
    pub fn from_ini_str(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "low_power" | "low" => Some(Self::LowPower),
            "high_performance" | "high" | "performance" => Some(Self::HighPerformance),
            _ => None,
        }
    }

    /// Label for developer UI.
    pub fn label(self) -> &'static str {
        match self {
            Self::LowPower => "Low power",
            Self::HighPerformance => "High performance",
        }
    }
}

/// Debug and diagnostics flags. Persisted as `[debug]`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DebugSettings {
    /// Request verbose logging (future hook into logger; stored for now).
    pub log_verbose: bool,
    /// GPU power preference hint for adapter selection (see [`PowerPreferenceSetting`]).
    pub power_preference: PowerPreferenceSetting,
}

/// Runtime settings for the renderer process: defaults, merged from INI, and edited via the debug UI.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RendererSettings {
    /// Display caps and related options.
    pub display: DisplaySettings,
    /// Rendering options (vsync, exposure, …).
    pub rendering: RenderingSettings,
    /// Debug-only flags.
    pub debug: DebugSettings,
}

impl RendererSettings {
    /// Hardcoded defaults only.
    pub fn from_defaults() -> Self {
        Self::default()
    }

    /// Applies recognized INI keys over current values.
    pub fn merge_from_ini(&mut self, document: &IniDocument) {
        if let Some(s) = document.get("display", "focused_fps") {
            if let Some(v) = parse_u32(s) {
                self.display.focused_fps_cap = v;
            }
        }
        if let Some(s) = document.get("display", "unfocused_fps") {
            if let Some(v) = parse_u32(s) {
                self.display.unfocused_fps_cap = v;
            }
        }
        if let Some(s) = document.get("rendering", "vsync") {
            if let Some(v) = parse_bool(s) {
                self.rendering.vsync = v;
            }
        }
        if let Some(s) = document.get("rendering", "exposure") {
            if let Some(v) = parse_f32(s) {
                self.rendering.exposure = v;
            }
        }
        if let Some(s) = document.get("debug", "log_verbose") {
            if let Some(v) = parse_bool(s) {
                self.debug.log_verbose = v;
            }
        }
        if let Some(s) = document.get("debug", "power_preference") {
            if let Some(v) = PowerPreferenceSetting::from_ini_str(s) {
                self.debug.power_preference = v;
            }
        }
    }

    /// Builds an [`IniDocument`] representing the full current settings (for save / round-trip).
    pub fn to_ini_document(&self) -> IniDocument {
        let mut doc = IniDocument::default();
        doc.set(
            "display",
            "focused_fps",
            self.display.focused_fps_cap.to_string(),
        );
        doc.set(
            "display",
            "unfocused_fps",
            self.display.unfocused_fps_cap.to_string(),
        );
        doc.set("rendering", "vsync", bool_ini(self.rendering.vsync));
        doc.set(
            "rendering",
            "exposure",
            format_float_trim(self.rendering.exposure),
        );
        doc.set("debug", "log_verbose", bool_ini(self.debug.log_verbose));
        doc.set(
            "debug",
            "power_preference",
            self.debug.power_preference.as_ini_str(),
        );
        doc
    }
}

fn bool_ini(b: bool) -> String {
    if b {
        "true".to_string()
    } else {
        "false".to_string()
    }
}

fn format_float_trim(x: f32) -> String {
    if x.fract() == 0.0 {
        format!("{}", x as i64)
    } else {
        format!("{x:.6}")
    }
}

fn parse_bool(s: &str) -> Option<bool> {
    match s.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn parse_u32(s: &str) -> Option<u32> {
    s.trim().parse().ok()
}

fn parse_f32(s: &str) -> Option<f32> {
    s.trim().parse().ok()
}

/// Full load result: resolved path, parsed document, parse warnings, and save path for persistence.
#[derive(Clone, Debug)]
pub struct ConfigLoadResult {
    /// Effective settings after merge.
    pub settings: RendererSettings,
    /// Path resolution diagnostics.
    pub resolve: ConfigResolveOutcome,
    /// Parsed document from disk when a file was read (may be empty).
    pub document: IniDocument,
    /// Non-fatal parse issues.
    pub parse_warnings: Vec<ParseWarning>,
    /// Target file for [`save_renderer_settings`] and the ImGui config window.
    pub save_path: PathBuf,
}

/// Shared handle for the process-wide settings store (read by the frame loop, written by the HUD).
pub type RendererSettingsHandle = Arc<std::sync::RwLock<RendererSettings>>;

/// Resolves `config.ini`, parses it, and builds [`RendererSettings`].
///
/// When no file exists and [`super::resolve::renderide_config_env_nonempty`] is false, writes
/// defaults to the save path (see [`super::resolve::resolve_save_path`]) and loads that file.
pub fn load_renderer_settings() -> ConfigLoadResult {
    let mut resolve = resolve_config_path();
    let mut settings = RendererSettings::from_defaults();
    let mut document = IniDocument::default();
    let mut parse_warnings = Vec::new();

    match resolve.loaded_path.as_ref() {
        Some(path) => {
            logger::info!("Loading renderer config from {}", path.display());
            match read_config_file(path) {
                Ok(content) => {
                    let (doc, warnings) = parse_ini_document(&content);
                    parse_warnings = warnings;
                    settings.merge_from_ini(&doc);
                    document = doc;
                    if !parse_warnings.is_empty() {
                        for w in &parse_warnings {
                            logger::debug!(
                                "config.ini parse warning line {}: {}",
                                w.line,
                                w.message
                            );
                        }
                    }
                }
                Err(e) => {
                    logger::warn!("Failed to read {}: {e}; using defaults", path.display());
                }
            }
        }
        None => {
            logger::info!("config.ini not found; using built-in defaults");
            logger::trace!(
                "config search tried {} path(s)",
                resolve.attempted_paths.len()
            );
        }
    }

    if resolve.loaded_path.is_none() && !renderide_config_env_nonempty() {
        let path = resolve_save_path(&resolve);
        if !path.exists() {
            if let Some(parent) = path.parent() {
                if is_dir_writable(parent) {
                    match save_renderer_settings(&path, &RendererSettings::from_defaults()) {
                        Ok(()) => {
                            logger::info!("Created default renderer config at {}", path.display());
                            apply_generated_config(&mut resolve, path.clone());
                            match read_config_file(&path) {
                                Ok(content) => {
                                    let (doc, warnings) = parse_ini_document(&content);
                                    parse_warnings = warnings;
                                    settings.merge_from_ini(&doc);
                                    document = doc;
                                    if !parse_warnings.is_empty() {
                                        for w in &parse_warnings {
                                            logger::debug!(
                                                "config.ini parse warning line {}: {}",
                                                w.line,
                                                w.message
                                            );
                                        }
                                    }
                                }
                                Err(e) => {
                                    logger::warn!(
                                        "Failed to read newly created {}: {e}; using defaults",
                                        path.display()
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            logger::warn!(
                                "Failed to create default config at {}: {e}",
                                path.display()
                            );
                        }
                    }
                } else {
                    logger::trace!(
                        "Not creating default config at {} (directory not writable)",
                        path.display()
                    );
                }
            }
        }
    }

    let save_path = resolve_save_path(&resolve);

    logger::trace!("Renderer config will persist to {}", save_path.display());

    ConfigLoadResult {
        settings,
        resolve,
        document,
        parse_warnings,
        save_path,
    }
}

/// Writes `settings` to `path` atomically (temp file in the same directory, then rename).
pub fn save_renderer_settings(path: &Path, settings: &RendererSettings) -> io::Result<()> {
    let contents = settings.to_ini_document().serialize();

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("config.ini");
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let tmp = parent.join(format!(".{file_name}.tmp"));
    std::fs::write(&tmp, contents.as_bytes())?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Persists using [`ConfigLoadResult::save_path`] and logs failures.
pub fn save_renderer_settings_from_load(load: &ConfigLoadResult, settings: &RendererSettings) {
    if let Err(e) = save_renderer_settings(&load.save_path, settings) {
        logger::warn!(
            "Failed to save renderer config to {}: {e}",
            load.save_path.display()
        );
    } else {
        logger::trace!("Saved renderer config to {}", load.save_path.display());
    }
}

/// Builds a [`RendererSettingsHandle`] from post-load settings.
pub fn settings_handle_from(load: &ConfigLoadResult) -> RendererSettingsHandle {
    Arc::new(std::sync::RwLock::new(load.settings.clone()))
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

    #[test]
    fn merge_roundtrip_document() {
        let mut s = RendererSettings::from_defaults();
        s.rendering.vsync = true;
        s.display.focused_fps_cap = 120;
        let doc = s.to_ini_document();
        let mut s2 = RendererSettings::from_defaults();
        s2.merge_from_ini(&doc);
        assert_eq!(s, s2);
    }

    #[test]
    fn atomic_save_roundtrip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.ini");
        let s = RendererSettings::from_defaults();
        save_renderer_settings(&path, &s).expect("save");
        let text = std::fs::read_to_string(&path).expect("read");
        let (doc, w) = parse_ini_document(&text);
        assert!(w.is_empty());
        let mut s2 = RendererSettings::from_defaults();
        s2.merge_from_ini(&doc);
        assert_eq!(s, s2);
    }

    #[test]
    fn save_path_prefers_loaded() {
        let resolve = ConfigResolveOutcome {
            attempted_paths: vec![],
            loaded_path: Some(PathBuf::from("/tmp/x/config.ini")),
            source: ConfigSource::Search,
        };
        assert_eq!(
            resolve_save_path(&resolve),
            PathBuf::from("/tmp/x/config.ini")
        );
    }
}
