//! Process-wide renderer settings merged from defaults and optional `config.ini`.

use super::parse::{parse_ini_document, IniDocument, ParseWarning};
use super::resolve::{read_config_file, resolve_config_path, ConfigResolveOutcome, ConfigSource};

/// Runtime settings for the renderer process. Start with [`Default`]; keys from INI will be merged
/// in [`Self::merge_from_ini`] as the surface grows.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RendererSettings {}

impl RendererSettings {
    /// Hardcoded defaults only.
    pub fn from_defaults() -> Self {
        Self::default()
    }

    /// Applies recognized INI keys. Currently a no-op; future options override defaults here.
    pub fn merge_from_ini(&mut self, _document: &IniDocument) {}
}

/// Full load result: resolved path, parsed document, and parse warnings.
#[derive(Clone, Debug)]
pub struct ConfigLoadResult {
    pub settings: RendererSettings,
    pub resolve: ConfigResolveOutcome,
    pub document: IniDocument,
    pub parse_warnings: Vec<ParseWarning>,
}

/// Resolves `config.ini`, parses it, and builds [`RendererSettings`].
pub fn load_renderer_settings() -> ConfigLoadResult {
    let resolve = resolve_config_path();
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

    ConfigLoadResult {
        settings,
        resolve,
        document,
        parse_warnings,
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
