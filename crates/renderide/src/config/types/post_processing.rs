//! Post-processing stack configuration. Persisted as `[post_processing]` (with sub-tables per effect).

use serde::{Deserialize, Serialize};

mod bloom;
mod gtao;
mod tonemap;

pub use bloom::{BloomCompositeMode, BloomSettings};
pub use gtao::GtaoSettings;
pub use tonemap::{TonemapMode, TonemapSettings};

/// Post-processing stack configuration. Persisted as `[post_processing]` (with sub-tables per effect).
///
/// Effects are organised as nested sub-structs (`tonemap`, future `bloom`, `color_grading`, etc.)
/// so each gets its own TOML sub-table (`[post_processing.tonemap]`, ...) and so the
/// [`crate::render_graph::post_processing::PostProcessChainSignature`] can be derived purely from
/// this value.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PostProcessingSettings {
    /// Master enable for the entire stack. When `false`, the render graph skips the chain entirely
    /// and `SceneColorComposePass` samples the raw forward HDR target.
    pub enabled: bool,
    /// Ground-Truth Ambient Occlusion (pre-tonemap HDR modulation). See [`GtaoSettings`].
    pub gtao: GtaoSettings,
    /// Dual-filter physically-based bloom (pre-tonemap HDR). See [`BloomSettings`].
    pub bloom: BloomSettings,
    /// Tonemapping (HDR -> display-referred 0..1 linear). See [`TonemapSettings`].
    pub tonemap: TonemapSettings,
}

impl Default for PostProcessingSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            gtao: GtaoSettings::default(),
            bloom: BloomSettings::default(),
            tonemap: TonemapSettings::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{PostProcessingSettings, TonemapMode};
    use crate::config::types::RendererSettings;

    #[test]
    fn defaults_enable_stack_with_aces_selected() {
        let s = PostProcessingSettings::default();
        assert!(s.enabled, "post-processing should default to enabled");
        assert_eq!(s.tonemap.mode, TonemapMode::AcesFitted);
    }

    #[test]
    fn renderer_settings_includes_post_processing_section() {
        let s = RendererSettings::default();
        assert_eq!(s.post_processing, PostProcessingSettings::default());
    }

    #[test]
    fn tonemap_mode_label_is_stable() {
        for mode in TonemapMode::ALL {
            assert!(!mode.label().is_empty());
        }
    }

    #[test]
    fn post_processing_toml_roundtrip_emits_expected_sections() {
        let mut s = RendererSettings::default();
        s.post_processing.enabled = true;
        s.post_processing.tonemap.mode = TonemapMode::AcesFitted;
        let toml = toml::to_string(&s).expect("serialize");
        assert!(
            toml.contains("[post_processing]"),
            "expected `[post_processing]` table, got:\n{toml}"
        );
        assert!(
            toml.contains("[post_processing.tonemap]"),
            "expected `[post_processing.tonemap]` sub-table, got:\n{toml}"
        );
        assert!(
            toml.contains("mode = \"aces_fitted\""),
            "expected snake_case mode value, got:\n{toml}"
        );
        let back: RendererSettings = toml::from_str(&toml).expect("deserialize");
        assert!(back.post_processing.enabled);
        assert_eq!(back.post_processing.tonemap.mode, TonemapMode::AcesFitted);
    }

    #[test]
    fn post_processing_toml_roundtrip_disabled_with_none_mode() {
        let mut s = RendererSettings::default();
        s.post_processing.enabled = false;
        s.post_processing.tonemap.mode = TonemapMode::None;
        let toml = toml::to_string(&s).expect("serialize");
        assert!(toml.contains("mode = \"none\""), "got:\n{toml}");
        let back: RendererSettings = toml::from_str(&toml).expect("deserialize");
        assert!(!back.post_processing.enabled);
        assert_eq!(back.post_processing.tonemap.mode, TonemapMode::None);
    }

    #[test]
    fn missing_post_processing_section_yields_defaults() {
        let toml = "\n[display]\nfocused_fps = 60\nunfocused_fps = 30\n";
        let s: RendererSettings = toml::from_str(toml).expect("deserialize");
        assert_eq!(s.post_processing, PostProcessingSettings::default());
    }
}
