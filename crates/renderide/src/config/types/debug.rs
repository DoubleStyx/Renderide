//! Debug, diagnostics, and adapter-selection settings. Persisted as `[debug]`.

use serde::{Deserialize, Serialize};

use crate::labeled_enum;

labeled_enum! {
    /// Preferred GPU power mode for future adapter selection (stored; changing at runtime may
    /// require re-initialization).
    pub enum PowerPreferenceSetting: "GPU power preference" {
        default => HighPerformance;

        /// Maps to [`wgpu::PowerPreference::LowPower`].
        LowPower => {
            persist: "low_power",
            label: "Low power",
            aliases: ["low"],
        },
        /// Maps to [`wgpu::PowerPreference::HighPerformance`].
        HighPerformance => {
            persist: "high_performance",
            label: "High performance",
            aliases: ["high", "performance"],
        },
    }
}

impl PowerPreferenceSetting {
    /// Stable string for TOML / UI (`low_power` / `high_performance`). Historical alias for
    /// [`Self::persist_str`].
    pub fn as_persist_str(self) -> &'static str {
        self.persist_str()
    }

    /// Parses case-insensitive persisted or UI tokens. Historical alias for
    /// [`Self::parse_persist`].
    pub fn from_persist_str(s: &str) -> Option<Self> {
        Self::parse_persist(s)
    }

    /// Maps the persisted setting to the corresponding [`wgpu::PowerPreference`] used by adapter
    /// selection.
    pub fn to_wgpu(self) -> wgpu::PowerPreference {
        match self {
            Self::LowPower => wgpu::PowerPreference::LowPower,
            Self::HighPerformance => wgpu::PowerPreference::HighPerformance,
        }
    }
}

labeled_enum! {
    /// Last selected tab in the **Renderide debug** HUD window.
    pub enum DebugHudMainTab: "debug HUD main tab" {
        default => Stats;

        /// Frame, adapter, host, IPC, scene, resource, and graph summary.
        Stats => {
            persist: "stats",
            label: "Stats",
        },
        /// Host shader -> renderer pipeline route table.
        ShaderRoutes => {
            persist: "shader_routes",
            label: "Shader routes",
            aliases: ["shaders"],
        },
        /// Submitted draw rows and material render-state overrides.
        DrawState => {
            persist: "draw_state",
            label: "Draw state",
            aliases: ["draws"],
        },
        /// Full wgpu allocator report.
        GpuMemory => {
            persist: "gpu_memory",
            label: "GPU memory",
            aliases: ["memory"],
        },
        /// Per-pass GPU timing breakdown.
        GpuPasses => {
            persist: "gpu_passes",
            label: "GPU passes",
            aliases: ["passes"],
        },
    }
}

labeled_enum! {
    /// Last selected tab in the **Renderer config** HUD window.
    pub enum DebugHudRendererConfigTab: "renderer config HUD tab" {
        default => Display;

        /// Display caps and present-related controls.
        Display => {
            persist: "display",
            label: "Display",
        },
        /// Rendering and graph controls.
        Rendering => {
            persist: "rendering",
            label: "Rendering",
        },
        /// Debug and diagnostics controls.
        Debug => {
            persist: "debug",
            label: "Debug",
        },
        /// Post-processing effect controls.
        PostProcessing => {
            persist: "post_processing",
            label: "Post-Processing",
            aliases: ["post-processing", "post"],
        },
    }
}

/// Visibility of closable tabs in the **Renderide debug** HUD window.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DebugHudMainTabVisibility {
    /// Whether the **Stats** tab is open.
    pub stats: bool,
    /// Whether the **Shader routes** tab is open.
    pub shader_routes: bool,
    /// Whether the **Draw state** tab is open.
    pub draw_state: bool,
    /// Whether the **GPU memory** tab is open.
    pub gpu_memory: bool,
    /// Whether the **GPU passes** tab is open.
    pub gpu_passes: bool,
}

impl Default for DebugHudMainTabVisibility {
    fn default() -> Self {
        Self {
            stats: true,
            shader_routes: true,
            draw_state: true,
            gpu_memory: true,
            gpu_passes: true,
        }
    }
}

impl DebugHudMainTabVisibility {
    /// Returns whether `tab` is currently open.
    pub fn is_open(self, tab: DebugHudMainTab) -> bool {
        match tab {
            DebugHudMainTab::Stats => self.stats,
            DebugHudMainTab::ShaderRoutes => self.shader_routes,
            DebugHudMainTab::DrawState => self.draw_state,
            DebugHudMainTab::GpuMemory => self.gpu_memory,
            DebugHudMainTab::GpuPasses => self.gpu_passes,
        }
    }

    /// Updates whether `tab` is currently open.
    pub fn set_open(&mut self, tab: DebugHudMainTab, value: bool) {
        match tab {
            DebugHudMainTab::Stats => self.stats = value,
            DebugHudMainTab::ShaderRoutes => self.shader_routes = value,
            DebugHudMainTab::DrawState => self.draw_state = value,
            DebugHudMainTab::GpuMemory => self.gpu_memory = value,
            DebugHudMainTab::GpuPasses => self.gpu_passes = value,
        }
    }

    /// Returns `true` when every tab is open.
    pub fn all_open(self) -> bool {
        self.stats && self.shader_routes && self.draw_state && self.gpu_memory && self.gpu_passes
    }
}

/// Visibility of closable tabs in the **Renderer config** HUD window.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DebugHudRendererConfigTabVisibility {
    /// Whether the **Display** tab is open.
    pub display: bool,
    /// Whether the **Rendering** tab is open.
    pub rendering: bool,
    /// Whether the **Debug** tab is open.
    pub debug: bool,
    /// Whether the **Post-Processing** tab is open.
    pub post_processing: bool,
}

impl Default for DebugHudRendererConfigTabVisibility {
    fn default() -> Self {
        Self {
            display: true,
            rendering: true,
            debug: true,
            post_processing: true,
        }
    }
}

impl DebugHudRendererConfigTabVisibility {
    /// Returns whether `tab` is currently open.
    pub fn is_open(self, tab: DebugHudRendererConfigTab) -> bool {
        match tab {
            DebugHudRendererConfigTab::Display => self.display,
            DebugHudRendererConfigTab::Rendering => self.rendering,
            DebugHudRendererConfigTab::Debug => self.debug,
            DebugHudRendererConfigTab::PostProcessing => self.post_processing,
        }
    }

    /// Updates whether `tab` is currently open.
    pub fn set_open(&mut self, tab: DebugHudRendererConfigTab, value: bool) {
        match tab {
            DebugHudRendererConfigTab::Display => self.display = value,
            DebugHudRendererConfigTab::Rendering => self.rendering = value,
            DebugHudRendererConfigTab::Debug => self.debug = value,
            DebugHudRendererConfigTab::PostProcessing => self.post_processing = value,
        }
    }

    /// Returns `true` when every tab is open.
    pub fn all_open(self) -> bool {
        self.display && self.rendering && self.debug && self.post_processing
    }
}

/// Persisted semantic state for the Dear ImGui diagnostics HUD.
///
/// ImGui-owned window placement and collapse data lives in the sidecar `.ini` file; this struct
/// keeps renderer-owned UI preferences in `config.toml` so they share the existing config save
/// path and write-suppression rules.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DebugHudSettings {
    /// Whether the renderer should load/save ImGui's raw `.ini` layout sidecar.
    pub persist_layout: bool,
    /// Global HUD text scale. Clamped at use sites by [`Self::resolved_ui_scale`].
    pub ui_scale: f32,
    /// Whether the **Renderer config** window is open.
    pub renderer_config_open: bool,
    /// Whether the **Scene transforms** window is open.
    pub scene_transforms_open: bool,
    /// Whether the **Textures** window is open.
    pub texture_debug_open: bool,
    /// Show only textures referenced by the current view in the **Textures** window.
    pub texture_debug_current_view_only: bool,
    /// Show only overlay/UI-ish draws in the **Draw state** tab.
    pub draw_state_ui_only: bool,
    /// Show only material rows with render-state overrides in the **Draw state** tab.
    pub draw_state_only_overrides: bool,
    /// Show only fallback shader routes in the **Shader routes** tab.
    pub shader_routes_only_fallback: bool,
    /// Last selected tab in **Renderide debug**.
    pub main_tab: DebugHudMainTab,
    /// Open/closed state for tabs in **Renderide debug**.
    pub main_tabs: DebugHudMainTabVisibility,
    /// Last selected tab in **Renderer config**.
    pub renderer_config_tab: DebugHudRendererConfigTab,
    /// Open/closed state for tabs in **Renderer config**.
    pub renderer_config_tabs: DebugHudRendererConfigTabVisibility,
    /// Last selected render-space tab in **Scene transforms**.
    pub scene_transforms_space_id: Option<i32>,
}

impl Default for DebugHudSettings {
    fn default() -> Self {
        Self {
            persist_layout: true,
            ui_scale: Self::DEFAULT_UI_SCALE,
            renderer_config_open: true,
            scene_transforms_open: true,
            texture_debug_open: true,
            texture_debug_current_view_only: false,
            draw_state_ui_only: false,
            draw_state_only_overrides: false,
            shader_routes_only_fallback: false,
            main_tab: DebugHudMainTab::default(),
            main_tabs: DebugHudMainTabVisibility::default(),
            renderer_config_tab: DebugHudRendererConfigTab::default(),
            renderer_config_tabs: DebugHudRendererConfigTabVisibility::default(),
            scene_transforms_space_id: None,
        }
    }
}

impl DebugHudSettings {
    /// Smallest accepted global HUD scale.
    pub const MIN_UI_SCALE: f32 = 0.5;
    /// Largest accepted global HUD scale.
    pub const MAX_UI_SCALE: f32 = 2.0;
    /// Default global HUD scale.
    pub const DEFAULT_UI_SCALE: f32 = 1.0;

    /// Returns a finite HUD scale clamped into the supported range.
    pub fn resolved_ui_scale(&self) -> f32 {
        if self.ui_scale.is_finite() {
            self.ui_scale.clamp(Self::MIN_UI_SCALE, Self::MAX_UI_SCALE)
        } else {
            Self::DEFAULT_UI_SCALE
        }
    }
}

/// Debug and diagnostics flags. Persisted as `[debug]`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DebugSettings {
    /// When the `-LogLevel` CLI argument is **not** present, selects [`logger::LogLevel::Trace`]
    /// if true or [`logger::LogLevel::Debug`] if false. If `-LogLevel` is present, it always
    /// overrides this flag.
    pub log_verbose: bool,
    /// GPU power preference hint for adapter selection (see [`PowerPreferenceSetting`]).
    pub power_preference: PowerPreferenceSetting,
    /// When true, request backend validation (e.g. Vulkan validation layers) via wgpu instance
    /// flags. Significantly slows rendering; use only when debugging GPU API misuse. Default
    /// false. Applies to both desktop wgpu init and the OpenXR Vulkan / wgpu-hal bootstrap.
    /// Native **stdout** and **stderr** are forwarded to the renderer log file after logging
    /// starts (see [`crate::app::run`]), so layer and spirv-val output is captured regardless of
    /// this flag. Applied when the GPU stack is first created, not on later config updates.
    /// [`crate::config::apply_renderide_gpu_validation_env`] and `WGPU_*` environment variables
    /// can still adjust flags at process start.
    pub gpu_validation_layers: bool,
    /// When true, show the **Frame timing** ImGui window (FPS and CPU/GPU submit-interval
    /// metrics). Cheap snapshot; independent of [`Self::debug_hud_enabled`]. Default true.
    #[serde(default = "default_debug_hud_frame_timing")]
    pub debug_hud_frame_timing: bool,
    /// When true, show **Renderide debug** (Stats / Shader routes) and run mesh-draw stats,
    /// frame diagnostics, and renderer info capture. Default false (performance-first; **Renderer
    /// config** or `debug_hud_enabled` in config).
    pub debug_hud_enabled: bool,
    /// When true, capture [`crate::diagnostics::SceneTransformsSnapshot`] each frame and show
    /// the **Scene transforms** ImGui window (can be expensive on large scenes). Independent of
    /// [`Self::debug_hud_enabled`] so you can enable transforms inspection without the main
    /// debug panels. Default false.
    pub debug_hud_transforms: bool,
    /// When true, show the **Textures** ImGui window listing GPU texture pool entries with
    /// format, resident/total mips, filter mode, wrap, aniso, and color profile. Useful for
    /// diagnosing mip / sampler issues. Default false.
    #[serde(default)]
    pub debug_hud_textures: bool,
    /// Semantic ImGui HUD state persisted through the renderer config.
    pub hud: DebugHudSettings,
}

impl Default for DebugSettings {
    fn default() -> Self {
        Self {
            log_verbose: false,
            power_preference: PowerPreferenceSetting::default(),
            gpu_validation_layers: false,
            debug_hud_frame_timing: true,
            debug_hud_enabled: false,
            debug_hud_transforms: false,
            debug_hud_textures: false,
            hud: DebugHudSettings::default(),
        }
    }
}

fn default_debug_hud_frame_timing() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::{
        DebugHudMainTab, DebugHudMainTabVisibility, DebugHudRendererConfigTab,
        DebugHudRendererConfigTabVisibility, DebugHudSettings, PowerPreferenceSetting,
    };
    use crate::config::RendererSettings;

    #[test]
    fn power_preference_from_persist_str() {
        assert_eq!(
            PowerPreferenceSetting::from_persist_str("low_power"),
            Some(PowerPreferenceSetting::LowPower)
        );
        assert_eq!(
            PowerPreferenceSetting::from_persist_str("LOW"),
            Some(PowerPreferenceSetting::LowPower)
        );
        assert_eq!(
            PowerPreferenceSetting::from_persist_str("performance"),
            Some(PowerPreferenceSetting::HighPerformance)
        );
        assert_eq!(
            PowerPreferenceSetting::from_persist_str("high_performance"),
            Some(PowerPreferenceSetting::HighPerformance)
        );
        assert_eq!(PowerPreferenceSetting::from_persist_str(""), None);
    }

    #[test]
    fn missing_hud_table_uses_defaults() {
        let s: RendererSettings = toml::from_str(
            r#"
            [debug]
            debug_hud_enabled = true
            "#,
        )
        .expect("old config without debug.hud should load");

        assert_eq!(s.debug.hud, DebugHudSettings::default());
        assert!(s.debug.debug_hud_enabled);
    }

    #[test]
    fn hud_tab_tokens_roundtrip() {
        let mut s = RendererSettings::default();
        s.debug.hud.main_tab = DebugHudMainTab::GpuPasses;
        s.debug.hud.renderer_config_tab = DebugHudRendererConfigTab::PostProcessing;

        let text = toml::to_string(&s).expect("serialize");
        assert!(text.contains("main_tab = \"gpu_passes\""));
        assert!(text.contains("renderer_config_tab = \"post_processing\""));

        let decoded: RendererSettings = toml::from_str(&text).expect("deserialize");
        assert_eq!(decoded.debug.hud.main_tab, DebugHudMainTab::GpuPasses);
        assert_eq!(
            decoded.debug.hud.renderer_config_tab,
            DebugHudRendererConfigTab::PostProcessing
        );
    }

    #[test]
    fn hud_ui_scale_resolves_to_supported_range() {
        let mut s = DebugHudSettings {
            ui_scale: 0.1,
            ..Default::default()
        };
        assert_eq!(s.resolved_ui_scale(), DebugHudSettings::MIN_UI_SCALE);

        s.ui_scale = 99.0;
        assert_eq!(s.resolved_ui_scale(), DebugHudSettings::MAX_UI_SCALE);

        s.ui_scale = f32::NAN;
        assert_eq!(s.resolved_ui_scale(), DebugHudSettings::DEFAULT_UI_SCALE);
    }

    #[test]
    fn hud_tab_visibility_defaults_open_and_maps_tabs() {
        let mut main = DebugHudMainTabVisibility::default();
        assert!(main.all_open());
        main.set_open(DebugHudMainTab::DrawState, false);
        assert!(!main.is_open(DebugHudMainTab::DrawState));
        assert!(!main.all_open());

        let mut config = DebugHudRendererConfigTabVisibility::default();
        assert!(config.all_open());
        config.set_open(DebugHudRendererConfigTab::Debug, false);
        assert!(!config.is_open(DebugHudRendererConfigTab::Debug));
        assert!(!config.all_open());
    }
}
