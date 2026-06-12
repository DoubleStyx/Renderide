//! Persisted HUD state: which **Renderide debug** and **Renderer config** tabs the user last
//! selected, which tabs are open, and global ImGui presentation flags.

use serde::{Deserialize, Serialize};

use crate::labeled_enum;

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
        /// Experimental renderer feature flags.
        Experimental => {
            persist: "experimental",
            label: "Experimental",
            aliases: ["experiments"],
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

    /// Returns `true` when at least one main debug tab is open.
    pub fn any_open(self) -> bool {
        DebugHudMainTab::ALL
            .iter()
            .copied()
            .any(|tab| self.is_open(tab))
    }

    /// Returns the first open main debug tab in stable display order.
    pub fn first_open(self) -> Option<DebugHudMainTab> {
        DebugHudMainTab::ALL
            .iter()
            .copied()
            .find(|&tab| self.is_open(tab))
    }

    /// Returns the selected tab when it is open, otherwise the first open tab.
    pub fn effective_tab(self, selected: DebugHudMainTab) -> Option<DebugHudMainTab> {
        if !self.any_open() {
            return None;
        }
        self.is_open(selected)
            .then_some(selected)
            .or_else(|| self.first_open())
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
        DebugHudMainTab::ALL
            .iter()
            .copied()
            .all(|tab| self.is_open(tab))
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
    /// Whether the **Experimental** tab is open.
    pub experimental: bool,
}

impl Default for DebugHudRendererConfigTabVisibility {
    fn default() -> Self {
        Self {
            display: true,
            rendering: true,
            debug: true,
            post_processing: true,
            experimental: true,
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
            DebugHudRendererConfigTab::Experimental => self.experimental,
        }
    }

    /// Updates whether `tab` is currently open.
    pub fn set_open(&mut self, tab: DebugHudRendererConfigTab, value: bool) {
        match tab {
            DebugHudRendererConfigTab::Display => self.display = value,
            DebugHudRendererConfigTab::Rendering => self.rendering = value,
            DebugHudRendererConfigTab::Debug => self.debug = value,
            DebugHudRendererConfigTab::PostProcessing => self.post_processing = value,
            DebugHudRendererConfigTab::Experimental => self.experimental = value,
        }
    }

    /// Returns `true` when every tab is open.
    pub fn all_open(self) -> bool {
        DebugHudRendererConfigTab::ALL
            .iter()
            .copied()
            .all(|tab| self.is_open(tab))
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
    /// Whether the renderer ImGui overlay is visible.
    pub imgui_visible: bool,
    /// Global HUD text scale. Clamped at use sites by [`Self::resolved_ui_scale`].
    pub ui_scale: f32,
    /// Retained for existing config files; **Renderer config** now stays visible whenever ImGui is visible.
    pub renderer_config_open: bool,
    /// Retained for existing config files; **Scene transforms** visibility now uses `debug_hud_transforms`.
    pub scene_transforms_open: bool,
    /// Retained for existing config files; **Textures** visibility now uses `debug_hud_textures`.
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
            imgui_visible: true,
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

#[cfg(test)]
mod tests {
    use super::{
        DebugHudMainTab, DebugHudMainTabVisibility, DebugHudRendererConfigTab,
        DebugHudRendererConfigTabVisibility, DebugHudSettings,
    };
    use crate::config::RendererSettings;

    #[test]
    fn hud_tab_tokens_roundtrip() {
        let mut s = RendererSettings::default();
        s.debug.hud.main_tab = DebugHudMainTab::GpuPasses;
        s.debug.hud.renderer_config_tab = DebugHudRendererConfigTab::Experimental;

        let text = toml::to_string(&s).expect("serialize");
        assert!(text.contains("main_tab = \"gpu_passes\""));
        assert!(text.contains("renderer_config_tab = \"experimental\""));

        let decoded: RendererSettings = toml::from_str(&text).expect("deserialize");
        assert_eq!(decoded.debug.hud.main_tab, DebugHudMainTab::GpuPasses);
        assert_eq!(
            decoded.debug.hud.renderer_config_tab,
            DebugHudRendererConfigTab::Experimental
        );
    }

    #[test]
    fn hud_imgui_visibility_defaults_on_and_roundtrips() {
        let defaults = DebugHudSettings::default();
        assert!(defaults.imgui_visible);

        let mut s = RendererSettings::default();
        s.debug.hud.imgui_visible = false;

        let text = toml::to_string(&s).expect("serialize");
        assert!(text.contains("imgui_visible = false"));

        let decoded: RendererSettings = toml::from_str(&text).expect("deserialize");
        assert!(!decoded.debug.hud.imgui_visible);
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
        assert!(main.any_open());
        assert_eq!(main.first_open(), Some(DebugHudMainTab::Stats));
        assert_eq!(
            main.effective_tab(DebugHudMainTab::DrawState),
            Some(DebugHudMainTab::Stats)
        );
        for &tab in DebugHudMainTab::ALL {
            main.set_open(tab, false);
        }
        assert!(!main.any_open());
        assert_eq!(main.first_open(), None);
        assert_eq!(main.effective_tab(DebugHudMainTab::Stats), None);

        let mut config = DebugHudRendererConfigTabVisibility::default();
        assert!(config.all_open());
        for &tab in DebugHudRendererConfigTab::ALL {
            assert!(config.is_open(tab), "{tab:?} should default open");
        }
        config.set_open(DebugHudRendererConfigTab::Debug, false);
        assert!(!config.is_open(DebugHudRendererConfigTab::Debug));
        assert!(!config.all_open());
        config.set_open(DebugHudRendererConfigTab::Debug, true);
        config.set_open(DebugHudRendererConfigTab::Experimental, false);
        assert!(!config.is_open(DebugHudRendererConfigTab::Experimental));
        assert!(!config.all_open());
    }
}
