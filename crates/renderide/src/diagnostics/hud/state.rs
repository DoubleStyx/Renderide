//! Mutable per-frame UI state for HUD windows: tab selections and filters.
//!
//! Lives on [`crate::diagnostics::DebugHud`] so window bodies can borrow exactly the state they
//! need without the host struct exposing each individual field.

use crate::config::{
    DebugHudMainTab, DebugHudMainTabVisibility, DebugHudRendererConfigTab,
    DebugHudRendererConfigTabVisibility, DebugHudSettings, DebugHudStatsSectionVisibility,
};

/// Per-tab state and filter toggles owned by [`crate::diagnostics::DebugHud`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HudUiState {
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
    /// Expanded/collapsed state for expensive sections inside the **Stats** tab.
    pub stats_sections: DebugHudStatsSectionVisibility,
    /// Last selected tab in **Renderer config**.
    pub renderer_config_tab: DebugHudRendererConfigTab,
    /// Open/closed state for tabs in **Renderer config**.
    pub renderer_config_tabs: DebugHudRendererConfigTabVisibility,
    /// Last selected render-space tab in **Scene transforms**.
    pub scene_transforms_space_id: Option<i32>,
    /// Whether the main debug tab selection still needs to be restored into ImGui.
    pub main_tab_restore_pending: bool,
    /// Whether the renderer config tab selection still needs to be restored into ImGui.
    pub renderer_config_tab_restore_pending: bool,
    /// Whether the scene transforms tab selection still needs to be restored into ImGui.
    pub scene_transforms_space_restore_pending: bool,
}

impl Default for HudUiState {
    fn default() -> Self {
        Self::from_settings(&DebugHudSettings::default())
    }
}

impl HudUiState {
    /// Builds process-local HUD state from persisted renderer config fields.
    pub fn from_settings(settings: &DebugHudSettings) -> Self {
        Self {
            texture_debug_current_view_only: settings.texture_debug_current_view_only,
            draw_state_ui_only: settings.draw_state_ui_only,
            draw_state_only_overrides: settings.draw_state_only_overrides,
            shader_routes_only_fallback: settings.shader_routes_only_fallback,
            main_tab: settings.main_tab,
            main_tabs: settings.main_tabs,
            stats_sections: settings.stats_sections,
            renderer_config_tab: settings.renderer_config_tab,
            renderer_config_tabs: settings.renderer_config_tabs,
            scene_transforms_space_id: settings.scene_transforms_space_id,
            main_tab_restore_pending: true,
            renderer_config_tab_restore_pending: true,
            scene_transforms_space_restore_pending: true,
        }
    }

    /// Projects process-local HUD state back onto the persisted config section.
    ///
    /// Returns `true` when this changed any persisted field.
    pub fn write_to_settings(self, settings: &mut DebugHudSettings) -> bool {
        let before = settings.clone();
        settings.texture_debug_current_view_only = self.texture_debug_current_view_only;
        settings.draw_state_ui_only = self.draw_state_ui_only;
        settings.draw_state_only_overrides = self.draw_state_only_overrides;
        settings.shader_routes_only_fallback = self.shader_routes_only_fallback;
        settings.main_tab = self.main_tab;
        settings.main_tabs = self.main_tabs;
        settings.stats_sections = self.stats_sections;
        settings.renderer_config_tab = self.renderer_config_tab;
        settings.renderer_config_tabs = self.renderer_config_tabs;
        settings.scene_transforms_space_id = self.scene_transforms_space_id;
        before != *settings
    }
}

#[cfg(test)]
mod tests {
    use crate::config::{
        DebugHudMainTab, DebugHudMainTabVisibility, DebugHudRendererConfigTab,
        DebugHudRendererConfigTabVisibility, DebugHudSettings, DebugHudStatsSectionVisibility,
    };

    use super::HudUiState;

    #[test]
    fn default_restores_tab_state_and_disables_every_filter() {
        let s = HudUiState::default();
        assert!(!s.texture_debug_current_view_only);
        assert!(!s.draw_state_ui_only);
        assert!(!s.draw_state_only_overrides);
        assert!(!s.shader_routes_only_fallback);
        assert_eq!(s.main_tab, DebugHudMainTab::Stats);
        assert_eq!(s.main_tabs, DebugHudMainTabVisibility::default());
        assert_eq!(s.stats_sections, DebugHudStatsSectionVisibility::default());
        assert_eq!(s.renderer_config_tab, DebugHudRendererConfigTab::Display);
        assert_eq!(
            s.renderer_config_tabs,
            DebugHudRendererConfigTabVisibility::default()
        );
        assert_eq!(s.scene_transforms_space_id, None);
        assert!(s.main_tab_restore_pending);
        assert!(s.renderer_config_tab_restore_pending);
        assert!(s.scene_transforms_space_restore_pending);
    }

    #[test]
    fn persisted_hud_state_maps_into_runtime_state() {
        let persisted = DebugHudSettings {
            renderer_config_open: false,
            scene_transforms_open: false,
            texture_debug_open: false,
            texture_debug_current_view_only: true,
            draw_state_ui_only: true,
            draw_state_only_overrides: true,
            shader_routes_only_fallback: true,
            main_tab: DebugHudMainTab::GpuMemory,
            main_tabs: DebugHudMainTabVisibility {
                gpu_memory: false,
                ..Default::default()
            },
            stats_sections: DebugHudStatsSectionVisibility {
                graph: false,
                ..Default::default()
            },
            renderer_config_tab: DebugHudRendererConfigTab::PostProcessing,
            renderer_config_tabs: DebugHudRendererConfigTabVisibility {
                post_processing: false,
                ..Default::default()
            },
            scene_transforms_space_id: Some(42),
            ..Default::default()
        };

        let state = HudUiState::from_settings(&persisted);

        assert!(state.texture_debug_current_view_only);
        assert!(state.draw_state_ui_only);
        assert!(state.draw_state_only_overrides);
        assert!(state.shader_routes_only_fallback);
        assert_eq!(state.main_tab, DebugHudMainTab::GpuMemory);
        assert!(!state.main_tabs.is_open(DebugHudMainTab::GpuMemory));
        assert!(!state.stats_sections.graph);
        assert_eq!(
            state.renderer_config_tab,
            DebugHudRendererConfigTab::PostProcessing
        );
        assert!(
            !state
                .renderer_config_tabs
                .is_open(DebugHudRendererConfigTab::PostProcessing)
        );
        assert_eq!(state.scene_transforms_space_id, Some(42));
    }

    #[test]
    fn write_to_settings_reports_dirty_only_for_persisted_fields() {
        let mut settings = DebugHudSettings::default();
        let mut state = HudUiState::from_settings(&settings);

        assert!(!state.write_to_settings(&mut settings));

        state.main_tab_restore_pending = false;
        assert!(!state.write_to_settings(&mut settings));

        state.texture_debug_current_view_only = true;
        assert!(state.write_to_settings(&mut settings));
        assert!(settings.texture_debug_current_view_only);

        assert!(!state.write_to_settings(&mut settings));
        state.main_tabs.set_open(DebugHudMainTab::Stats, false);
        assert!(state.write_to_settings(&mut settings));
        assert!(!settings.main_tabs.stats);

        assert!(!state.write_to_settings(&mut settings));
        state.stats_sections.assets = false;
        assert!(state.write_to_settings(&mut settings));
        assert!(!settings.stats_sections.assets);
    }

    #[test]
    fn write_to_settings_preserves_compatibility_window_open_fields() {
        let mut settings = DebugHudSettings {
            renderer_config_open: false,
            scene_transforms_open: false,
            texture_debug_open: false,
            ..Default::default()
        };
        let mut state = HudUiState::from_settings(&settings);
        state.texture_debug_current_view_only = true;

        assert!(state.write_to_settings(&mut settings));
        assert!(!settings.renderer_config_open);
        assert!(!settings.scene_transforms_open);
        assert!(!settings.texture_debug_open);
    }
}
