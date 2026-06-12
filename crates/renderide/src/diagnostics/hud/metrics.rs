//! Metric-interest derivation for the diagnostics HUD.
//!
//! Runtime capture uses this type as the single translation layer from persisted HUD visibility
//! state to the concrete metrics that should be sampled for the next frame.

use crate::config::{DebugHudMainTab, RendererSettings};

/// Per-frame HUD metric interests derived from renderer settings and retained tab state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DebugHudMetricInterest {
    /// Whether the **Frame timing** window needs its compact timing snapshot.
    pub frame_timing: bool,
    /// Effective active top-level tab in **Renderide debug**, if that window should collect data.
    pub main_tab: Option<DebugHudMainTab>,
    /// Whether the **Scene transforms** window needs transform rows.
    pub scene_transforms: bool,
    /// Whether the **Textures** window needs texture rows and current-view usage.
    pub textures: bool,
}

impl DebugHudMetricInterest {
    /// Builds metric interests from current renderer settings.
    pub fn from_settings(settings: &RendererSettings) -> Self {
        let hud = &settings.debug.hud;
        if !hud.imgui_visible {
            return Self::default();
        }

        let main_tab = settings
            .debug
            .debug_hud_enabled
            .then(|| hud.main_tabs.effective_tab(hud.main_tab))
            .flatten();

        Self {
            frame_timing: settings.debug.debug_hud_frame_timing,
            main_tab,
            scene_transforms: settings.debug.debug_hud_transforms,
            textures: settings.debug.debug_hud_textures,
        }
    }

    /// Returns `true` when the main debug panel has an active tab.
    pub fn wants_main_debug(self) -> bool {
        self.main_tab.is_some()
    }

    /// Returns `true` when the **Stats** tab should collect metrics.
    pub fn wants_stats(self) -> bool {
        self.main_tab == Some(DebugHudMainTab::Stats)
    }

    /// Returns `true` when the **Shader routes** tab should collect metrics.
    pub fn wants_shader_routes(self) -> bool {
        self.main_tab == Some(DebugHudMainTab::ShaderRoutes)
    }

    /// Returns `true` when the **Draw state** tab should collect metrics.
    pub fn wants_draw_state(self) -> bool {
        self.main_tab == Some(DebugHudMainTab::DrawState)
    }

    /// Returns `true` when the **GPU memory** tab should collect metrics.
    pub fn wants_gpu_memory(self) -> bool {
        self.main_tab == Some(DebugHudMainTab::GpuMemory)
    }

    /// Returns `true` when the **GPU passes** tab should collect metrics.
    pub fn wants_gpu_passes(self) -> bool {
        self.main_tab == Some(DebugHudMainTab::GpuPasses)
    }

    /// Returns `true` when a throttled allocator report is needed this frame.
    pub fn wants_allocator_totals(self) -> bool {
        self.frame_timing || self.wants_stats() || self.wants_gpu_memory()
    }
}

#[cfg(test)]
mod tests {
    use crate::config::{DebugHudMainTab, DebugHudMainTabVisibility, RendererSettings};

    use super::DebugHudMetricInterest;

    #[test]
    fn hidden_imgui_disables_every_metric() {
        let mut settings = RendererSettings::default();
        settings.debug.debug_hud_frame_timing = true;
        settings.debug.debug_hud_enabled = true;
        settings.debug.debug_hud_transforms = true;
        settings.debug.debug_hud_textures = true;
        settings.debug.hud.imgui_visible = false;

        assert_eq!(
            DebugHudMetricInterest::from_settings(&settings),
            DebugHudMetricInterest::default()
        );
    }

    #[test]
    fn disabled_main_hud_disables_main_tab_metrics_only() {
        let mut settings = RendererSettings::default();
        settings.debug.debug_hud_frame_timing = true;
        settings.debug.debug_hud_enabled = false;
        settings.debug.debug_hud_transforms = true;
        settings.debug.debug_hud_textures = true;

        let interest = DebugHudMetricInterest::from_settings(&settings);

        assert!(interest.frame_timing);
        assert_eq!(interest.main_tab, None);
        assert!(interest.scene_transforms);
        assert!(interest.textures);
    }

    #[test]
    fn selected_closed_main_tab_falls_back_to_first_open_tab() {
        let mut settings = RendererSettings::default();
        settings.debug.debug_hud_enabled = true;
        settings.debug.hud.main_tab = DebugHudMainTab::DrawState;
        settings.debug.hud.main_tabs = DebugHudMainTabVisibility {
            stats: false,
            shader_routes: true,
            draw_state: false,
            gpu_memory: true,
            gpu_passes: true,
        };

        let interest = DebugHudMetricInterest::from_settings(&settings);

        assert_eq!(interest.main_tab, Some(DebugHudMainTab::ShaderRoutes));
        assert!(interest.wants_shader_routes());
        assert!(!interest.wants_draw_state());
    }

    #[test]
    fn all_main_tabs_closed_disable_main_tab_metrics() {
        let mut settings = RendererSettings::default();
        settings.debug.debug_hud_enabled = true;
        for &tab in DebugHudMainTab::ALL {
            settings.debug.hud.main_tabs.set_open(tab, false);
        }

        let interest = DebugHudMetricInterest::from_settings(&settings);

        assert_eq!(interest.main_tab, None);
        assert!(!interest.wants_main_debug());
    }

    #[test]
    fn only_effective_main_tab_reports_interest() {
        let mut settings = RendererSettings::default();
        settings.debug.debug_hud_enabled = true;
        settings.debug.hud.main_tab = DebugHudMainTab::GpuMemory;

        let interest = DebugHudMetricInterest::from_settings(&settings);

        assert_eq!(interest.main_tab, Some(DebugHudMainTab::GpuMemory));
        assert!(interest.wants_gpu_memory());
        assert!(!interest.wants_stats());
        assert!(!interest.wants_shader_routes());
        assert!(!interest.wants_draw_state());
        assert!(!interest.wants_gpu_passes());
    }
}
