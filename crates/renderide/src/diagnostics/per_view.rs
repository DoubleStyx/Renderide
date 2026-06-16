//! Per-view HUD capture policy derived from diagnostics overlay state.

use super::hud::metrics::DebugHudMetricInterest;
use crate::hud_contract::PerViewHudConfig;

impl DebugHudMetricInterest {
    /// Converts the HUD metric interests into per-view capture switches.
    pub(crate) fn per_view_hud_config(self) -> PerViewHudConfig {
        PerViewHudConfig {
            capture_world_mesh_draw_stats: self.wants_stats(),
            capture_world_mesh_view_stats: self.wants_visibility(),
            capture_world_mesh_draw_state_rows: self.wants_draw_state(),
            capture_current_view_texture_2d_asset_ids: self.textures,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::config::DebugHudMainTab;

    use super::DebugHudMetricInterest;

    #[test]
    fn stats_tab_requests_draw_stats_only() {
        let config = DebugHudMetricInterest {
            main_tab: Some(DebugHudMainTab::Stats),
            ..Default::default()
        }
        .per_view_hud_config();

        assert!(config.capture_world_mesh_draw_stats);
        assert!(!config.capture_world_mesh_view_stats);
        assert!(!config.capture_world_mesh_draw_state_rows);
        assert!(!config.capture_current_view_texture_2d_asset_ids);
    }

    #[test]
    fn visibility_stats_section_requests_per_view_stats() {
        let config = DebugHudMetricInterest {
            main_tab: Some(DebugHudMainTab::Stats),
            stats_visibility: true,
            ..Default::default()
        }
        .per_view_hud_config();

        assert!(config.capture_world_mesh_draw_stats);
        assert!(config.capture_world_mesh_view_stats);
        assert!(!config.capture_world_mesh_draw_state_rows);
        assert!(!config.capture_current_view_texture_2d_asset_ids);
    }

    #[test]
    fn draw_state_tab_requests_draw_rows_only() {
        let config = DebugHudMetricInterest {
            main_tab: Some(DebugHudMainTab::DrawState),
            ..Default::default()
        }
        .per_view_hud_config();

        assert!(!config.capture_world_mesh_draw_stats);
        assert!(!config.capture_world_mesh_view_stats);
        assert!(config.capture_world_mesh_draw_state_rows);
        assert!(!config.capture_current_view_texture_2d_asset_ids);
    }

    #[test]
    fn textures_window_requests_current_view_texture_ids() {
        let config = DebugHudMetricInterest {
            textures: true,
            ..Default::default()
        }
        .per_view_hud_config();

        assert!(!config.capture_world_mesh_draw_stats);
        assert!(!config.capture_world_mesh_view_stats);
        assert!(!config.capture_world_mesh_draw_state_rows);
        assert!(config.capture_current_view_texture_2d_asset_ids);
    }

    #[test]
    fn non_world_mesh_tabs_request_no_per_view_world_mesh_data() {
        for main_tab in [
            DebugHudMainTab::ShaderRoutes,
            DebugHudMainTab::GpuMemory,
            DebugHudMainTab::GpuPasses,
        ] {
            let config = DebugHudMetricInterest {
                main_tab: Some(main_tab),
                ..Default::default()
            }
            .per_view_hud_config();

            assert!(!config.capture_world_mesh_draw_stats);
            assert!(!config.capture_world_mesh_view_stats);
            assert!(!config.capture_world_mesh_draw_state_rows);
            assert!(!config.capture_current_view_texture_2d_asset_ids);
        }
    }
}
