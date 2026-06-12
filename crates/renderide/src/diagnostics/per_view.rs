//! Per-view HUD outputs and capture configuration consumed by the diagnostics overlay.

use super::hud::metrics::DebugHudMetricInterest;
use crate::render_graph::blackboard::blackboard_slot;

blackboard_slot! {
    /// Blackboard slot for per-view HUD data collected during recording and merged on the main thread.
    pub PerViewHudOutputsSlot => PerViewHudOutputs,
}

/// HUD payload produced by one view during recording.
#[derive(Default)]
pub struct PerViewHudOutputs {
    /// Latest world-mesh draw stats for the view when the **Stats** tab is active.
    pub world_mesh_draw_stats: Option<crate::world_mesh::WorldMeshDrawStats>,
    /// Latest world-mesh draw-state rows for the view when the **Draw state** tab is active.
    pub world_mesh_draw_state_rows: Option<Vec<crate::world_mesh::WorldMeshDrawStateRow>>,
    /// Texture2D asset ids used by the view when the textures HUD is enabled.
    pub current_view_texture_2d_asset_ids: Vec<i32>,
}

/// Read-only HUD capture switches needed during per-view recording.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PerViewHudConfig {
    /// Whether the **Stats** tab wants world-mesh stats from the current view.
    pub capture_world_mesh_draw_stats: bool,
    /// Whether the **Draw state** tab wants resolved draw rows from the current view.
    pub capture_world_mesh_draw_state_rows: bool,
    /// Whether the textures HUD wants current-view Texture2D ids.
    pub capture_current_view_texture_2d_asset_ids: bool,
}

impl From<DebugHudMetricInterest> for PerViewHudConfig {
    fn from(interest: DebugHudMetricInterest) -> Self {
        Self {
            capture_world_mesh_draw_stats: interest.wants_stats(),
            capture_world_mesh_draw_state_rows: interest.wants_draw_state(),
            capture_current_view_texture_2d_asset_ids: interest.textures,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::config::DebugHudMainTab;

    use super::{DebugHudMetricInterest, PerViewHudConfig};

    #[test]
    fn stats_tab_requests_draw_stats_only() {
        let config = PerViewHudConfig::from(DebugHudMetricInterest {
            main_tab: Some(DebugHudMainTab::Stats),
            ..Default::default()
        });

        assert!(config.capture_world_mesh_draw_stats);
        assert!(!config.capture_world_mesh_draw_state_rows);
        assert!(!config.capture_current_view_texture_2d_asset_ids);
    }

    #[test]
    fn draw_state_tab_requests_draw_rows_only() {
        let config = PerViewHudConfig::from(DebugHudMetricInterest {
            main_tab: Some(DebugHudMainTab::DrawState),
            ..Default::default()
        });

        assert!(!config.capture_world_mesh_draw_stats);
        assert!(config.capture_world_mesh_draw_state_rows);
        assert!(!config.capture_current_view_texture_2d_asset_ids);
    }

    #[test]
    fn textures_window_requests_current_view_texture_ids() {
        let config = PerViewHudConfig::from(DebugHudMetricInterest {
            textures: true,
            ..Default::default()
        });

        assert!(!config.capture_world_mesh_draw_stats);
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
            let config = PerViewHudConfig::from(DebugHudMetricInterest {
                main_tab: Some(main_tab),
                ..Default::default()
            });

            assert!(!config.capture_world_mesh_draw_stats);
            assert!(!config.capture_world_mesh_draw_state_rows);
            assert!(!config.capture_current_view_texture_2d_asset_ids);
        }
    }
}
