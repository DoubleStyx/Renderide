//! Per-view HUD outputs and capture configuration consumed by the diagnostics overlay.

use crate::render_graph::blackboard::BlackboardSlot;

/// Blackboard slot for per-view HUD data collected during recording and merged on the main thread.
pub struct PerViewHudOutputsSlot;
impl BlackboardSlot for PerViewHudOutputsSlot {
    type Value = PerViewHudOutputs;
}

/// HUD payload produced by one view during recording.
#[derive(Default)]
pub struct PerViewHudOutputs {
    /// Latest world-mesh draw stats for the view when the main HUD is enabled.
    pub world_mesh_draw_stats: Option<crate::world_mesh::WorldMeshDrawStats>,
    /// Latest world-mesh draw-state rows for the view when the main HUD is enabled.
    pub world_mesh_draw_state_rows: Option<Vec<crate::world_mesh::WorldMeshDrawStateRow>>,
    /// Texture2D asset ids used by the view when the textures HUD is enabled.
    pub current_view_texture_2d_asset_ids: Vec<i32>,
}

/// Read-only HUD capture switches needed during per-view recording.
#[derive(Clone, Copy, Debug, Default)]
pub struct PerViewHudConfig {
    /// Whether the main HUD wants world-mesh stats and rows from the current view.
    pub main_enabled: bool,
    /// Whether the textures HUD wants current-view Texture2D ids.
    pub textures_enabled: bool,
}
