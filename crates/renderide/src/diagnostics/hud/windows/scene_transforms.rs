//! Scene transforms overlay: per-render-space world TRS tables.

use imgui::{ListClipper, TableFlags};

use crate::diagnostics::{RenderSpaceTransformsSnapshot, SceneTransformsSnapshot};

use super::super::layout::{self, Viewport, WindowSlot};
use super::super::state::HudUiState;
use super::super::view::HudWindow;

const SCENE_W: f32 = 720.0;
const SCENE_H: f32 = 420.0;

/// **Scene transforms** HUD window -- one tab per render space, clipped TRS table per tab.
///
/// First-use Y prefers the bottom of the viewport minus [`SCENE_H`] but never crosses
/// [`layout::scene_transforms_min_y`] (avoids covering the **Renderer config** + **Frame timing**
/// stack on first use).
pub struct SceneTransformsWindow;

impl HudWindow for SceneTransformsWindow {
    type Data<'a> = &'a SceneTransformsSnapshot;
    type State = HudUiState;

    fn title(&self) -> &str {
        "Scene transforms"
    }

    fn anchor(&self, viewport: Viewport) -> WindowSlot {
        let y = layout::scene_transforms_y(viewport.height as f32, SCENE_H);
        WindowSlot {
            position: [layout::MARGIN, y],
            size_min: [SCENE_W, SCENE_H],
            size_max: [SCENE_W, SCENE_H],
        }
    }

    fn bg_alpha(&self) -> f32 {
        0.85
    }

    fn read_open_flag(&self, state: &Self::State) -> Option<bool> {
        Some(state.scene_transforms_open)
    }

    fn write_open_flag(&self, state: &mut Self::State, value: bool) {
        state.scene_transforms_open = value;
    }

    fn body(&self, ui: &imgui::Ui, snapshot: Self::Data<'_>, _state: &mut Self::State) {
        if snapshot.spaces.is_empty() {
            ui.text("No render spaces.");
            return;
        }
        if let Some(_bar) = ui.tab_bar("scene_transform_tabs") {
            for space in &snapshot.spaces {
                let tab_label = format!("Space {}##tab_space_{}", space.space_id, space.space_id);
                if let Some(_tab) = ui.tab_item(tab_label) {
                    scene_transform_space_tab(ui, space);
                }
            }
        }
    }
}

/// Renders space header fields and the transform table for the active tab.
fn scene_transform_space_tab(ui: &imgui::Ui, space: &RenderSpaceTransformsSnapshot) {
    ui.text(format!(
        "active={}  overlay={}  private={}",
        space.is_active, space.is_overlay, space.is_private
    ));
    let rows = &space.rows;
    let n = rows.len();
    let table_id = format!("transforms##space_{}", space.space_id);
    let table_flags = TableFlags::BORDERS
        | TableFlags::ROW_BG
        | TableFlags::SCROLL_Y
        | TableFlags::RESIZABLE
        | TableFlags::SIZING_STRETCH_PROP;
    if let Some(_table) = ui.begin_table_with_sizing(&table_id, 5, table_flags, [0.0, 320.0], 0.0) {
        ui.table_setup_column("ID");
        ui.table_setup_column("Parent");
        ui.table_setup_column("Translation (world)");
        ui.table_setup_column("Rotation (xyzw)");
        ui.table_setup_column("Scale (world)");
        ui.table_headers_row();

        let clip = ListClipper::new(n as i32);
        let tok = clip.begin(ui);
        for row_i in tok.iter() {
            let row = &rows[row_i as usize];
            ui.table_next_row();
            ui.table_next_column();
            ui.text(format!("{}", row.transform_id));
            ui.table_next_column();
            ui.text(format!("{}", row.parent_id));
            match &row.world {
                None => {
                    ui.table_next_column();
                    ui.text_disabled("--");
                    ui.table_next_column();
                    ui.text_disabled("--");
                    ui.table_next_column();
                    ui.text_disabled("--");
                }
                Some(w) => {
                    ui.table_next_column();
                    ui.text(format!(
                        "{:.4}  {:.4}  {:.4}",
                        w.translation.x, w.translation.y, w.translation.z
                    ));
                    ui.table_next_column();
                    ui.text(format!(
                        "{:.4}  {:.4}  {:.4}  {:.4}",
                        w.rotation.x, w.rotation.y, w.rotation.z, w.rotation.w
                    ));
                    ui.table_next_column();
                    ui.text(format!(
                        "{:.4}  {:.4}  {:.4}",
                        w.scale.x, w.scale.y, w.scale.z
                    ));
                }
            }
        }
    }
}
