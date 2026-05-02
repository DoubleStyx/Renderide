//! **Draw state** tab -- sorted mesh draws with material pipeline state.

use imgui::ListClipper;

use crate::diagnostics::FrameDiagnosticsSnapshot;
use crate::world_mesh::WorldMeshDrawStateRow;

use super::super::super::state::HudUiState;
use super::super::super::view::TabView;
use super::super::labels::{
    blend_mode_label, color_mask_label, draw_state_has_override, draw_state_is_uiish, offset_label,
    pipeline_label, stencil_label, ztest_label,
};
use super::super::table_helpers::scrolling_table_flags;

/// **Draw state** tab dispatched from [`super::MainDebugWindow`].
pub struct DrawStateTab;

impl TabView for DrawStateTab {
    type Data<'a> = Option<&'a FrameDiagnosticsSnapshot>;
    type State = HudUiState;

    fn render(&self, ui: &imgui::Ui, data: Self::Data<'_>, state: &mut Self::State) {
        let Some(d) = data else {
            ui.text("Waiting for frame diagnostics");
            return;
        };
        ui.checkbox("Only UI / alpha rows", &mut state.draw_state_ui_only);
        ui.checkbox(
            "Only render-state overrides",
            &mut state.draw_state_only_overrides,
        );

        let rows: Vec<&WorldMeshDrawStateRow> = d
            .mesh_draw
            .draw_state_rows
            .iter()
            .filter(|row| !state.draw_state_ui_only || draw_state_is_uiish(row))
            .filter(|row| !state.draw_state_only_overrides || draw_state_has_override(row))
            .collect();
        ui.text(format!(
            "{} rows ({} submitted)",
            rows.len(),
            d.mesh_draw.draw_state_rows.len()
        ));

        if let Some(_table) = ui.begin_table_with_sizing(
            "draw_state_rows",
            11,
            scrolling_table_flags(),
            [0.0, 360.0],
            0.0,
        ) {
            ui.table_setup_column("Draw");
            ui.table_setup_column("Node");
            ui.table_setup_column("Mesh");
            ui.table_setup_column("Material");
            ui.table_setup_column("Pipeline");
            ui.table_setup_column("Blend");
            ui.table_setup_column("ZWrite");
            ui.table_setup_column("ZTest");
            ui.table_setup_column("Offset");
            ui.table_setup_column("Color");
            ui.table_setup_column("Stencil");
            ui.table_headers_row();
            let clip = ListClipper::new(rows.len() as i32);
            let tok = clip.begin(ui);
            for row_i in tok.iter() {
                let row = rows[row_i as usize];
                ui.table_next_row();
                ui.table_next_column();
                ui.text(format!("{}", row.draw_index));
                ui.table_next_column();
                ui.text(format!("{}", row.node_id));
                ui.table_next_column();
                ui.text(format!("{}:{}", row.mesh_asset_id, row.slot_index));
                ui.table_next_column();
                ui.text(format!(
                    "{} / {:?}",
                    row.material_asset_id, row.property_block_slot0
                ));
                ui.table_next_column();
                ui.text_wrapped(pipeline_label(&row.pipeline));
                ui.table_next_column();
                ui.text(blend_mode_label(row.blend_mode));
                ui.table_next_column();
                ui.text(match row.depth_write {
                    Some(true) => "on",
                    Some(false) => "off",
                    None => "pass",
                });
                ui.table_next_column();
                ui.text(ztest_label(row.depth_compare));
                ui.table_next_column();
                ui.text(offset_label(row.depth_offset));
                ui.table_next_column();
                ui.text(color_mask_label(row.color_mask));
                ui.table_next_column();
                ui.text_wrapped(stencil_label(row));
            }
        }
    }
}
