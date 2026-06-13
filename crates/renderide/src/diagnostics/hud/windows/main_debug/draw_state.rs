//! **Draw state** tab -- sorted mesh draws with material pipeline state.
//!
//! Column-density redesign:
//! * Narrow fixed-width cells for `Draw / Node / Overlay` so the wide cells
//!   (`Mesh / Material / Pipeline / Blend / Depth / Stencil`) get room to breathe.
//! * No `text_wrapped` -- wrapped cells inflate row height and pixel-shred everything.
//! * Color tags: overlay-layer rows, UI-ish pipelines, non-opaque blends, `ZTest=always`,
//!   stencil-on rows. Designed for at-a-glance "which draws are misrouted".

use imgui::{ListClipper, TableColumnFlags, TableColumnSetup};

use crate::diagnostics::FrameDiagnosticsSnapshot;
use crate::world_mesh::WorldMeshDrawStateRow;

use super::super::super::state::HudUiState;
use super::super::super::view::TabView;
use super::super::labels::{
    blend_mode_label, color_mask_label, draw_state_has_override, draw_state_is_uiish, offset_label,
    pipeline_label, stencil_label, transparent_class_label, ztest_label,
};
use super::super::sections::collapsible_section;
use super::super::table_helpers::scrolling_table_flags;

/// Bright green: row is in the overlay layer (screen-anchored draw).
const TAG_OVERLAY: [f32; 4] = [0.40, 1.00, 0.55, 1.00];
/// Soft gray: dim "not applicable / default" cell.
const DIM: [f32; 4] = [0.55, 0.55, 0.55, 1.00];
/// Cyan: UI / overlay-style pipeline (ui_*, text, overlay).
const TAG_UI: [f32; 4] = [0.55, 0.85, 1.00, 1.00];
/// Orange: alpha / additive / non-opaque blend.
const TAG_BLEND: [f32; 4] = [1.00, 0.72, 0.38, 1.00];
/// Yellow: `ZTest=always` (overlay-always style).
const TAG_ZALWAYS: [f32; 4] = [1.00, 0.90, 0.40, 1.00];
/// Magenta: stencil writes / non-default stencil state.
const TAG_STENCIL: [f32; 4] = [1.00, 0.55, 1.00, 1.00];

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
        collapsible_section(ui, "Filters", true, |ui| draw_filters(ui, state));
        collapsible_section(ui, "Draw rows", true, |ui| {
            let rows = filtered_rows(d, state);
            let overlay_count = rows.iter().filter(|r| r.is_overlay).count();
            let row_count = rows.len();
            let submitted_count = d.mesh_draw.draw_state_rows.len();
            ui.text_disabled(format!(
                "{row_count} rows shown  ({submitted_count} submitted, {overlay_count} overlay)"
            ));

            draw_state_table(ui, &rows);
        });
    }
}

fn draw_filters(ui: &imgui::Ui, state: &mut HudUiState) {
    ui.checkbox("UI / alpha", &mut state.draw_state_ui_only);
    ui.same_line();
    ui.checkbox("Has override", &mut state.draw_state_only_overrides);
    if ui.small_button("Reset filters") {
        state.draw_state_ui_only = false;
        state.draw_state_only_overrides = false;
    }
}

fn filtered_rows<'a>(
    d: &'a FrameDiagnosticsSnapshot,
    state: &HudUiState,
) -> Vec<&'a WorldMeshDrawStateRow> {
    d.mesh_draw
        .draw_state_rows
        .iter()
        .filter(|row| !state.draw_state_ui_only || draw_state_is_uiish(row))
        .filter(|row| !state.draw_state_only_overrides || draw_state_has_override(row))
        .collect()
}

fn draw_state_table(ui: &imgui::Ui, rows: &[&WorldMeshDrawStateRow]) {
    if let Some(_table) = ui.begin_table_with_sizing(
        "draw_state_rows",
        9,
        scrolling_table_flags(),
        [0.0, 360.0],
        0.0,
    ) {
        fixed_col(ui, "Draw", 44.0);
        fixed_col(ui, "Node", 56.0);
        fixed_col(ui, "OVL", 44.0);
        stretch_col(ui, "Mesh");
        stretch_col(ui, "Material");
        stretch_col(ui, "Pipeline");
        stretch_col(ui, "Blend");
        stretch_col(ui, "Depth");
        stretch_col(ui, "Stencil");
        ui.table_headers_row();

        let clip = ListClipper::new(rows.len() as i32);
        let tok = clip.begin(ui);
        for row_i in tok.iter() {
            let row = rows[row_i as usize];
            draw_state_row(ui, row);
        }
    }
}

fn draw_state_row(ui: &imgui::Ui, row: &WorldMeshDrawStateRow) {
    ui.table_next_row();

    ui.table_next_column();
    ui.text(format!("{}", row.draw_index));

    ui.table_next_column();
    ui.text(format!("{}", row.node_id));

    ui.table_next_column();
    if row.is_overlay {
        ui.text_colored(TAG_OVERLAY, "OVL");
    } else {
        ui.text_disabled("--");
    }

    ui.table_next_column();
    ui.text(format!("{}:{}", row.mesh_asset_id, row.slot_index));

    ui.table_next_column();
    ui.text(format!(
        "{} / {:?}",
        row.material_asset_id, row.property_block_slot0
    ));

    ui.table_next_column();
    draw_pipeline_cell(ui, row);

    ui.table_next_column();
    draw_blend_cell(ui, row);

    ui.table_next_column();
    draw_depth_cell(ui, row);

    ui.table_next_column();
    if row.stencil_enabled {
        ui.text_colored(TAG_STENCIL, stencil_label(row));
    } else {
        ui.text_disabled("--");
    }
}

fn draw_pipeline_cell(ui: &imgui::Ui, row: &WorldMeshDrawStateRow) {
    let pipeline = pipeline_label(&row.pipeline);
    if is_ui_pipeline(&pipeline) {
        ui.text_colored(TAG_UI, &pipeline);
    } else {
        ui.text(&pipeline);
    }
}

fn draw_blend_cell(ui: &imgui::Ui, row: &WorldMeshDrawStateRow) {
    let blend = blend_mode_label(row.blend_mode);
    let transparent_class = transparent_class_label(row.transparent_class);
    if transparent_class != "opaque" {
        ui.text_colored(TAG_BLEND, format!("{blend} [{transparent_class}]"));
    } else if blend == "opaque" || blend == "stem" {
        ui.text(&blend);
    } else {
        ui.text_colored(TAG_BLEND, &blend);
    }
}

fn fixed_col(ui: &imgui::Ui, label: &str, width: f32) {
    ui.table_setup_column_with(TableColumnSetup {
        name: label,
        flags: TableColumnFlags::WIDTH_FIXED | TableColumnFlags::NO_HIDE,
        init_width_or_weight: width,
        user_id: imgui::Id::default(),
    });
}

fn stretch_col(ui: &imgui::Ui, label: &str) {
    ui.table_setup_column_with(TableColumnSetup {
        name: label,
        flags: TableColumnFlags::WIDTH_STRETCH,
        init_width_or_weight: 1.0,
        user_id: imgui::Id::default(),
    });
}

/// `ui_*`, `text*`, `overlay*` pipeline stems are treated as UI-ish for the cyan tag.
fn is_ui_pipeline(label: &str) -> bool {
    label.starts_with("ui_") || label.contains("text") || label.contains("overlay")
}

/// Single combined depth cell: `<ztest>/<zwrite>[+offset]`. ZTest=always gets a yellow tag.
fn draw_depth_cell(ui: &imgui::Ui, row: &WorldMeshDrawStateRow) {
    let ztest = ztest_label(row.depth_compare);
    let zwrite = match row.depth_write {
        Some(true) => "on",
        Some(false) => "off",
        None => "pass",
    };
    let offset = offset_label(row.depth_offset);
    let mut text = format!("{ztest}/{zwrite}");
    if offset != "pass" && !offset.is_empty() {
        text.push('+');
        text.push_str(&offset);
    }
    if ztest == "always" {
        ui.text_colored(TAG_ZALWAYS, &text);
    } else {
        ui.text(&text);
    }
    // Dim the "Color" mask details inline only when not RGBA. Keeps a column off the table.
    let mask = color_mask_label(row.color_mask);
    if mask != "RGBA" && mask != "pass" {
        ui.same_line();
        ui.text_colored(DIM, format!("[{mask}]"));
    }
}
