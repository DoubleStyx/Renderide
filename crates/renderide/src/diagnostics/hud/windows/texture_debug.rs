//! Texture pool debug window with current-view filtering.

use imgui::ListClipper;

use crate::diagnostics::TextureDebugSnapshot;

use super::super::fmt as hud_fmt;
use super::super::layout::{self, Viewport, WindowSlot};
use super::super::state::HudUiState;
use super::super::view::HudWindow;
use super::table_helpers::scrolling_table_flags;

const TEXTURE_DEBUG_W: f32 = 860.0;
const TEXTURE_DEBUG_H: f32 = 420.0;
/// Y offset stacks the **Textures** window below the **Renderer config** window on first use.
const TEXTURE_DEBUG_Y_OFFSET: f32 = 360.0;

/// **Textures** HUD window -- texture pool listing with current-view filtering.
pub struct TextureDebugWindow;

impl HudWindow for TextureDebugWindow {
    type Data<'a> = &'a TextureDebugSnapshot;
    type State = HudUiState;

    fn title(&self) -> &str {
        "Textures"
    }

    fn anchor(&self, _viewport: Viewport) -> WindowSlot {
        WindowSlot {
            position: [layout::MARGIN, layout::MARGIN + TEXTURE_DEBUG_Y_OFFSET],
            size_min: [TEXTURE_DEBUG_W, TEXTURE_DEBUG_H],
            size_max: [TEXTURE_DEBUG_W, TEXTURE_DEBUG_H],
        }
    }

    fn bg_alpha(&self) -> f32 {
        0.85
    }

    fn read_open_flag(&self, state: &Self::State) -> Option<bool> {
        Some(state.texture_debug_open)
    }

    fn write_open_flag(&self, state: &mut Self::State, value: bool) {
        state.texture_debug_open = value;
    }

    fn body(&self, ui: &imgui::Ui, snapshot: Self::Data<'_>, state: &mut Self::State) {
        ui.checkbox(
            "Only current view",
            &mut state.texture_debug_current_view_only,
        );
        ui.text(format!(
            "{} textures  |  {} current-view  |  {} total",
            snapshot.rows.len(),
            snapshot.current_view_texture_count,
            hud_fmt::bytes_compact(snapshot.total_resident_bytes)
        ));
        let rows: Vec<_> = snapshot
            .rows
            .iter()
            .filter(|row| !state.texture_debug_current_view_only || row.used_by_current_view)
            .collect();
        if let Some(_table) = ui.begin_table_with_sizing(
            "texture_debug_rows",
            8,
            scrolling_table_flags(),
            [0.0, 330.0],
            0.0,
        ) {
            ui.table_setup_column("Asset");
            ui.table_setup_column("Size");
            ui.table_setup_column("Mips");
            ui.table_setup_column("Bytes");
            ui.table_setup_column("Host");
            ui.table_setup_column("GPU");
            ui.table_setup_column("Sampler");
            ui.table_setup_column("View");
            ui.table_headers_row();
            let clip = ListClipper::new(rows.len() as i32);
            let tok = clip.begin(ui);
            for row_i in tok.iter() {
                let row = rows[row_i as usize];
                ui.table_next_row();
                ui.table_next_column();
                ui.text(format!("{}", row.asset_id));
                ui.table_next_column();
                ui.text(format!("{}x{}", row.width, row.height));
                ui.table_next_column();
                ui.text(format!(
                    "{}/{}",
                    row.mip_levels_resident, row.mip_levels_total
                ));
                ui.table_next_column();
                ui.text(hud_fmt::bytes_compact(row.resident_bytes));
                ui.table_next_column();
                ui.text(format!("{:?} {:?}", row.host_format, row.color_profile));
                ui.table_next_column();
                ui.text(format!("{:?}", row.wgpu_format));
                ui.table_next_column();
                ui.text(format!(
                    "{:?} aniso={} wrap={:?}/{:?} bias={:.2}",
                    row.filter_mode, row.aniso_level, row.wrap_u, row.wrap_v, row.mipmap_bias
                ));
                ui.table_next_column();
                ui.text(if row.used_by_current_view {
                    "current"
                } else {
                    ""
                });
            }
        }
    }
}
