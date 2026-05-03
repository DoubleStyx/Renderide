//! **GPU memory** tab -- full wgpu allocator report (refreshed on a timer).

use imgui::{ListClipper, TableFlags};

use crate::diagnostics::FrameDiagnosticsSnapshot;

use super::super::super::fmt as hud_fmt;
use super::super::super::state::HudUiState;
use super::super::super::view::TabView;

/// **GPU memory** tab dispatched from [`super::MainDebugWindow`].
///
/// Row labels mirror wgpu `label` strings on buffers and textures (often empty).
pub struct GpuMemoryTab;

impl TabView for GpuMemoryTab {
    type Data<'a> = Option<&'a FrameDiagnosticsSnapshot>;
    type State = HudUiState;

    fn render(&self, ui: &imgui::Ui, data: Self::Data<'_>, _state: &mut Self::State) {
        let Some(d) = data else {
            ui.text("Waiting for frame diagnostics...");
            return;
        };

        ui.text_disabled(format!(
            "Next full report refresh in ~{:.1} s (detail lags Stats totals by up to one interval).",
            d.gpu_allocator.report_next_refresh_in_secs
        ));

        let Some(hud) = &d.gpu_allocator.report else {
            ui.separator();
            ui.text_wrapped(
                "Full allocator report unavailable: unsupported backend, or not yet collected. \
                 The Stats tab still shows totals when the device reports them.",
            );
            return;
        };

        let r = hud.report.as_ref();

        ui.separator();
        ui.text("Summary (wgpu device allocator)");
        ui.text(format!(
            "{} / {}  allocated / reserved  |  {} blocks  |  {} sub-allocations",
            hud_fmt::bytes_compact(r.total_allocated_bytes),
            hud_fmt::bytes_compact(r.total_reserved_bytes),
            r.blocks.len(),
            r.allocations.len(),
        ));
        ui.text_disabled(
            "Sizes are device-local sub-allocations; Vulkan memory-type names are not exposed here.",
        );

        ui.separator();
        ui.text("Sub-allocations (by size, largest first)");
        let n = hud.allocation_indices_by_size.len();
        let table_flags = TableFlags::BORDERS
            | TableFlags::ROW_BG
            | TableFlags::SCROLL_Y
            | TableFlags::RESIZABLE
            | TableFlags::SIZING_STRETCH_PROP;
        if let Some(_table) =
            ui.begin_table_with_sizing("gpu_alloc_rows", 3, table_flags, [0.0, 360.0], 0.0)
        {
            ui.table_setup_column("Size");
            ui.table_setup_column("Offset");
            ui.table_setup_column("Label");
            ui.table_headers_row();
            let clip = ListClipper::new(n as i32);
            let tok = clip.begin(ui);
            for row_i in tok.iter() {
                let idx = hud.allocation_indices_by_size[row_i as usize];
                let a = &r.allocations[idx];
                ui.table_next_row();
                ui.table_next_column();
                ui.text(hud_fmt::bytes_compact(a.size));
                ui.table_next_column();
                ui.text(format!("{}", a.offset));
                let name = if a.name.is_empty() {
                    "(no label)"
                } else {
                    a.name.as_str()
                };
                ui.table_next_column();
                ui.text_wrapped(name);
            }
        }

        ui.separator();
        if let Some(_node) = ui.tree_node("Memory blocks") {
            let nb = r.blocks.len();
            if let Some(_table) = ui.begin_table_with_sizing(
                "gpu_mem_blocks",
                3,
                TableFlags::BORDERS | TableFlags::ROW_BG | TableFlags::SIZING_STRETCH_PROP,
                [0.0, 200.0],
                0.0,
            ) {
                ui.table_setup_column("Block");
                ui.table_setup_column("Size");
                ui.table_setup_column("Sub-allocs");
                ui.table_headers_row();
                for bi in 0..nb {
                    let b = &r.blocks[bi];
                    let sub = b.allocations.end.saturating_sub(b.allocations.start);
                    ui.table_next_row();
                    ui.table_next_column();
                    ui.text(format!("{bi}"));
                    ui.table_next_column();
                    ui.text(hud_fmt::bytes_compact(b.size));
                    ui.table_next_column();
                    ui.text(format!("{sub}"));
                }
            }
        }
    }
}
