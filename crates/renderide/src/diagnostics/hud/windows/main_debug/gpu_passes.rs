//! **GPU passes** tab -- per-pass GPU timing breakdown.
//!
//! Rows come from [`crate::gpu::GpuContext::latest_gpu_pass_timings_handle`], populated by
//! [`crate::gpu::GpuContext::end_gpu_profiler_frame`] each tick. The table is empty until a
//! profiled frame has completed (GPU results lag recording by 1-2 frames).

use crate::profiling::GpuPassEntry;

use super::super::super::state::HudUiState;
use super::super::super::view::TabView;
use super::super::table_helpers::scrolling_table_flags;

/// **GPU passes** tab dispatched from [`super::MainDebugWindow`].
pub struct GpuPassesTab;

impl TabView for GpuPassesTab {
    type Data<'a> = &'a [GpuPassEntry];
    type State = HudUiState;

    fn render(&self, ui: &imgui::Ui, timings: Self::Data<'_>, _state: &mut Self::State) {
        if timings.is_empty() {
            ui.text("Waiting for GPU pass timings...");
            ui.text_disabled(
                "Requires the `tracy` Cargo feature and an adapter with TIMESTAMP_QUERY.",
            );
            return;
        }

        let total_ms: f32 = timings.iter().filter(|e| e.depth == 0).map(|e| e.ms).sum();
        ui.text(format!(
            "{} passes * {:.3} ms total (top-level sum)",
            timings.len(),
            total_ms
        ));
        ui.text_disabled(
            "Depth indent shows nesting from parent phase queries; self-time is the measured pass range.",
        );
        ui.separator();

        if let Some(_table) = ui.begin_table_with_sizing(
            "gpu_pass_rows",
            2,
            scrolling_table_flags(),
            [0.0, 360.0],
            0.0,
        ) {
            ui.table_setup_column("Pass");
            ui.table_setup_column("Time (ms)");
            ui.table_headers_row();
            for entry in timings {
                ui.table_next_row();
                ui.table_next_column();
                let indent = "  ".repeat(entry.depth as usize);
                ui.text(format!("{indent}{}", entry.name));
                ui.table_next_column();
                ui.text(format!("{:.3}", entry.ms));
            }
        }
    }
}
