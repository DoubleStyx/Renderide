//! Per-pass GPU timing breakdown from [`crate::profiling::GpuProfilerHandle`].
//!
//! Rows come from [`crate::gpu::GpuContext::latest_gpu_pass_timings_handle`], populated by
//! [`crate::gpu::GpuContext::end_gpu_profiler_frame`] each tick. The table is empty until a
//! profiled frame has completed (GPU results lag recording by 1-2 frames).

use imgui::TableFlags;

use crate::profiling::GpuPassEntry;

/// Render the **GPU passes** tab inside the Renderide debug panel.
pub(super) fn gpu_passes_tab(ui: &imgui::Ui, timings: &[GpuPassEntry]) {
    if timings.is_empty() {
        ui.text("Waiting for GPU pass timings…");
        ui.text_disabled("Requires the `tracy` Cargo feature and an adapter with TIMESTAMP_QUERY.");
        return;
    }

    let total_ms: f32 = timings.iter().filter(|e| e.depth == 0).map(|e| e.ms).sum();
    ui.text(format!(
        "{} passes · {:.3} ms total (top-level sum)",
        timings.len(),
        total_ms
    ));
    ui.text_disabled(
        "Depth indent shows nesting from parent phase queries; self-time is the measured pass range.",
    );
    ui.separator();

    let table_flags = TableFlags::BORDERS
        | TableFlags::ROW_BG
        | TableFlags::SCROLL_Y
        | TableFlags::RESIZABLE
        | TableFlags::SIZING_STRETCH_PROP;
    if let Some(_table) =
        ui.begin_table_with_sizing("gpu_pass_rows", 2, table_flags, [0.0, 360.0], 0.0)
    {
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
