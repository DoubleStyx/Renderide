//! Frame timing HUD window — MangoHud-style overlay with FPS, CPU/GPU, RAM/VRAM and a frametime graph.

use crate::diagnostics::FrameTimingHudSnapshot;
use imgui::{Condition, WindowFlags};

use super::super::layout as overlay_layout;

const LABEL_COLOR: [f32; 4] = [0.75, 0.75, 0.80, 1.0];
const VALUE_COLOR: [f32; 4] = [1.00, 1.00, 1.00, 1.0];
const DIM_COLOR: [f32; 4] = [0.62, 0.62, 0.68, 1.0];
const CPU_HEAD_COLOR: [f32; 4] = [0.42, 0.82, 1.00, 1.0];
const GPU_HEAD_COLOR: [f32; 4] = [0.60, 0.90, 0.50, 1.0];
const RAM_HEAD_COLOR: [f32; 4] = [1.00, 0.75, 0.35, 1.0];
const FPS_HEAD_COLOR: [f32; 4] = [1.00, 1.00, 1.00, 1.0];
const GRAPH_COLOR: [f32; 4] = [0.50, 1.00, 0.55, 1.0];

/// Column x offsets relative to the window content origin (after padding). Widest value
/// strings (`23.03 GiB (72%)`, `peak 27.51 ms`) set the overall window width.
const COL_LABEL_X: f32 = 0.0;
const COL_PRIMARY_X: f32 = 54.0;
const COL_SECONDARY_LABEL_X: f32 = 150.0;
const COL_SECONDARY_VALUE_X: f32 = 196.0;
const CONTENT_WIDTH: f32 = 320.0;
const GRAPH_HEIGHT: f32 = 46.0;

pub(super) fn frame_timing_window(ui: &imgui::Ui, timing: Option<&FrameTimingHudSnapshot>) {
    let window_flags = WindowFlags::ALWAYS_AUTO_RESIZE
        | WindowFlags::NO_SAVED_SETTINGS
        | WindowFlags::NO_FOCUS_ON_APPEARING
        | WindowFlags::NO_NAV;
    ui.window("Frame timing")
        .position(overlay_layout::frame_timing_xy(), Condition::FirstUseEver)
        .bg_alpha(0.82)
        .flags(window_flags)
        .build(|| {
            let Some(t) = timing else {
                ui.text("Waiting for snapshot...");
                return;
            };
            // Reserve a fixed content width so every row and the graph share the same extent.
            ui.dummy([CONTENT_WIDTH, 0.0]);
            render_rows(ui, t);
            ui.separator();
            render_frametime_graph(ui, t);
        });
}

fn fps_color(fps: f64) -> [f32; 4] {
    if fps >= 90.0 {
        [0.50, 1.00, 0.55, 1.0]
    } else if fps >= 45.0 {
        [1.00, 0.95, 0.40, 1.0]
    } else {
        [1.00, 0.45, 0.40, 1.0]
    }
}

fn render_rows(ui: &imgui::Ui, t: &FrameTimingHudSnapshot) {
    let fps = t.fps_from_wall();
    row(
        ui,
        ("FPS", FPS_HEAD_COLOR),
        (format!("{fps:6.1}"), fps_color(fps)),
        Some(("Frame", LABEL_COLOR)),
        Some((format!("{:5.2} ms", t.wall_frame_time_ms), VALUE_COLOR)),
    );

    let cpu_ms = ms_or_dash(t.cpu_frame_until_submit_ms);
    row(
        ui,
        ("CPU", CPU_HEAD_COLOR),
        (format!("{:5.1}%", t.host_cpu_usage_percent), VALUE_COLOR),
        Some(("tick", LABEL_COLOR)),
        Some((format!("{cpu_ms} ms"), VALUE_COLOR)),
    );

    let gpu_ms = ms_or_dash(t.gpu_frame_after_submit_ms);
    row(
        ui,
        ("GPU", GPU_HEAD_COLOR),
        (format!("{gpu_ms} ms"), VALUE_COLOR),
        None,
        None,
    );

    let proc_ram = t
        .process_ram_bytes
        .map(format_bytes_gib)
        .unwrap_or_else(|| "-".to_string());
    let host_ram = format_bytes_gib(t.host_ram_used_bytes);
    let host_ram_pct = if t.host_ram_total_bytes > 0 {
        (t.host_ram_used_bytes as f64 / t.host_ram_total_bytes as f64) * 100.0
    } else {
        0.0
    };
    row(
        ui,
        ("RAM", RAM_HEAD_COLOR),
        (proc_ram, VALUE_COLOR),
        Some(("host", DIM_COLOR)),
        Some((format!("{host_ram} ({host_ram_pct:.0}%)"), DIM_COLOR)),
    );
}

fn row(
    ui: &imgui::Ui,
    (label, label_color): (&str, [f32; 4]),
    (value, value_color): (String, [f32; 4]),
    secondary_label: Option<(&str, [f32; 4])>,
    secondary_value: Option<(String, [f32; 4])>,
) {
    // Start the row at the natural cursor (respects window padding), then align columns with
    // `same_line_with_pos` which is window-local, padding-aware.
    if COL_LABEL_X > 0.0 {
        ui.same_line_with_pos(COL_LABEL_X);
    }
    ui.text_colored(label_color, label);

    ui.same_line_with_pos(COL_PRIMARY_X);
    ui.text_colored(value_color, value);

    if let Some((slabel, slabel_color)) = secondary_label {
        ui.same_line_with_pos(COL_SECONDARY_LABEL_X);
        ui.text_colored(slabel_color, slabel);
    }
    if let Some((sval, sval_color)) = secondary_value {
        ui.same_line_with_pos(COL_SECONDARY_VALUE_X);
        ui.text_colored(sval_color, sval);
    }
}

fn render_frametime_graph(ui: &imgui::Ui, t: &FrameTimingHudSnapshot) {
    let width = ui.content_region_avail()[0].max(CONTENT_WIDTH);
    if t.frame_time_history.is_empty() {
        ui.dummy([width, GRAPH_HEIGHT]);
        return;
    }
    let peak = t.frame_time_history.iter().copied().fold(0.0_f32, f32::max);
    let (lo, hi) = scale_bounds(&t.frame_time_history);
    let overlay = format!("peak {peak:5.2} ms");
    let style = ui.push_style_color(imgui::StyleColor::PlotLines, GRAPH_COLOR);
    ui.plot_lines("##frametime", &t.frame_time_history)
        .scale_min(lo)
        .scale_max(hi)
        .graph_size([width, GRAPH_HEIGHT])
        .overlay_text(overlay)
        .build();
    style.pop();
}

fn scale_bounds(values: &[f32]) -> (f32, f32) {
    let mut hi = 0.0_f32;
    for &v in values {
        if v.is_finite() && v > hi {
            hi = v;
        }
    }
    if hi <= f32::EPSILON {
        return (0.0, 16.67);
    }
    (0.0, hi * 1.1)
}

fn ms_or_dash(ms: Option<f64>) -> String {
    match ms {
        Some(v) => format!("{v:5.2}"),
        None => "  -  ".to_string(),
    }
}

fn format_bytes_gib(bytes: u64) -> String {
    let gib = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    if gib >= 1.0 {
        format!("{gib:4.2} GiB")
    } else {
        let mib = bytes as f64 / (1024.0 * 1024.0);
        format!("{mib:5.0} MiB")
    }
}
