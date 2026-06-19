//! Frame timing HUD window with FPS, CPU/GPU/host pacing, memory, and rolling stats.

use imgui::WindowFlags;

use crate::diagnostics::{FrameTimingHudSnapshot, FrameTimingOnePercentStats};

use super::super::layout::{self, Viewport, WindowSlot};
use super::super::state::HudUiState;
use super::super::view::HudWindow;

const VALUE_COLOR: [f32; 4] = [1.00, 1.00, 1.00, 1.0];
const DIM_COLOR: [f32; 4] = [0.62, 0.62, 0.68, 1.0];
const CPU_HEAD_COLOR: [f32; 4] = [0.42, 0.82, 1.00, 1.0];
const GPU_HEAD_COLOR: [f32; 4] = [0.60, 0.90, 0.50, 1.0];
const HOST_HEAD_COLOR: [f32; 4] = [1.00, 0.82, 0.35, 1.0];
const RAM_HEAD_COLOR: [f32; 4] = [1.00, 0.75, 0.35, 1.0];
const FPS_HEAD_COLOR: [f32; 4] = [1.00, 1.00, 1.00, 1.0];
const GRAPH_COLOR: [f32; 4] = [0.50, 1.00, 0.55, 1.0];

const COL_LABEL_X: f32 = 0.0;
const COL_PRIMARY_X: f32 = 62.0;
const COL_SECONDARY_LABEL_X: f32 = 178.0;
const COL_SECONDARY_VALUE_X: f32 = 250.0;
const CONTENT_WIDTH: f32 = 370.0;
const GRAPH_HEIGHT: f32 = 46.0;

/// **Frame timing** HUD window anchored under the **Renderer config** column.
pub struct FrameTimingWindow;

impl HudWindow for FrameTimingWindow {
    type Data<'a> = Option<&'a FrameTimingHudSnapshot>;
    type State = HudUiState;

    fn title(&self) -> &str {
        "Frame timing"
    }

    fn anchor(&self, _viewport: Viewport) -> WindowSlot {
        let xy = layout::frame_timing_xy();
        WindowSlot {
            position: xy,
            size: [CONTENT_WIDTH, 185.0],
            size_min: [CONTENT_WIDTH, 0.0],
            size_max: [f32::INFINITY, f32::INFINITY],
        }
    }

    fn flags(&self) -> WindowFlags {
        WindowFlags::ALWAYS_AUTO_RESIZE | WindowFlags::NO_FOCUS_ON_APPEARING | WindowFlags::NO_NAV
    }

    fn bg_alpha(&self) -> f32 {
        0.82
    }

    fn body(&self, ui: &imgui::Ui, data: Self::Data<'_>, _state: &mut Self::State) {
        let Some(t) = data else {
            ui.text("Waiting for snapshot...");
            return;
        };
        ui.dummy([CONTENT_WIDTH, 0.0]);
        render_rows(ui, t);
        ui.separator();
        render_frametime_graph(ui, t);
    }
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
    render_cadence_rows(ui, t);
    render_work_rows(ui, t);
    render_memory_rows(ui, t);
}

fn render_cadence_rows(ui: &imgui::Ui, t: &FrameTimingHudSnapshot) {
    let fps = t.fps_from_wall();
    metric_row(
        ui,
        ("FPS", FPS_HEAD_COLOR),
        (format!("{fps:6.1}"), fps_color(fps)),
        Some(("Low/High", format_percent_pair_fps(t.history_stats.fps))),
        Some(FPS_TOOLTIP),
    );

    metric_row(
        ui,
        ("Frame", FPS_HEAD_COLOR),
        (
            format!("{:5.2} ms", t.wall_frame_time_ms_smoothed),
            VALUE_COLOR,
        ),
        Some(("Low/High", format_percent_pair_ms(t.history_stats.frame_ms))),
        Some(FRAME_TOOLTIP),
    );
}

fn render_work_rows(ui: &imgui::Ui, t: &FrameTimingHudSnapshot) {
    metric_row(
        ui,
        ("CPU", CPU_HEAD_COLOR),
        (
            format!("{} ms", ms_or_dash(t.cpu_frame_ms_smoothed)),
            VALUE_COLOR,
        ),
        Some(("Low/High", format_percent_pair_ms(t.history_stats.cpu_ms))),
        Some(CPU_TOOLTIP),
    );

    metric_row(
        ui,
        ("GPU", GPU_HEAD_COLOR),
        (
            format!("{} ms", ms_or_dash(t.gpu_frame_ms_smoothed)),
            VALUE_COLOR,
        ),
        Some(("Low/High", format_percent_pair_ms(t.history_stats.gpu_ms))),
        Some(GPU_TOOLTIP),
    );

    metric_row(
        ui,
        ("Host", HOST_HEAD_COLOR),
        (
            format!("{} ms", ms_or_dash(t.host_frame_ms_smoothed)),
            VALUE_COLOR,
        ),
        Some(("Low/High", format_percent_pair_ms(t.history_stats.host_ms))),
        Some(HOST_TOOLTIP),
    );
}

fn render_memory_rows(ui: &imgui::Ui, t: &FrameTimingHudSnapshot) {
    memory_row(
        ui,
        ("RAM", RAM_HEAD_COLOR),
        t.process_ram_bytes
            .map_or_else(|| "-".to_string(), format_bytes_gib),
        "host",
        host_ram_text(t),
        Some(RAM_TOOLTIP),
    );

    memory_row(
        ui,
        ("VRAM", GPU_HEAD_COLOR),
        t.gpu_allocator_allocated_bytes
            .map_or_else(|| "-".to_string(), format_bytes_gib),
        "resv",
        t.gpu_allocator_reserved_bytes
            .map_or_else(|| "-".to_string(), format_bytes_gib),
        Some(VRAM_TOOLTIP),
    );
}

fn metric_row(
    ui: &imgui::Ui,
    (label, label_color): (&str, [f32; 4]),
    (value, value_color): (String, [f32; 4]),
    secondary: Option<(&str, String)>,
    tooltip: Option<&str>,
) {
    if COL_LABEL_X > 0.0 {
        ui.same_line_with_pos(COL_LABEL_X);
    }
    ui.text_colored(label_color, label);
    maybe_tooltip(ui, tooltip);

    ui.same_line_with_pos(COL_PRIMARY_X);
    ui.text_colored(value_color, value);

    if let Some((secondary_label, secondary_value)) = secondary {
        ui.same_line_with_pos(COL_SECONDARY_LABEL_X);
        ui.text_colored(DIM_COLOR, secondary_label);
        ui.same_line_with_pos(COL_SECONDARY_VALUE_X);
        ui.text_colored(DIM_COLOR, secondary_value);
    }
}

fn memory_row(
    ui: &imgui::Ui,
    (label, label_color): (&str, [f32; 4]),
    value: String,
    secondary_label: &str,
    secondary_value: String,
    tooltip: Option<&str>,
) {
    if COL_LABEL_X > 0.0 {
        ui.same_line_with_pos(COL_LABEL_X);
    }
    ui.text_colored(label_color, label);
    maybe_tooltip(ui, tooltip);

    ui.same_line_with_pos(COL_PRIMARY_X);
    ui.text_colored(VALUE_COLOR, value);
    ui.same_line_with_pos(COL_SECONDARY_LABEL_X);
    ui.text_colored(DIM_COLOR, secondary_label);
    ui.same_line_with_pos(COL_SECONDARY_VALUE_X);
    ui.text_colored(DIM_COLOR, secondary_value);
}

fn maybe_tooltip(ui: &imgui::Ui, tooltip: Option<&str>) {
    if let Some(text) = tooltip
        && ui.is_item_hovered()
    {
        ui.tooltip_text(text);
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

fn format_percent_pair_fps(stats: FrameTimingOnePercentStats) -> String {
    match (stats.low, stats.high) {
        (Some(low), Some(high)) => format!("{low:.1}/{high:.1}"),
        _ => "-/-".to_string(),
    }
}

fn format_percent_pair_ms(stats: FrameTimingOnePercentStats) -> String {
    match (stats.low, stats.high) {
        (Some(low), Some(high)) => format!("{high:.2}/{low:.2}"),
        _ => "-/-".to_string(),
    }
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

fn host_ram_text(t: &FrameTimingHudSnapshot) -> String {
    let host_ram = format_bytes_gib(t.host_ram_used_bytes);
    let host_ram_pct = if t.host_ram_total_bytes > 0 {
        (t.host_ram_used_bytes as f64 / t.host_ram_total_bytes as f64) * 100.0
    } else {
        0.0
    };
    format!("{host_ram} ({host_ram_pct:.0}%)")
}

const FPS_TOOLTIP: &str = "Smoothed wall-frame FPS. Low/High is a raw rolling one-second percentile window; spikes remain until they age out.";
const FRAME_TOOLTIP: &str = "Wall-clock between consecutive winit ticks. Includes presentation mode, FPS caps, host lockstep, presentation waits, and event-loop pacing.";
const CPU_TOOLTIP: &str = "Main-thread active renderer work from frame start to submit dispatch, minus explicit pacing waits.";
const GPU_TOOLTIP: &str = "Real primary GPU busy time measured by hardware timestamp brackets around primary render submits.";
const HOST_TOOLTIP: &str = "Renderer-observed host update turnaround: outgoing FrameStartData send to matching inbound FrameSubmitData queue receipt.";
const RAM_TOOLTIP: &str = "RAM shows renderer resident process memory plus OS-reported host used/total memory. It is not allocator reserved memory.";
const VRAM_TOOLTIP: &str = "Wgpu allocator memory tracked by the active backend. Shows live allocated bytes and reserved allocator capacity.";

#[cfg(test)]
mod tests {
    use super::{FrameTimingOnePercentStats, format_percent_pair_fps, format_percent_pair_ms};

    #[test]
    fn fps_low_high_uses_realtime_precision() {
        let text = format_percent_pair_fps(FrameTimingOnePercentStats {
            low: Some(59.876),
            high: Some(120.123),
        });
        assert_eq!(text, "59.9/120.1");
    }

    #[test]
    fn ms_low_high_displays_slowest_then_fastest() {
        let text = format_percent_pair_ms(FrameTimingOnePercentStats {
            low: Some(8.333),
            high: Some(16.666),
        });
        assert_eq!(text, "16.67/8.33");
    }

    #[test]
    fn missing_low_high_uses_compact_dash() {
        let stats = FrameTimingOnePercentStats::default();
        assert_eq!(format_percent_pair_fps(stats), "-/-");
        assert_eq!(format_percent_pair_ms(stats), "-/-");
    }
}
