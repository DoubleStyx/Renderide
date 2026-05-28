//! Crash context string formatting for ordinary logs.

use std::fmt::Write;

use super::CrashContextSnapshot;

/// Formats a provided crash-context snapshot for tests and higher-level reports.
pub(super) fn format_snapshot_from(s: &CrashContextSnapshot) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "Renderer crash context: uptime_ms={} tick={} phase={} cpu_phase={} mode={} target={} init={} last_host_frame={} prepared_views={} ipc_drop_streaks=primary:{} background:{} driver_backlog={} last_graph_error={} driver_stage={} openxr_call={} xr_finalize={} xr_image={} xr_frame_seq={} xr_command_buffers={} xr_extent={} xr_predicted_time_ns={}",
        s.uptime_ms,
        s.tick_sequence,
        s.tick_phase.as_str(),
        s.cpu_render_phase.as_str(),
        s.render_mode.as_str(),
        s.target_mode.as_str(),
        s.init_state.as_str(),
        s.last_host_frame_index,
        s.prepared_view_count,
        s.primary_ipc_drop_streak,
        s.background_ipc_drop_streak,
        s.driver_backlog,
        s.last_graph_error.as_str(),
        s.driver_stage.as_str(),
        s.openxr_call.as_str(),
        s.xr_finalize_kind.as_str(),
        format_optional_u32(s.xr_finalize_image_index),
        s.xr_finalize_frame_seq,
        s.xr_finalize_command_buffers,
        format_optional_extent(s.xr_finalize_extent),
        format_optional_i64(s.xr_finalize_predicted_display_time_nanos)
    );
    out
}

fn format_optional_u32(value: Option<u32>) -> String {
    value.map_or_else(|| "none".to_owned(), |v| v.to_string())
}

fn format_optional_i64(value: Option<i64>) -> String {
    value.map_or_else(|| "none".to_owned(), |v| v.to_string())
}

fn format_optional_extent(value: Option<(u32, u32)>) -> String {
    value.map_or_else(
        || "none".to_owned(),
        |(width, height)| format!("{width}x{height}"),
    )
}
