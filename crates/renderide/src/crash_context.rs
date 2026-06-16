//! Process-wide crash context updated by cold control-flow boundaries.

/// GPU driver-thread crash context categories.
mod driver;
/// Crash context string formatting for ordinary logs.
mod format;
/// Render-graph error crash context categories.
mod graph_error;
/// Allocation-free crash context formatting for native crash handlers.
mod minimal;
/// Renderer mode crash context categories.
mod modes;
/// Frame and CPU render phase crash context categories.
mod phases;
/// Point-in-time crash context snapshots.
mod snapshot;
/// Process-wide crash context storage and mutation.
mod state;
/// Crash context unit tests.
#[cfg(test)]
mod tests;
/// OpenXR crash context categories.
mod xr;

pub(crate) use driver::DriverStage;
pub(crate) use graph_error::GraphErrorKind;
pub(crate) use minimal::write_minimal_snapshot;
pub(crate) use modes::{InitState, RenderMode, TargetMode};
pub(crate) use phases::{CpuRenderPhase, TickPhase};
pub(crate) use snapshot::CrashContextSnapshot;
pub(crate) use state::{
    clear_openxr_call_if, clear_xr_finalize_state, init_process_context, record_tick_start,
    set_cpu_render_phase, set_driver_backlog, set_driver_stage, set_init_state,
    set_ipc_drop_streaks, set_last_graph_error, set_last_host_frame_index, set_openxr_call,
    set_prepared_view_count, set_render_mode, set_target_mode, set_tick_phase,
    set_tick_phase_label, set_xr_finalize_state,
};
pub(crate) use xr::{OpenXrCall, XrFinalizeKind};

/// Formats the current crash context for ordinary logs and panic reports.
pub(crate) fn format_snapshot() -> String {
    format_snapshot_from(&snapshot())
}

/// Formats a provided crash-context snapshot for tests and higher-level reports.
pub(crate) fn format_snapshot_from(s: &CrashContextSnapshot) -> String {
    format::format_snapshot_from(s)
}

/// Captures the current crash context from atomics.
pub(crate) fn snapshot() -> CrashContextSnapshot {
    state::snapshot()
}
