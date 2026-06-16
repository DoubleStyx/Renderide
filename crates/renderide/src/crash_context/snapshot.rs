//! Point-in-time crash context snapshots.

use super::{
    CpuRenderPhase, DriverStage, GraphErrorKind, InitState, OpenXrCall, RenderMode, TargetMode,
    TickPhase, XrFinalizeKind,
};

/// Point-in-time renderer crash context.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CrashContextSnapshot {
    /// Last recorded process uptime in milliseconds.
    pub(crate) uptime_ms: u64,
    /// Last recorded renderer tick sequence.
    pub(crate) tick_sequence: u64,
    /// Last recorded renderer tick phase.
    pub(crate) tick_phase: TickPhase,
    /// Last recorded CPU render schedule phase.
    pub(crate) cpu_render_phase: CpuRenderPhase,
    /// Last recorded renderer mode.
    pub(crate) render_mode: RenderMode,
    /// Last recorded host initialization state.
    pub(crate) init_state: InitState,
    /// Last recorded requested target mode.
    pub(crate) target_mode: TargetMode,
    /// Last host frame index, or `-1` when no frame submit was applied.
    pub(crate) last_host_frame_index: i64,
    /// Last prepared view count.
    pub(crate) prepared_view_count: u32,
    /// Last observed primary IPC drop streak.
    pub(crate) primary_ipc_drop_streak: u32,
    /// Last observed background IPC drop streak.
    pub(crate) background_ipc_drop_streak: u32,
    /// Last observed GPU driver backlog.
    pub(crate) driver_backlog: u32,
    /// Last recorded render graph error category.
    pub(crate) last_graph_error: GraphErrorKind,
    /// Last recorded GPU driver-thread stage.
    pub(crate) driver_stage: DriverStage,
    /// Currently active OpenXR call.
    pub(crate) openxr_call: OpenXrCall,
    /// Active OpenXR finalize kind.
    pub(crate) xr_finalize_kind: XrFinalizeKind,
    /// Active OpenXR swapchain image index, when known.
    pub(crate) xr_finalize_image_index: Option<u32>,
    /// Active OpenXR finalize frame sequence.
    pub(crate) xr_finalize_frame_seq: u64,
    /// Command buffers in the active OpenXR finalize submit.
    pub(crate) xr_finalize_command_buffers: u32,
    /// OpenXR swapchain extent for the active finalize.
    pub(crate) xr_finalize_extent: Option<(u32, u32)>,
    /// Predicted display time for the active OpenXR finalize.
    pub(crate) xr_finalize_predicted_display_time_nanos: Option<i64>,
}
