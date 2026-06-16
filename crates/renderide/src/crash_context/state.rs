//! Process-wide crash context storage and mutation.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicI64, AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::time::Instant;

use super::phases::tick_phase_from_label;
use super::{
    CpuRenderPhase, CrashContextSnapshot, DriverStage, GraphErrorKind, InitState, OpenXrCall,
    RenderMode, TargetMode, TickPhase, XrFinalizeKind,
};

const NONE_U32: u32 = u32::MAX;
const NONE_I64: i64 = i64::MIN;

static START_INSTANT: OnceLock<Instant> = OnceLock::new();
static UPTIME_MS: AtomicU64 = AtomicU64::new(0);
static TICK_SEQUENCE: AtomicU64 = AtomicU64::new(0);
static TICK_PHASE: AtomicU8 = AtomicU8::new(TickPhase::Unknown as u8);
static CPU_RENDER_PHASE: AtomicU8 = AtomicU8::new(CpuRenderPhase::Unknown as u8);
static RENDER_MODE: AtomicU8 = AtomicU8::new(RenderMode::Unknown as u8);
static INIT_STATE: AtomicU8 = AtomicU8::new(InitState::NotStarted as u8);
static TARGET_MODE: AtomicU8 = AtomicU8::new(TargetMode::Unknown as u8);
static LAST_HOST_FRAME_INDEX: AtomicI64 = AtomicI64::new(-1);
static PREPARED_VIEW_COUNT: AtomicU32 = AtomicU32::new(0);
static PRIMARY_IPC_DROP_STREAK: AtomicU32 = AtomicU32::new(0);
static BACKGROUND_IPC_DROP_STREAK: AtomicU32 = AtomicU32::new(0);
static DRIVER_BACKLOG: AtomicU32 = AtomicU32::new(0);
static LAST_GRAPH_ERROR: AtomicU8 = AtomicU8::new(GraphErrorKind::None as u8);
static DRIVER_STAGE: AtomicU8 = AtomicU8::new(DriverStage::Unknown as u8);
static OPENXR_CALL: AtomicU8 = AtomicU8::new(OpenXrCall::None as u8);
static XR_FINALIZE_KIND: AtomicU8 = AtomicU8::new(XrFinalizeKind::None as u8);
static XR_FINALIZE_IMAGE_INDEX: AtomicU32 = AtomicU32::new(NONE_U32);
static XR_FINALIZE_FRAME_SEQ: AtomicU64 = AtomicU64::new(0);
static XR_FINALIZE_COMMAND_BUFFERS: AtomicU32 = AtomicU32::new(0);
static XR_FINALIZE_EXTENT_WIDTH: AtomicU32 = AtomicU32::new(NONE_U32);
static XR_FINALIZE_EXTENT_HEIGHT: AtomicU32 = AtomicU32::new(NONE_U32);
static XR_FINALIZE_PREDICTED_TIME_NS: AtomicI64 = AtomicI64::new(NONE_I64);

/// Initializes process lifetime tracking.
pub(crate) fn init_process_context() {
    let _ = START_INSTANT.set(Instant::now());
    refresh_uptime();
}

/// Records the start of a renderer tick.
pub(crate) fn record_tick_start() {
    TICK_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    refresh_uptime();
}

/// Records the active tick phase.
pub(crate) fn set_tick_phase(phase: TickPhase) {
    TICK_PHASE.store(phase as u8, Ordering::Relaxed);
    refresh_uptime();
}

/// Records the active tick phase from an app-driver trace label.
pub(crate) fn set_tick_phase_label(label: &str) {
    set_tick_phase(tick_phase_from_label(label));
}

/// Records the active CPU render schedule phase.
pub(crate) fn set_cpu_render_phase(phase: CpuRenderPhase) {
    CPU_RENDER_PHASE.store(phase as u8, Ordering::Relaxed);
    refresh_uptime();
}

/// Records the active renderer mode.
pub(crate) fn set_render_mode(mode: RenderMode) {
    RENDER_MODE.store(mode as u8, Ordering::Relaxed);
    refresh_uptime();
}

/// Records the host initialization state.
pub(crate) fn set_init_state(state: InitState) {
    INIT_STATE.store(state as u8, Ordering::Relaxed);
    refresh_uptime();
}

/// Records the requested target mode.
pub(crate) fn set_target_mode(mode: TargetMode) {
    TARGET_MODE.store(mode as u8, Ordering::Relaxed);
    refresh_uptime();
}

/// Records the last host frame index applied by the scene.
pub(crate) fn set_last_host_frame_index(frame_index: i64) {
    LAST_HOST_FRAME_INDEX.store(frame_index, Ordering::Relaxed);
    refresh_uptime();
}

/// Records the number of views prepared for the next graph execution.
pub(crate) fn set_prepared_view_count(count: usize) {
    let clamped = count.min(u32::MAX as usize) as u32;
    PREPARED_VIEW_COUNT.store(clamped, Ordering::Relaxed);
    refresh_uptime();
}

/// Records current IPC drop streak counters.
pub(crate) fn set_ipc_drop_streaks(primary: u32, background: u32) {
    PRIMARY_IPC_DROP_STREAK.store(primary, Ordering::Relaxed);
    BACKGROUND_IPC_DROP_STREAK.store(background, Ordering::Relaxed);
    refresh_uptime();
}

/// Records the current GPU driver submit backlog.
pub(crate) fn set_driver_backlog(backlog: u64) {
    let clamped = backlog.min(u64::from(u32::MAX)) as u32;
    DRIVER_BACKLOG.store(clamped, Ordering::Relaxed);
    refresh_uptime();
}

/// Records the last render-graph error category.
pub(crate) fn set_last_graph_error(kind: GraphErrorKind) {
    LAST_GRAPH_ERROR.store(kind as u8, Ordering::Relaxed);
    refresh_uptime();
}

/// Records the current GPU driver-thread stage.
pub(crate) fn set_driver_stage(stage: DriverStage) {
    DRIVER_STAGE.store(stage as u8, Ordering::Relaxed);
    refresh_uptime();
}

/// Records the currently active OpenXR call.
pub(crate) fn set_openxr_call(call: OpenXrCall) {
    OPENXR_CALL.store(call as u8, Ordering::Relaxed);
    refresh_uptime();
}

/// Clears the currently active OpenXR call only when it still matches `call`.
pub(crate) fn clear_openxr_call_if(call: OpenXrCall) {
    let _ = OPENXR_CALL.compare_exchange(
        call as u8,
        OpenXrCall::None as u8,
        Ordering::Relaxed,
        Ordering::Relaxed,
    );
    refresh_uptime();
}

/// Records the currently active OpenXR finalize payload.
pub(crate) fn set_xr_finalize_state(
    kind: XrFinalizeKind,
    image_index: Option<u32>,
    frame_seq: u64,
    command_buffers: usize,
    extent: Option<(u32, u32)>,
    predicted_display_time_nanos: Option<i64>,
) {
    XR_FINALIZE_KIND.store(kind as u8, Ordering::Relaxed);
    XR_FINALIZE_IMAGE_INDEX.store(image_index.unwrap_or(NONE_U32), Ordering::Relaxed);
    XR_FINALIZE_FRAME_SEQ.store(frame_seq, Ordering::Relaxed);
    let clamped_command_buffers = command_buffers.min(u32::MAX as usize) as u32;
    XR_FINALIZE_COMMAND_BUFFERS.store(clamped_command_buffers, Ordering::Relaxed);
    let (width, height) = extent.unwrap_or((NONE_U32, NONE_U32));
    XR_FINALIZE_EXTENT_WIDTH.store(width, Ordering::Relaxed);
    XR_FINALIZE_EXTENT_HEIGHT.store(height, Ordering::Relaxed);
    XR_FINALIZE_PREDICTED_TIME_NS.store(
        predicted_display_time_nanos.unwrap_or(NONE_I64),
        Ordering::Relaxed,
    );
    refresh_uptime();
}

/// Clears the active OpenXR finalize payload.
pub(crate) fn clear_xr_finalize_state() {
    set_xr_finalize_state(XrFinalizeKind::None, None, 0, 0, None, None);
}

/// Captures the current crash context from atomics.
pub(crate) fn snapshot() -> CrashContextSnapshot {
    refresh_uptime();
    let xr_image_index = optional_u32_from_atomic(XR_FINALIZE_IMAGE_INDEX.load(Ordering::Relaxed));
    let xr_extent_width = XR_FINALIZE_EXTENT_WIDTH.load(Ordering::Relaxed);
    let xr_extent_height = XR_FINALIZE_EXTENT_HEIGHT.load(Ordering::Relaxed);
    let xr_finalize_extent = optional_extent_from_atomics(xr_extent_width, xr_extent_height);
    let xr_predicted_time =
        optional_i64_from_atomic(XR_FINALIZE_PREDICTED_TIME_NS.load(Ordering::Relaxed));
    CrashContextSnapshot {
        uptime_ms: UPTIME_MS.load(Ordering::Relaxed),
        tick_sequence: TICK_SEQUENCE.load(Ordering::Relaxed),
        tick_phase: TickPhase::from_u8(TICK_PHASE.load(Ordering::Relaxed)),
        cpu_render_phase: CpuRenderPhase::from_u8(CPU_RENDER_PHASE.load(Ordering::Relaxed)),
        render_mode: RenderMode::from_u8(RENDER_MODE.load(Ordering::Relaxed)),
        init_state: InitState::from_u8(INIT_STATE.load(Ordering::Relaxed)),
        target_mode: TargetMode::from_u8(TARGET_MODE.load(Ordering::Relaxed)),
        last_host_frame_index: LAST_HOST_FRAME_INDEX.load(Ordering::Relaxed),
        prepared_view_count: PREPARED_VIEW_COUNT.load(Ordering::Relaxed),
        primary_ipc_drop_streak: PRIMARY_IPC_DROP_STREAK.load(Ordering::Relaxed),
        background_ipc_drop_streak: BACKGROUND_IPC_DROP_STREAK.load(Ordering::Relaxed),
        driver_backlog: DRIVER_BACKLOG.load(Ordering::Relaxed),
        last_graph_error: GraphErrorKind::from_u8(LAST_GRAPH_ERROR.load(Ordering::Relaxed)),
        driver_stage: DriverStage::from_u8(DRIVER_STAGE.load(Ordering::Relaxed)),
        openxr_call: OpenXrCall::from_u8(OPENXR_CALL.load(Ordering::Relaxed)),
        xr_finalize_kind: XrFinalizeKind::from_u8(XR_FINALIZE_KIND.load(Ordering::Relaxed)),
        xr_finalize_image_index: xr_image_index,
        xr_finalize_frame_seq: XR_FINALIZE_FRAME_SEQ.load(Ordering::Relaxed),
        xr_finalize_command_buffers: XR_FINALIZE_COMMAND_BUFFERS.load(Ordering::Relaxed),
        xr_finalize_extent,
        xr_finalize_predicted_display_time_nanos: xr_predicted_time,
    }
}

/// Refreshes the process uptime snapshot when process tracking has started.
pub(super) fn refresh_uptime() {
    if let Some(start) = START_INSTANT.get() {
        let elapsed_ms = start.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
        UPTIME_MS.store(elapsed_ms, Ordering::Relaxed);
    }
}

fn optional_u32_from_atomic(value: u32) -> Option<u32> {
    if value == NONE_U32 { None } else { Some(value) }
}

fn optional_i64_from_atomic(value: i64) -> Option<i64> {
    if value == NONE_I64 { None } else { Some(value) }
}

fn optional_extent_from_atomics(width: u32, height: u32) -> Option<(u32, u32)> {
    if width == NONE_U32 || height == NONE_U32 {
        None
    } else {
        Some((width, height))
    }
}

/// Resets all mutable crash context fields for tests.
#[cfg(test)]
pub(super) fn reset_for_test() {
    UPTIME_MS.store(0, Ordering::Relaxed);
    TICK_SEQUENCE.store(0, Ordering::Relaxed);
    TICK_PHASE.store(TickPhase::Unknown as u8, Ordering::Relaxed);
    CPU_RENDER_PHASE.store(CpuRenderPhase::Unknown as u8, Ordering::Relaxed);
    RENDER_MODE.store(RenderMode::Unknown as u8, Ordering::Relaxed);
    INIT_STATE.store(InitState::NotStarted as u8, Ordering::Relaxed);
    TARGET_MODE.store(TargetMode::Unknown as u8, Ordering::Relaxed);
    LAST_HOST_FRAME_INDEX.store(-1, Ordering::Relaxed);
    PREPARED_VIEW_COUNT.store(0, Ordering::Relaxed);
    PRIMARY_IPC_DROP_STREAK.store(0, Ordering::Relaxed);
    BACKGROUND_IPC_DROP_STREAK.store(0, Ordering::Relaxed);
    DRIVER_BACKLOG.store(0, Ordering::Relaxed);
    LAST_GRAPH_ERROR.store(GraphErrorKind::None as u8, Ordering::Relaxed);
    DRIVER_STAGE.store(DriverStage::Unknown as u8, Ordering::Relaxed);
    OPENXR_CALL.store(OpenXrCall::None as u8, Ordering::Relaxed);
    XR_FINALIZE_KIND.store(XrFinalizeKind::None as u8, Ordering::Relaxed);
    XR_FINALIZE_IMAGE_INDEX.store(NONE_U32, Ordering::Relaxed);
    XR_FINALIZE_FRAME_SEQ.store(0, Ordering::Relaxed);
    XR_FINALIZE_COMMAND_BUFFERS.store(0, Ordering::Relaxed);
    XR_FINALIZE_EXTENT_WIDTH.store(NONE_U32, Ordering::Relaxed);
    XR_FINALIZE_EXTENT_HEIGHT.store(NONE_U32, Ordering::Relaxed);
    XR_FINALIZE_PREDICTED_TIME_NS.store(NONE_I64, Ordering::Relaxed);
}
