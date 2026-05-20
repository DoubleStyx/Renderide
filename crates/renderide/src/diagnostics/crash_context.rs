//! Process-wide crash context updated by cold control-flow boundaries.

use std::fmt::Write;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicI64, AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::time::Instant;

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

/// Renderer tick phase recorded for crash diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum TickPhase {
    /// No phase has been recorded yet.
    Unknown = 0,
    /// Process and renderer startup work.
    Startup = 1,
    /// Runtime construction or IPC initialization.
    RuntimeInit = 2,
    /// Host IPC polling.
    IpcPoll = 3,
    /// Host frame submit application.
    FrameSubmit = 4,
    /// Asset/material integration.
    AssetIntegration = 5,
    /// OpenXR frame begin or view location.
    XrBegin = 6,
    /// Lock-step host synchronization.
    Lockstep = 7,
    /// Render-view planning or graph execution.
    RenderViews = 8,
    /// Surface presentation and readback completion.
    Present = 9,
    /// End-of-frame cleanup.
    Epilogue = 10,
    /// Headless runtime tick.
    Headless = 11,
    /// Graceful or fatal shutdown boundary.
    Shutdown = 12,
    /// Start-of-frame bookkeeping.
    Prologue = 13,
}

impl TickPhase {
    fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Startup,
            2 => Self::RuntimeInit,
            3 => Self::IpcPoll,
            4 => Self::FrameSubmit,
            5 => Self::AssetIntegration,
            6 => Self::XrBegin,
            7 => Self::Lockstep,
            8 => Self::RenderViews,
            9 => Self::Present,
            10 => Self::Epilogue,
            11 => Self::Headless,
            12 => Self::Shutdown,
            13 => Self::Prologue,
            _ => Self::Unknown,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Startup => "startup",
            Self::RuntimeInit => "runtime-init",
            Self::IpcPoll => "ipc-poll",
            Self::FrameSubmit => "frame-submit",
            Self::AssetIntegration => "asset-integration",
            Self::XrBegin => "xr-begin",
            Self::Lockstep => "lockstep",
            Self::RenderViews => "render-views",
            Self::Present => "present",
            Self::Epilogue => "epilogue",
            Self::Headless => "headless",
            Self::Shutdown => "shutdown",
            Self::Prologue => "prologue",
        }
    }
}

/// CPU render schedule phase recorded for crash diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum CpuRenderPhase {
    /// No CPU render phase is active.
    Unknown = 0,
    /// Extract immutable frame inputs from runtime state.
    Extract = 1,
    /// Prepare asset and material state needed by this frame.
    AssetPrepare = 2,
    /// Plan the ordered views for this submission.
    ViewPlanning = 3,
    /// Queue visible draw candidates for planned views.
    DrawQueue = 4,
    /// Sort and arrange queued draws into renderable phase order.
    Sort = 5,
    /// Prepare CPU/GPU frame resources before command encoding.
    ResourcePrepare = 6,
    /// Record and submit render-graph commands.
    CommandRecord = 7,
    /// Release frame-local or one-shot CPU render state.
    Cleanup = 8,
}

impl CpuRenderPhase {
    fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Extract,
            2 => Self::AssetPrepare,
            3 => Self::ViewPlanning,
            4 => Self::DrawQueue,
            5 => Self::Sort,
            6 => Self::ResourcePrepare,
            7 => Self::CommandRecord,
            8 => Self::Cleanup,
            _ => Self::Unknown,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Extract => "extract",
            Self::AssetPrepare => "asset-prepare",
            Self::ViewPlanning => "view-planning",
            Self::DrawQueue => "draw-queue",
            Self::Sort => "sort",
            Self::ResourcePrepare => "resource-prepare",
            Self::CommandRecord => "command-record",
            Self::Cleanup => "cleanup",
        }
    }
}

/// Active renderer presentation mode recorded for crash diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum RenderMode {
    /// Mode has not been determined yet.
    Unknown = 0,
    /// Renderer is running without host queues.
    Standalone = 1,
    /// Renderer is consuming host IPC in desktop presentation mode.
    IpcDesktop = 2,
    /// Renderer is consuming host IPC in OpenXR presentation mode.
    IpcOpenXr = 3,
    /// Renderer is running headless.
    Headless = 4,
    /// HMD multiview rendering is active.
    HmdMultiview = 5,
    /// HMD secondary views are being rendered without the main HMD view.
    VrSecondariesOnly = 6,
}

impl RenderMode {
    fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Standalone,
            2 => Self::IpcDesktop,
            3 => Self::IpcOpenXr,
            4 => Self::Headless,
            5 => Self::HmdMultiview,
            6 => Self::VrSecondariesOnly,
            _ => Self::Unknown,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Standalone => "standalone",
            Self::IpcDesktop => "ipc-desktop",
            Self::IpcOpenXr => "ipc-openxr",
            Self::Headless => "headless",
            Self::HmdMultiview => "hmd-multiview",
            Self::VrSecondariesOnly => "vr-secondaries-only",
        }
    }
}

/// Host initialization state recorded for crash diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum InitState {
    /// Renderer startup has not reached host initialization.
    NotStarted = 0,
    /// Renderer is waiting for `RendererInitData`.
    WaitingForInitData = 1,
    /// Renderer received `RendererInitData`.
    InitDataReceived = 2,
    /// Renderer received `InitializationComplete`.
    InitializationComplete = 3,
    /// Renderer has finalized host-driven initialization.
    Finalized = 4,
}

impl InitState {
    fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::WaitingForInitData,
            2 => Self::InitDataReceived,
            3 => Self::InitializationComplete,
            4 => Self::Finalized,
            _ => Self::NotStarted,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::NotStarted => "not-started",
            Self::WaitingForInitData => "waiting-for-init-data",
            Self::InitDataReceived => "init-data-received",
            Self::InitializationComplete => "initialization-complete",
            Self::Finalized => "finalized",
        }
    }
}

/// Requested renderer target mode recorded for crash diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum TargetMode {
    /// Target mode has not been determined.
    Unknown = 0,
    /// Desktop window/surface target.
    Desktop = 1,
    /// OpenXR HMD target.
    OpenXr = 2,
    /// Headless target.
    Headless = 3,
}

impl TargetMode {
    fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Desktop,
            2 => Self::OpenXr,
            3 => Self::Headless,
            _ => Self::Unknown,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Desktop => "desktop",
            Self::OpenXr => "openxr",
            Self::Headless => "headless",
        }
    }
}

/// Last render-graph error category recorded for crash diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum GraphErrorKind {
    /// No graph error has been recorded.
    None = 0,
    /// Graph did not contain frame work.
    NoFrameGraph = 1,
    /// Presentation target acquisition or configuration failed.
    Present = 2,
    /// Depth target setup failed.
    DepthTarget = 3,
    /// Swapchain view was unavailable.
    MissingSwapchainView = 4,
    /// Swapchain rendering required a window that was unavailable.
    SwapchainRequiresWindow = 5,
    /// Graph attachment lookup failed.
    MissingGraphAttachment = 6,
    /// Raster pipeline template lookup failed.
    MissingRasterTemplate = 7,
    /// Render pass execution failed.
    Pass = 8,
    /// Graph execution was asked to process an empty view batch.
    NoViewsInBatch = 9,
    /// Transient pool allocation or aliasing failed.
    TransientPool = 10,
    /// History resource registry failed.
    HistoryRegistry = 11,
    /// Required transient resources were missing.
    MissingTransientResources = 12,
    /// Required per-view resources were missing.
    MissingPerViewResources = 13,
    /// Required history texture was missing.
    MissingHistoryTexture = 14,
    /// Required history buffer was missing.
    MissingHistoryBuffer = 15,
    /// History texture was not allocated.
    UnallocatedHistoryTexture = 16,
    /// History buffer was not allocated.
    UnallocatedHistoryBuffer = 17,
    /// Error type did not map to a narrower category.
    Other = 18,
}

impl GraphErrorKind {
    fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::NoFrameGraph,
            2 => Self::Present,
            3 => Self::DepthTarget,
            4 => Self::MissingSwapchainView,
            5 => Self::SwapchainRequiresWindow,
            6 => Self::MissingGraphAttachment,
            7 => Self::MissingRasterTemplate,
            8 => Self::Pass,
            9 => Self::NoViewsInBatch,
            10 => Self::TransientPool,
            11 => Self::HistoryRegistry,
            12 => Self::MissingTransientResources,
            13 => Self::MissingPerViewResources,
            14 => Self::MissingHistoryTexture,
            15 => Self::MissingHistoryBuffer,
            16 => Self::UnallocatedHistoryTexture,
            17 => Self::UnallocatedHistoryBuffer,
            18 => Self::Other,
            _ => Self::None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::NoFrameGraph => "no-frame-graph",
            Self::Present => "present",
            Self::DepthTarget => "depth-target",
            Self::MissingSwapchainView => "missing-swapchain-view",
            Self::SwapchainRequiresWindow => "swapchain-requires-window",
            Self::MissingGraphAttachment => "missing-graph-attachment",
            Self::MissingRasterTemplate => "missing-raster-template",
            Self::Pass => "pass",
            Self::NoViewsInBatch => "no-views-in-batch",
            Self::TransientPool => "transient-pool",
            Self::HistoryRegistry => "history-registry",
            Self::MissingTransientResources => "missing-transient-resources",
            Self::MissingPerViewResources => "missing-per-view-resources",
            Self::MissingHistoryTexture => "missing-history-texture",
            Self::MissingHistoryBuffer => "missing-history-buffer",
            Self::UnallocatedHistoryTexture => "unallocated-history-texture",
            Self::UnallocatedHistoryBuffer => "unallocated-history-buffer",
            Self::Other => "other",
        }
    }
}

/// Maps a render-graph execution error into the compact crash-context category.
pub(crate) fn graph_error_kind(error: &crate::render_graph::GraphExecuteError) -> GraphErrorKind {
    match error {
        crate::render_graph::GraphExecuteError::NoFrameGraph => GraphErrorKind::NoFrameGraph,
        crate::render_graph::GraphExecuteError::Present(_) => GraphErrorKind::Present,
        crate::render_graph::GraphExecuteError::DepthTarget(_) => GraphErrorKind::DepthTarget,
        crate::render_graph::GraphExecuteError::MissingSwapchainView => {
            GraphErrorKind::MissingSwapchainView
        }
        crate::render_graph::GraphExecuteError::SwapchainRequiresWindow => {
            GraphErrorKind::SwapchainRequiresWindow
        }
        crate::render_graph::GraphExecuteError::MissingGraphAttachment { .. } => {
            GraphErrorKind::MissingGraphAttachment
        }
        crate::render_graph::GraphExecuteError::MissingRasterTemplate { .. } => {
            GraphErrorKind::MissingRasterTemplate
        }
        crate::render_graph::GraphExecuteError::Pass(_) => GraphErrorKind::Pass,
        crate::render_graph::GraphExecuteError::NoViewsInBatch => GraphErrorKind::NoViewsInBatch,
        crate::render_graph::GraphExecuteError::TransientPool(_) => GraphErrorKind::TransientPool,
        crate::render_graph::GraphExecuteError::HistoryRegistry(_) => {
            GraphErrorKind::HistoryRegistry
        }
        crate::render_graph::GraphExecuteError::MissingTransientResources => {
            GraphErrorKind::MissingTransientResources
        }
        crate::render_graph::GraphExecuteError::MissingPerViewResources { .. } => {
            GraphErrorKind::MissingPerViewResources
        }
        crate::render_graph::GraphExecuteError::MissingHistoryTexture { .. } => {
            GraphErrorKind::MissingHistoryTexture
        }
        crate::render_graph::GraphExecuteError::MissingHistoryBuffer { .. } => {
            GraphErrorKind::MissingHistoryBuffer
        }
        crate::render_graph::GraphExecuteError::UnallocatedHistoryTexture { .. } => {
            GraphErrorKind::UnallocatedHistoryTexture
        }
        crate::render_graph::GraphExecuteError::UnallocatedHistoryBuffer { .. } => {
            GraphErrorKind::UnallocatedHistoryBuffer
        }
    }
}

/// Records the active tick phase from an app-driver trace label.
pub(crate) fn set_tick_phase_label(label: &str) {
    let phase = match label {
        "frame_tick_prologue" => TickPhase::Prologue,
        "poll_ipc_and_window" => TickPhase::IpcPoll,
        "lock_step_exchange" => TickPhase::Lockstep,
        "render_views" => TickPhase::RenderViews,
        "present_and_diagnostics" => TickPhase::Present,
        "frame_tick_epilogue" => TickPhase::Epilogue,
        _ => TickPhase::Unknown,
    };
    set_tick_phase(phase);
}

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
}

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

/// Captures the current crash context from atomics.
pub(crate) fn snapshot() -> CrashContextSnapshot {
    refresh_uptime();
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
    }
}

/// Formats the current crash context for ordinary logs and panic reports.
pub(crate) fn format_snapshot() -> String {
    format_snapshot_from(&snapshot())
}

/// Formats a provided crash-context snapshot for tests and higher-level reports.
pub(crate) fn format_snapshot_from(s: &CrashContextSnapshot) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "Renderer crash context: uptime_ms={} tick={} phase={} cpu_phase={} mode={} target={} init={} last_host_frame={} prepared_views={} ipc_drop_streaks=primary:{} background:{} driver_backlog={} last_graph_error={}",
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
        s.last_graph_error.as_str()
    );
    out
}

/// Writes a compact, allocation-free snapshot for native crash handlers.
pub(crate) fn write_minimal_snapshot(out: &mut [u8]) -> usize {
    refresh_uptime();
    let s = snapshot();
    let mut w = 0usize;
    push(out, &mut w, b"CRASH_CONTEXT uptime_ms=");
    push_u64(out, &mut w, s.uptime_ms);
    push(out, &mut w, b" tick=");
    push_u64(out, &mut w, s.tick_sequence);
    push(out, &mut w, b" phase=");
    push(out, &mut w, s.tick_phase.as_str().as_bytes());
    push(out, &mut w, b" cpu_phase=");
    push(out, &mut w, s.cpu_render_phase.as_str().as_bytes());
    push(out, &mut w, b" mode=");
    push(out, &mut w, s.render_mode.as_str().as_bytes());
    push(out, &mut w, b" target=");
    push(out, &mut w, s.target_mode.as_str().as_bytes());
    push(out, &mut w, b" init=");
    push(out, &mut w, s.init_state.as_str().as_bytes());
    push(out, &mut w, b" last_host_frame=");
    push_i64(out, &mut w, s.last_host_frame_index);
    push(out, &mut w, b" prepared_views=");
    push_u64(out, &mut w, u64::from(s.prepared_view_count));
    push(out, &mut w, b" ipc_drop=");
    push_u64(out, &mut w, u64::from(s.primary_ipc_drop_streak));
    push(out, &mut w, b"/");
    push_u64(out, &mut w, u64::from(s.background_ipc_drop_streak));
    push(out, &mut w, b" driver_backlog=");
    push_u64(out, &mut w, u64::from(s.driver_backlog));
    push(out, &mut w, b" graph_error=");
    push(out, &mut w, s.last_graph_error.as_str().as_bytes());
    push(out, &mut w, b"\n");
    w
}

fn refresh_uptime() {
    if let Some(start) = START_INSTANT.get() {
        let elapsed_ms = start.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
        UPTIME_MS.store(elapsed_ms, Ordering::Relaxed);
    }
}

fn push(out: &mut [u8], w: &mut usize, bytes: &[u8]) {
    let remaining = out.len().saturating_sub(*w);
    let n = bytes.len().min(remaining);
    if n > 0 {
        out[*w..*w + n].copy_from_slice(&bytes[..n]);
        *w += n;
    }
}

fn push_u64(out: &mut [u8], w: &mut usize, mut value: u64) {
    if value == 0 {
        push(out, w, b"0");
        return;
    }
    let mut tmp = [0u8; 20];
    let mut i = 0usize;
    while value > 0 {
        tmp[i] = b'0' + (value % 10) as u8;
        value /= 10;
        i += 1;
    }
    while i > 0 {
        i -= 1;
        push(out, w, &tmp[i..=i]);
    }
}

fn push_i64(out: &mut [u8], w: &mut usize, value: i64) {
    if value < 0 {
        push(out, w, b"-");
        push_u64(out, w, value.unsigned_abs());
    } else {
        push_u64(out, w, value as u64);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use parking_lot::Mutex;

    use super::{
        BACKGROUND_IPC_DROP_STREAK, CPU_RENDER_PHASE, CpuRenderPhase, CrashContextSnapshot,
        DRIVER_BACKLOG, GraphErrorKind, INIT_STATE, InitState, LAST_GRAPH_ERROR,
        LAST_HOST_FRAME_INDEX, PREPARED_VIEW_COUNT, PRIMARY_IPC_DROP_STREAK, RENDER_MODE,
        RenderMode, TARGET_MODE, TICK_PHASE, TICK_SEQUENCE, TargetMode, TickPhase, UPTIME_MS,
        format_snapshot_from, set_cpu_render_phase, set_init_state, set_last_graph_error,
        set_last_host_frame_index, set_prepared_view_count, set_render_mode, set_target_mode,
        set_tick_phase, snapshot, write_minimal_snapshot,
    };

    static CRASH_CONTEXT_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn reset_crash_context_for_test() {
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
    }

    fn lock_reset_crash_context() -> parking_lot::MutexGuard<'static, ()> {
        let guard = CRASH_CONTEXT_TEST_LOCK.lock();
        reset_crash_context_for_test();
        guard
    }

    #[test]
    fn snapshot_format_includes_high_value_fields() {
        let s = CrashContextSnapshot {
            uptime_ms: 123,
            tick_sequence: 45,
            tick_phase: TickPhase::RenderViews,
            cpu_render_phase: CpuRenderPhase::CommandRecord,
            render_mode: RenderMode::HmdMultiview,
            init_state: InitState::Finalized,
            target_mode: TargetMode::OpenXr,
            last_host_frame_index: 9001,
            prepared_view_count: 3,
            primary_ipc_drop_streak: 2,
            background_ipc_drop_streak: 1,
            driver_backlog: 4,
            last_graph_error: GraphErrorKind::Pass,
        };
        let line = format_snapshot_from(&s);
        assert!(line.contains("phase=render-views"));
        assert!(line.contains("cpu_phase=command-record"));
        assert!(line.contains("mode=hmd-multiview"));
        assert!(line.contains("target=openxr"));
        assert!(line.contains("init=finalized"));
        assert!(line.contains("last_host_frame=9001"));
        assert!(line.contains("last_graph_error=pass"));
    }

    #[test]
    fn atomics_snapshot_round_trips_core_fields() {
        let _guard = lock_reset_crash_context();

        set_tick_phase(TickPhase::AssetIntegration);
        set_cpu_render_phase(CpuRenderPhase::Sort);
        set_render_mode(RenderMode::IpcDesktop);
        set_target_mode(TargetMode::Desktop);
        set_init_state(InitState::InitializationComplete);
        set_last_host_frame_index(77);
        set_prepared_view_count(2);
        set_last_graph_error(GraphErrorKind::TransientPool);
        let s = snapshot();
        assert_eq!(s.tick_phase, TickPhase::AssetIntegration);
        assert_eq!(s.cpu_render_phase, CpuRenderPhase::Sort);
        assert_eq!(s.render_mode, RenderMode::IpcDesktop);
        assert_eq!(s.target_mode, TargetMode::Desktop);
        assert_eq!(s.init_state, InitState::InitializationComplete);
        assert_eq!(s.last_host_frame_index, 77);
        assert_eq!(s.prepared_view_count, 2);
        assert_eq!(s.last_graph_error, GraphErrorKind::TransientPool);
    }

    #[test]
    fn minimal_snapshot_is_ascii_and_includes_labels() {
        let _guard = lock_reset_crash_context();

        set_tick_phase(TickPhase::Shutdown);
        set_cpu_render_phase(CpuRenderPhase::Cleanup);
        set_render_mode(RenderMode::Headless);
        let mut out = [0u8; 512];
        let n = write_minimal_snapshot(&mut out);
        let line = std::str::from_utf8(&out[..n]).expect("utf8");
        assert!(line.starts_with("CRASH_CONTEXT"));
        assert!(line.contains("phase=shutdown"));
        assert!(line.contains("cpu_phase=cleanup"));
        assert!(line.contains("mode=headless"));
        assert!(line.ends_with('\n'));
    }
}
