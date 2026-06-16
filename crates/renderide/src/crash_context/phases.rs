//! Frame and CPU render phase crash context categories.

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
    /// Converts an atomic storage value into a tick phase.
    pub(super) const fn from_u8(value: u8) -> Self {
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

    /// Returns the stable crash-context label for this tick phase.
    pub(super) const fn as_str(self) -> &'static str {
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
    /// Converts an atomic storage value into a CPU render phase.
    pub(super) const fn from_u8(value: u8) -> Self {
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

    /// Returns the stable crash-context label for this CPU render phase.
    pub(super) const fn as_str(self) -> &'static str {
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

/// Converts an app-driver trace label into a tick phase.
pub(super) fn tick_phase_from_label(label: &str) -> TickPhase {
    match label {
        "frame_tick_prologue" => TickPhase::Prologue,
        "poll_ipc_and_window" => TickPhase::IpcPoll,
        "lock_step_exchange" => TickPhase::Lockstep,
        "render_views" => TickPhase::RenderViews,
        "present_and_diagnostics" => TickPhase::Present,
        "frame_tick_epilogue" => TickPhase::Epilogue,
        _ => TickPhase::Unknown,
    }
}
