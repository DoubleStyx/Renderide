//! Renderer mode crash context categories.

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
    /// Converts an atomic storage value into a render mode.
    pub(super) const fn from_u8(value: u8) -> Self {
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

    /// Returns the stable crash-context label for this render mode.
    pub(super) const fn as_str(self) -> &'static str {
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
    /// Converts an atomic storage value into a host initialization state.
    pub(super) const fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::WaitingForInitData,
            2 => Self::InitDataReceived,
            3 => Self::InitializationComplete,
            4 => Self::Finalized,
            _ => Self::NotStarted,
        }
    }

    /// Returns the stable crash-context label for this initialization state.
    pub(super) const fn as_str(self) -> &'static str {
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
    /// Converts an atomic storage value into a target mode.
    pub(super) const fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Desktop,
            2 => Self::OpenXr,
            3 => Self::Headless,
            _ => Self::Unknown,
        }
    }

    /// Returns the stable crash-context label for this target mode.
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Desktop => "desktop",
            Self::OpenXr => "openxr",
            Self::Headless => "headless",
        }
    }
}
