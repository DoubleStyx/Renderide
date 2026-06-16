//! OpenXR crash context categories.

/// Currently active OpenXR call recorded for crash diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum OpenXrCall {
    /// No OpenXR call is currently active.
    None = 0,
    /// `xrPollEvent` loop.
    PollEvents = 1,
    /// Wait for previous deferred finalize signal.
    WaitPreviousFinalize = 2,
    /// `xrWaitFrame`.
    WaitFrame = 3,
    /// `xrBeginFrame`.
    BeginFrame = 4,
    /// `xrLocateViews`.
    LocateViews = 5,
    /// `xrAcquireSwapchainImage`.
    AcquireImage = 6,
    /// `xrWaitSwapchainImage`.
    WaitImage = 7,
    /// `xrReleaseSwapchainImage`.
    ReleaseImage = 8,
    /// Projection `xrEndFrame`.
    EndFrameProjection = 9,
    /// Empty `xrEndFrame`.
    EndFrameEmpty = 10,
}

impl OpenXrCall {
    /// Converts an atomic storage value into an active OpenXR call.
    pub(super) const fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::PollEvents,
            2 => Self::WaitPreviousFinalize,
            3 => Self::WaitFrame,
            4 => Self::BeginFrame,
            5 => Self::LocateViews,
            6 => Self::AcquireImage,
            7 => Self::WaitImage,
            8 => Self::ReleaseImage,
            9 => Self::EndFrameProjection,
            10 => Self::EndFrameEmpty,
            _ => Self::None,
        }
    }

    /// Returns the stable crash-context label for this OpenXR call.
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::PollEvents => "poll-events",
            Self::WaitPreviousFinalize => "wait-previous-finalize",
            Self::WaitFrame => "wait-frame",
            Self::BeginFrame => "begin-frame",
            Self::LocateViews => "locate-views",
            Self::AcquireImage => "acquire-image",
            Self::WaitImage => "wait-image",
            Self::ReleaseImage => "release-image",
            Self::EndFrameProjection => "end-frame-projection",
            Self::EndFrameEmpty => "end-frame-empty",
        }
    }
}

/// Active OpenXR finalize kind recorded for crash diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum XrFinalizeKind {
    /// No OpenXR finalize work is currently active.
    None = 0,
    /// Stereo projection finalize is active.
    Projection = 1,
    /// Empty-frame finalize is active.
    Empty = 2,
}

impl XrFinalizeKind {
    /// Converts an atomic storage value into an OpenXR finalize kind.
    pub(super) const fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Projection,
            2 => Self::Empty,
            _ => Self::None,
        }
    }

    /// Returns the stable crash-context label for this OpenXR finalize kind.
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Projection => "projection",
            Self::Empty => "empty",
        }
    }
}
