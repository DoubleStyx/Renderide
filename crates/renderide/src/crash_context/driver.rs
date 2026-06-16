//! GPU driver-thread crash context categories.

/// GPU driver-thread stage recorded for crash diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum DriverStage {
    /// No driver stage has been recorded yet.
    Unknown = 0,
    /// Batch was enqueued by the producer.
    Enqueued = 1,
    /// Batch was dropped because the driver thread exited.
    DroppedAfterExit = 2,
    /// Queue submit is about to run.
    SubmitStart = 3,
    /// Queue submit returned.
    SubmitDone = 4,
    /// Surface present is about to run.
    PresentStart = 5,
    /// Surface present returned.
    PresentDone = 6,
    /// OpenXR finalize is about to run.
    XrFinalizeStart = 7,
    /// OpenXR finalize returned.
    XrFinalizeDone = 8,
}

impl DriverStage {
    /// Converts an atomic storage value into a GPU driver-thread stage.
    pub(super) const fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Enqueued,
            2 => Self::DroppedAfterExit,
            3 => Self::SubmitStart,
            4 => Self::SubmitDone,
            5 => Self::PresentStart,
            6 => Self::PresentDone,
            7 => Self::XrFinalizeStart,
            8 => Self::XrFinalizeDone,
            _ => Self::Unknown,
        }
    }

    /// Returns the stable crash-context label for this GPU driver-thread stage.
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Enqueued => "enqueued",
            Self::DroppedAfterExit => "dropped-after-exit",
            Self::SubmitStart => "submit-start",
            Self::SubmitDone => "submit-done",
            Self::PresentStart => "present-start",
            Self::PresentDone => "present-done",
            Self::XrFinalizeStart => "xr-finalize-start",
            Self::XrFinalizeDone => "xr-finalize-done",
        }
    }
}
