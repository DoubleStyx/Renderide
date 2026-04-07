//! Errors returned when opening queue backing storage or semaphores.

use std::fmt;
use std::io;

/// Error opening shared queue memory or creating the wakeup semaphore.
#[derive(Debug)]
pub struct OpenError(
    /// Underlying OS or I/O error.
    pub io::Error,
);

impl fmt::Display for OpenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for OpenError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.0.source()
    }
}

/// Legacy alias used by earlier call sites.
pub type BackingError = OpenError;
