//! Errors returned when opening queue backing storage or semaphores.

use std::io;

use thiserror::Error;

/// Error opening shared queue memory or creating the wakeup semaphore.
///
/// Wraps [`std::io::Error`] so filesystem, mapping, and semaphore failures share one surface type.
#[derive(Debug, Error)]
#[error(transparent)]
pub struct OpenError(
    /// Underlying I/O error from the file mapping or semaphore syscall.
    #[from]
    pub io::Error,
);

#[cfg(test)]
mod tests {
    use std::io;

    use super::*;

    #[test]
    fn open_error_display_forwards_io_message() {
        let inner = io::Error::new(io::ErrorKind::NotFound, "no mapping");
        let e = OpenError(inner);
        assert_eq!(e.to_string(), "no mapping");
    }

    #[test]
    fn open_error_preserves_io_kind() {
        let inner = io::Error::new(io::ErrorKind::PermissionDenied, "denied");
        let e = OpenError(inner);
        assert_eq!(e.0.kind(), io::ErrorKind::PermissionDenied);
    }

    #[test]
    fn open_error_from_io() {
        let inner = io::Error::other("x");
        let e: OpenError = inner.into();
        assert_eq!(e.to_string(), "x");
    }
}
