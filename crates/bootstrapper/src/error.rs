//! Errors returned by the bootstrapper `run` entry point.

use interprocess::OpenError;
use thiserror::Error;

/// Top-level failure from [`crate::run`].
#[derive(Debug, Error)]
pub enum BootstrapError {
    /// Forwarded I/O error (filesystem, processes, etc.).
    #[error("io: {0}")]
    Io(
        #[from]
        #[source]
        std::io::Error,
    ),
    /// Invalid queue capacity, alignment, or other [`interprocess::QueueOptions`] validation failure.
    #[error("queue options: {0}")]
    QueueOptionsInvalid(String),
    /// Opening shared queue memory or creating the wakeup semaphore failed.
    #[error(transparent)]
    InterprocessOpen(#[from] OpenError),
    /// Logging could not be initialized.
    #[error("logging: {0}")]
    Logging(#[source] std::io::Error),
    /// Working directory could not be resolved.
    #[error("current directory: {0}")]
    CurrentDir(#[source] std::io::Error),
    /// Shared-memory prefix could not be generated securely.
    #[error("prefix generation: {0}")]
    Prefix(#[source] getrandom::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn display_io_forwards_message() {
        let inner = std::io::Error::new(std::io::ErrorKind::NotFound, "missing");
        let e = BootstrapError::Io(inner);
        assert!(e.to_string().contains("missing"));
        assert!(e.to_string().starts_with("io:"));
    }

    #[test]
    fn display_queue_options_invalid() {
        let e = BootstrapError::QueueOptionsInvalid("capacity too small".to_string());
        assert!(e.to_string().contains("queue options"));
        assert!(e.to_string().contains("capacity too small"));
    }

    #[test]
    fn display_logging_prefix() {
        let inner = std::io::Error::other("disk full");
        let e = BootstrapError::Logging(inner);
        assert!(e.to_string().contains("logging"));
        assert!(e.to_string().contains("disk full"));
    }

    #[test]
    fn display_current_dir_prefix() {
        let inner = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let e = BootstrapError::CurrentDir(inner);
        assert!(e.to_string().contains("current directory"));
    }

    #[test]
    fn display_interprocess_open() {
        let inner = std::io::Error::other("mapping failed");
        let e = BootstrapError::InterprocessOpen(OpenError(inner));
        assert_eq!(e.to_string(), "mapping failed");
    }

    #[test]
    fn from_io_error_yields_io_variant() {
        let io = std::io::Error::other("x");
        let e: BootstrapError = io.into();
        assert!(matches!(e, BootstrapError::Io(_)));
    }

    #[test]
    fn from_open_error_yields_interprocess_variant() {
        let o = OpenError(std::io::Error::other("open"));
        let e: BootstrapError = o.into();
        assert!(matches!(e, BootstrapError::InterprocessOpen(_)));
    }

    #[test]
    fn error_source_io_variants() {
        let io = std::io::Error::other("x");
        assert!(BootstrapError::Io(io).source().is_some());
        let io2 = std::io::Error::other("y");
        assert!(BootstrapError::Logging(io2).source().is_some());
    }
}
