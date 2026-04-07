//! Errors returned by the bootstrapper `run` entry point.

use std::fmt;

/// Top-level failure from [`crate::run`].
#[derive(Debug)]
pub enum BootstrapError {
    /// Forwarded I/O error (filesystem, processes, etc.).
    Io(std::io::Error),
    /// Queue option or open failure with context.
    Interprocess(String),
    /// Logging could not be initialized.
    Logging(std::io::Error),
    /// Working directory could not be resolved.
    CurrentDir(std::io::Error),
    /// Shared-memory prefix could not be generated securely.
    Prefix(getrandom::Error),
}

impl fmt::Display for BootstrapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BootstrapError::Io(e) => write!(f, "{e}"),
            BootstrapError::Interprocess(s) => write!(f, "{s}"),
            BootstrapError::Logging(e) => write!(f, "logging: {e}"),
            BootstrapError::CurrentDir(e) => write!(f, "current directory: {e}"),
            BootstrapError::Prefix(e) => write!(f, "prefix generation: {e}"),
        }
    }
}

impl std::error::Error for BootstrapError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            BootstrapError::Io(e) => Some(e),
            BootstrapError::Logging(e) => Some(e),
            BootstrapError::CurrentDir(e) => Some(e),
            BootstrapError::Prefix(_) => None,
            BootstrapError::Interprocess(_) => None,
        }
    }
}

impl From<std::io::Error> for BootstrapError {
    fn from(value: std::io::Error) -> Self {
        BootstrapError::Io(value)
    }
}
