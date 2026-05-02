//! CPU-side texture upload staging / mip layout errors (host bytes -> GPU).

use thiserror::Error;

/// Host layout, dimensions, or format could not be staged for `write_texture` / mip upload.
#[derive(Debug, Clone, Error)]
pub enum TextureUploadError {
    /// Printable failure reason (layout, bounds, format mismatch).
    #[error("{0}")]
    Message(String),
}

impl From<String> for TextureUploadError {
    fn from(msg: String) -> Self {
        Self::Message(msg)
    }
}

impl From<&str> for TextureUploadError {
    fn from(msg: &str) -> Self {
        Self::Message(msg.to_string())
    }
}
