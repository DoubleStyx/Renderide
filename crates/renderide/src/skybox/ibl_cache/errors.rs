//! Error types shared by every IBL bake and convolve path.

use thiserror::Error;

/// Errors returned while preparing an IBL bake.
#[derive(Debug, Error)]
pub(super) enum SkyboxIblBakeError {
    /// Embedded WGSL source was not available at compose time.
    #[error("embedded shader {0} not found")]
    MissingShader(&'static str),
    /// The sliced bake state machine reached an impossible internal state.
    #[error("invalid sliced IBL bake state: {0}")]
    InvalidSlicedBakeState(&'static str),
}

/// Errors returned while encoding GGX convolve mips for an existing cubemap.
#[derive(Debug, Error)]
pub(crate) enum SkyboxIblConvolveError {
    /// Embedded WGSL source was not available at compose time.
    #[error("embedded shader {0} not found")]
    MissingShader(&'static str),
}
