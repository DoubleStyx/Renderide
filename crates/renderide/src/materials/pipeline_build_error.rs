//! Errors when building reflective raster mesh pipelines from runtime shader package WGSL.

use thiserror::Error;

use super::wgsl_reflect::ReflectError;

/// Failure to load packaged WGSL, reflect bind layouts, or validate the per-draw contract.
#[derive(Debug, Error)]
pub enum PipelineBuildError {
    /// Naga parse/validate or bind layout rules failed for the shader source.
    #[error(transparent)]
    Reflect(#[from] ReflectError),
    /// No runtime shader package WGSL payload for `stem`.
    #[error("shader package WGSL missing for composed stem `{0}` (run build with shaders/)")]
    MissingEmbeddedShader(String),
    /// The active device does not expose all features required by the shader package target.
    #[error("shader package WGSL `{stem}` requires unavailable device features {missing:?}")]
    MissingDeviceFeatures {
        /// Composed material target stem that declared the feature requirement.
        stem: String,
        /// Required feature bits absent from the active device.
        missing: wgpu::Features,
    },
    /// Reflective pipeline build was invoked with no pass descriptors (invalid stem or build output).
    #[error("reflective raster pipeline requires at least one pass (stem `{label}`)")]
    EmptyPasses {
        /// Material stem label used in pipeline creation.
        label: String,
    },
}
