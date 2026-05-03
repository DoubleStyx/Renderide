//! Errors when building reflective raster mesh pipelines from composed embedded WGSL.

use thiserror::Error;

use super::wgsl_reflect::ReflectError;

/// Failure to load embedded WGSL, reflect bind layouts, or validate the per-draw contract.
#[derive(Debug, Error)]
pub enum PipelineBuildError {
    /// Naga parse/validate or bind layout rules failed for the shader source.
    #[error(transparent)]
    Reflect(#[from] ReflectError),
    /// No embedded `shaders/target/{stem}.wgsl` string in the build output for `stem`.
    #[error("embedded WGSL missing for composed stem `{0}` (run build with shaders/)")]
    MissingEmbeddedShader(String),
    /// Reflective pipeline build was invoked with no pass descriptors (invalid stem or build output).
    #[error("reflective raster pipeline requires at least one pass (stem `{label}`)")]
    EmptyPasses {
        /// Material stem label used in pipeline creation.
        label: &'static str,
    },
    /// wgpu reported a scoped pipeline-build failure (validation/OOM/internal).
    #[error("wgpu pipeline build error: {0}")]
    WgpuScoped(String),
}
