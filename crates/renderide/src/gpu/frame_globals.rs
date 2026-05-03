//! CPU layout for `shaders/modules/globals.wgsl` (`FrameGlobals` at `@group(0) @binding(0)`).
//!
//! Submodules:
//! - [`uniforms`] -- the [`FrameGpuUniforms`] WGSL-matched Pod struct + per-eye / SH math.
//! - [`skybox_specular`] -- [`SkyboxSpecularUniformParams`] / [`SkyboxSpecularSourceKind`]
//!   for indirect specular sampling.
//! - [`clustered`] -- [`ClusteredFrameGlobalsParams`] input bundle and the
//!   [`FrameGpuUniforms::new_clustered`] constructor.

mod clustered;
mod skybox_specular;
mod uniforms;

#[cfg(test)]
mod tests;

pub use clustered::ClusteredFrameGlobalsParams;
pub use skybox_specular::{SkyboxSpecularSourceKind, SkyboxSpecularUniformParams};
pub use uniforms::FrameGpuUniforms;
