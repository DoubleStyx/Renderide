//! Shader permutations for variant-specific WGSL and raster pipeline keys.
//!
//! Production renderers compile **variant-specific** WGSL by baking `#ifdef`-style choices into the
//! source string (or templating) before [`wgpu::Device::create_shader_module`]. [`ShaderPermutation`]
//! selects those static features (e.g. multiview). Cached [`wgpu::RenderPipeline`] instances for
//! materials are owned by [`crate::materials::MaterialPipelineCache`] ([`crate::materials::cache`]),
//! keyed by [`crate::materials::MaterialPipelineCacheKey`] (permutation + surface format + layout).

pub mod raster;

pub use raster::SHADER_PERM_MULTIVIEW_STEREO;

/// Bit flags selecting static shader features (depth-only, alpha clip, multiview stereo, etc.).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ShaderPermutation(pub u32);
