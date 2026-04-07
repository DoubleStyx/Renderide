//! Shader permutations and pipeline caching (skeleton).
//!
//! Production renderers compile **variant-specific** WGSL by baking `#ifdef`-style choices into the
//! source string (or templating) before [`wgpu::Device::create_shader_module`], then key pipelines by
//! a [`PipelineKey`] that hashes permutation + format state. This avoids dynamic branch cost in
//! shaders and avoids a second offline compiler when WGSL is the authoring format.

pub mod raster;

pub use raster::SHADER_PERM_MULTIVIEW_STEREO;

use std::collections::HashMap;
use std::sync::Arc;

use wgpu::TextureFormat;

/// Bit flags selecting static shader features (depth-only, alpha clip, etc.).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ShaderPermutation(pub u32);

/// Cache key for [`PipelineLibrary`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PipelineKey {
    pub permutation: ShaderPermutation,
    pub surface_format: TextureFormat,
}

/// Lazily built pipeline cache; extend with `get_or_create` when draws exist.
#[derive(Debug)]
pub struct PipelineLibrary {
    device: Arc<wgpu::Device>,
    pipelines: HashMap<PipelineKey, wgpu::RenderPipeline>,
}

impl PipelineLibrary {
    /// Creates an empty library bound to `device`.
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        Self {
            device,
            pipelines: HashMap::new(),
        }
    }

    /// Returns the cache entry if a pipeline was built for `key`.
    pub fn get(&self, key: &PipelineKey) -> Option<&wgpu::RenderPipeline> {
        self.pipelines.get(key)
    }

    /// Logical device used when creating new pipelines.
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    /// Inserts a pre-built pipeline (used by tests or future loaders).
    pub fn insert(&mut self, key: PipelineKey, pipeline: wgpu::RenderPipeline) {
        self.pipelines.insert(key, pipeline);
    }

    /// Builds WGSL by applying [`ShaderPermutation`] bits to a template (placeholder).
    pub fn wgsl_from_template(template: &str, _permutation: ShaderPermutation) -> String {
        template.to_string()
    }
}
