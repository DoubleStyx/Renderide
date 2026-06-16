//! Frame-global reflection-probe specular resources.

use std::sync::Arc;

/// Resources bound to frame-global reflection-probe slots.
#[derive(Clone)]
pub struct ReflectionProbeSpecularResources {
    /// 2D-array atlas view sampled by PBS materials.
    pub array_view: Arc<wgpu::TextureView>,
    /// Sampler paired with [`Self::array_view`].
    pub sampler: Arc<wgpu::Sampler>,
    /// Storage buffer of [`crate::gpu::GpuReflectionProbeMetadata`] rows.
    pub metadata_buffer: Arc<wgpu::Buffer>,
    /// Monotonic resource version; incremented when bindings need recreation.
    pub version: u64,
}
