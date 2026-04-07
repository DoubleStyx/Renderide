//! Stable keys for **graphics pipeline library** caching: pairs [`PipelineVariant`] with surface
//! format (and future specialization flags) so [`crate::gpu::PipelineDescriptorCache`] can hash a
//! single [`PipelineStateKey`] instead of ad-hoc tag constants per pipeline family.

use std::hash::{Hash, Hasher};

use crate::gpu::PipelineVariant;

/// Identifies a cached `wgpu::RenderPipeline` instance for a given swapchain / pass target set.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct PipelineStateKey {
    /// Logical mesh / material pipeline variant (debug, PBR, native UI, …).
    pub variant: PipelineVariant,
    /// Primary color attachment format (typically the swapchain format).
    pub surface_format: wgpu::TextureFormat,
}

impl PipelineStateKey {
    /// Hash compatible with [`crate::gpu::pipeline_descriptor_cache::PipelineDescriptorCache::builtin_key`].
    pub fn cache_hash(self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        0xB0_u8.hash(&mut h);
        self.variant.hash(&mut h);
        self.surface_format.hash(&mut h);
        h.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::PipelineStateKey;
    use crate::gpu::PipelineVariant;

    #[test]
    fn pipeline_state_key_matches_builtin_descriptor_cache_key() {
        let fmt = wgpu::TextureFormat::Bgra8UnormSrgb;
        let v = PipelineVariant::Pbr;
        let k = PipelineStateKey {
            variant: v,
            surface_format: fmt,
        };
        assert_eq!(
            k.cache_hash(),
            super::super::pipeline_descriptor_cache::PipelineDescriptorCache::builtin_key(v, fmt)
        );
    }
}
