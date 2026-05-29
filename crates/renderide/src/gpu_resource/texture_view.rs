//! Shared cache for compatible `wgpu::TextureView` handles.

use hashbrown::HashMap;
use parking_lot::Mutex;

use super::{AtomicCacheCounters, CacheStats};

/// Compatibility key for a `wgpu::TextureViewDescriptor`.
///
/// Labels are intentionally excluded because they do not affect view compatibility. The parent
/// texture identity and generation are owned by the cache holder, so the key only covers fields
/// that change the view shape or allowed usage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct TextureViewDescriptorKey {
    /// Optional view format override.
    pub(crate) format: Option<wgpu::TextureFormat>,
    /// Optional view dimension override.
    pub(crate) dimension: Option<wgpu::TextureViewDimension>,
    /// Optional view usage mask.
    pub(crate) usage_bits: Option<u32>,
    /// Texture aspect selected by the view.
    pub(crate) aspect: wgpu::TextureAspect,
    /// First mip level in the view.
    pub(crate) base_mip_level: u32,
    /// Optional mip level count.
    pub(crate) mip_level_count: Option<u32>,
    /// First array layer in the view.
    pub(crate) base_array_layer: u32,
    /// Optional array layer count.
    pub(crate) array_layer_count: Option<u32>,
}

impl TextureViewDescriptorKey {
    /// Builds a compatibility key from a concrete descriptor.
    pub(crate) fn from_descriptor(desc: &wgpu::TextureViewDescriptor<'_>) -> Self {
        Self {
            format: desc.format,
            dimension: desc.dimension,
            usage_bits: desc.usage.map(|usage| usage.bits()),
            aspect: desc.aspect,
            base_mip_level: desc.base_mip_level,
            mip_level_count: desc.mip_level_count,
            base_array_layer: desc.base_array_layer,
            array_layer_count: desc.array_layer_count,
        }
    }

    /// Reconstructs a descriptor for creating a cached view.
    pub(crate) fn to_descriptor<'a>(
        self,
        label: Option<&'a str>,
    ) -> wgpu::TextureViewDescriptor<'a> {
        wgpu::TextureViewDescriptor {
            label,
            format: self.format,
            dimension: self.dimension,
            usage: self.usage_bits.map(wgpu::TextureUsages::from_bits_retain),
            aspect: self.aspect,
            base_mip_level: self.base_mip_level,
            mip_level_count: self.mip_level_count,
            base_array_layer: self.base_array_layer,
            array_layer_count: self.array_layer_count,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct TextureViewCacheKey {
    resource_generation: u64,
    descriptor: TextureViewDescriptorKey,
}

/// Per-texture compatible view cache.
#[derive(Debug, Default)]
pub(crate) struct TextureViewCache {
    entries: Mutex<HashMap<TextureViewCacheKey, wgpu::TextureView>>,
    stats: AtomicCacheCounters,
}

impl TextureViewCache {
    /// Returns a cached texture view for `key`, creating it from `texture` on miss.
    pub(crate) fn get_or_create(
        &self,
        texture: &wgpu::Texture,
        resource_generation: u64,
        descriptor: TextureViewDescriptorKey,
        label: Option<&str>,
    ) -> (wgpu::TextureView, bool) {
        profiling::scope!("gpu_resource::texture_view_cache_lookup");
        let key = TextureViewCacheKey {
            resource_generation,
            descriptor,
        };
        {
            let entries = self.entries.lock();
            if let Some(view) = entries.get(&key).cloned() {
                drop(entries);
                profiling::scope!("gpu_resource::texture_view_cache_hit");
                self.stats.note_hit();
                return (view, false);
            }
        }

        profiling::scope!("gpu_resource::texture_view_cache_miss");
        self.stats.note_miss();
        let view = texture.create_view(&descriptor.to_descriptor(label));
        let mut entries = self.entries.lock();
        if let Some(existing) = entries.get(&key).cloned() {
            drop(entries);
            self.stats.note_hit();
            return (existing, false);
        }
        entries.insert(key, view.clone());
        drop(entries);
        self.stats.note_insertion();
        (view, true)
    }

    /// Clears cached views and records evictions for diagnostics.
    pub(crate) fn clear(&self) -> usize {
        let mut entries = self.entries.lock();
        let dropped = entries.len();
        entries.clear();
        drop(entries);
        for _ in 0..dropped {
            self.stats.note_eviction();
        }
        dropped
    }

    /// Returns the current entry count.
    pub(crate) fn len(&self) -> usize {
        self.entries.lock().len()
    }

    /// Returns a point-in-time counter snapshot.
    pub(crate) fn stats(&self) -> CacheStats {
        self.stats.snapshot()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_descriptor() -> wgpu::TextureViewDescriptor<'static> {
        wgpu::TextureViewDescriptor {
            label: Some("ignored"),
            format: Some(wgpu::TextureFormat::Rgba16Float),
            dimension: Some(wgpu::TextureViewDimension::D2),
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        }
    }

    #[test]
    fn descriptor_key_ignores_label() {
        let mut a = base_descriptor();
        let mut b = base_descriptor();
        a.label = Some("a");
        b.label = Some("b");

        assert_eq!(
            TextureViewDescriptorKey::from_descriptor(&a),
            TextureViewDescriptorKey::from_descriptor(&b)
        );
    }

    #[test]
    fn descriptor_key_tracks_compatible_view_shape() {
        let base = TextureViewDescriptorKey::from_descriptor(&base_descriptor());

        let mut changed = base_descriptor();
        changed.dimension = Some(wgpu::TextureViewDimension::D2Array);
        assert_ne!(base, TextureViewDescriptorKey::from_descriptor(&changed));

        let mut changed = base_descriptor();
        changed.aspect = wgpu::TextureAspect::DepthOnly;
        assert_ne!(base, TextureViewDescriptorKey::from_descriptor(&changed));

        let mut changed = base_descriptor();
        changed.base_mip_level = 1;
        assert_ne!(base, TextureViewDescriptorKey::from_descriptor(&changed));

        let mut changed = base_descriptor();
        changed.base_array_layer = 1;
        assert_ne!(base, TextureViewDescriptorKey::from_descriptor(&changed));
    }

    #[test]
    fn cache_key_includes_resource_generation() {
        let descriptor = TextureViewDescriptorKey::from_descriptor(&base_descriptor());
        assert_ne!(
            TextureViewCacheKey {
                resource_generation: 1,
                descriptor
            },
            TextureViewCacheKey {
                resource_generation: 2,
                descriptor
            }
        );
    }
}
