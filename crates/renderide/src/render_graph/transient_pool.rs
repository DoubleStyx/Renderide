//! Transient resource pool metadata and reuse policy.
//!
//! The runtime allocator can attach actual `wgpu` objects to these entries; the keying, hit/miss,
//! and garbage-collection behavior is deterministic and unit-testable without a GPU.

use std::collections::HashMap;

use super::resources::{BufferSizePolicy, TransientExtent};

/// Concrete texture allocation key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureKey {
    /// Texture format.
    pub format: wgpu::TextureFormat,
    /// Resolved extent policy.
    pub extent: TransientExtent,
    /// Mip count.
    pub mip_levels: u32,
    /// Sample count.
    pub sample_count: u32,
    /// Texture dimension.
    pub dimension: wgpu::TextureDimension,
    /// Array-layer count.
    pub array_layers: u32,
    /// Usage bitset.
    pub usage_bits: u64,
}

/// Concrete buffer allocation key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferKey {
    /// Size policy.
    pub size_policy: BufferSizePolicy,
    /// Usage bitset.
    pub usage_bits: u64,
}

/// Lightweight texture-pool entry.
#[derive(Debug)]
struct PooledTexture {
    key: TextureKey,
    texture: Option<wgpu::Texture>,
    view: Option<wgpu::TextureView>,
    last_used_gen: u64,
}

/// Lightweight buffer-pool entry.
#[derive(Debug)]
struct PooledBuffer {
    key: BufferKey,
    buffer: Option<wgpu::Buffer>,
    size: u64,
    last_used_gen: u64,
}

/// Runtime texture borrowed from the transient pool by handle clone.
#[derive(Debug)]
pub struct PooledTextureLease {
    /// Pool entry id to release after the frame.
    pub pool_id: usize,
    /// Texture handle.
    pub texture: wgpu::Texture,
    /// Default full-resource texture view.
    pub view: wgpu::TextureView,
}

/// Runtime buffer borrowed from the transient pool by handle clone.
#[derive(Debug)]
pub struct PooledBufferLease {
    /// Pool entry id to release after the frame.
    pub pool_id: usize,
    /// Buffer handle.
    pub buffer: wgpu::Buffer,
    /// Buffer size in bytes.
    pub size: u64,
}

/// Pool statistics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TransientPoolMetrics {
    /// Texture reuse hits.
    pub texture_hits: usize,
    /// Texture allocation misses.
    pub texture_misses: usize,
    /// Buffer reuse hits.
    pub buffer_hits: usize,
    /// Buffer allocation misses.
    pub buffer_misses: usize,
    /// Texture entries currently retained.
    pub retained_textures: usize,
    /// Buffer entries currently retained.
    pub retained_buffers: usize,
}

/// Transient pool metadata. Actual GPU allocation is layered on top of this key map.
#[derive(Debug, Default)]
pub struct TransientPool {
    textures: Vec<PooledTexture>,
    buffers: Vec<PooledBuffer>,
    free_textures: HashMap<TextureKey, Vec<usize>>,
    free_buffers: HashMap<BufferKey, Vec<usize>>,
    lru_gen: u64,
    metrics: TransientPoolMetrics,
}

impl TransientPool {
    /// Creates an empty pool.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current generation.
    pub fn generation(&self) -> u64 {
        self.lru_gen
    }

    /// Marks a new frame/generation.
    pub fn begin_generation(&mut self) {
        self.lru_gen = self.lru_gen.saturating_add(1);
    }

    /// Acquires a texture entry id for `key`, reusing a matching free entry when available.
    pub fn acquire_texture(&mut self, key: TextureKey) -> usize {
        if let Some(list) = self.free_textures.get_mut(&key) {
            if let Some(id) = list.pop() {
                self.metrics.texture_hits += 1;
                self.textures[id].last_used_gen = self.lru_gen;
                return id;
            }
        }
        let id = self.textures.len();
        self.textures.push(PooledTexture {
            key,
            texture: None,
            view: None,
            last_used_gen: self.lru_gen,
        });
        self.metrics.texture_misses += 1;
        id
    }

    /// Acquires a real GPU texture entry for `key`, allocating on a miss.
    pub fn acquire_texture_resource(
        &mut self,
        device: &wgpu::Device,
        key: TextureKey,
        label: &'static str,
        usage: wgpu::TextureUsages,
    ) -> PooledTextureLease {
        if let Some(list) = self.free_textures.get_mut(&key) {
            if let Some(id) = list.pop() {
                self.metrics.texture_hits += 1;
                self.textures[id].last_used_gen = self.lru_gen;
                if self.textures[id].texture.is_none() {
                    let (texture, view) = create_texture_and_view(device, key, label, usage);
                    self.textures[id].texture = Some(texture);
                    self.textures[id].view = Some(view);
                }
                return texture_lease_from_entry(id, &self.textures[id]);
            }
        }

        let (texture, view) = create_texture_and_view(device, key, label, usage);
        let id = self.textures.len();
        self.textures.push(PooledTexture {
            key,
            texture: Some(texture),
            view: Some(view),
            last_used_gen: self.lru_gen,
        });
        self.metrics.texture_misses += 1;
        texture_lease_from_entry(id, &self.textures[id])
    }

    /// Releases a texture entry back to the matching-key free list.
    pub fn release_texture(&mut self, id: usize) {
        if let Some(entry) = self.textures.get(id) {
            self.free_textures.entry(entry.key).or_default().push(id);
        }
    }

    /// Acquires a buffer entry id for `key`, reusing a matching free entry when available.
    pub fn acquire_buffer(&mut self, key: BufferKey) -> usize {
        if let Some(list) = self.free_buffers.get_mut(&key) {
            if let Some(id) = list.pop() {
                self.metrics.buffer_hits += 1;
                self.buffers[id].last_used_gen = self.lru_gen;
                return id;
            }
        }
        let id = self.buffers.len();
        self.buffers.push(PooledBuffer {
            key,
            buffer: None,
            size: 0,
            last_used_gen: self.lru_gen,
        });
        self.metrics.buffer_misses += 1;
        id
    }

    /// Acquires a real GPU buffer entry for `key`, allocating on a miss.
    pub fn acquire_buffer_resource(
        &mut self,
        device: &wgpu::Device,
        key: BufferKey,
        label: &'static str,
        usage: wgpu::BufferUsages,
        size: u64,
    ) -> PooledBufferLease {
        if let Some(list) = self.free_buffers.get_mut(&key) {
            if let Some(id) = list.pop() {
                self.metrics.buffer_hits += 1;
                self.buffers[id].last_used_gen = self.lru_gen;
                if self.buffers[id].buffer.is_none() || self.buffers[id].size != size {
                    self.buffers[id].buffer = Some(create_buffer(device, label, usage, size));
                    self.buffers[id].size = size;
                }
                return buffer_lease_from_entry(id, &self.buffers[id]);
            }
        }

        let id = self.buffers.len();
        self.buffers.push(PooledBuffer {
            key,
            buffer: Some(create_buffer(device, label, usage, size)),
            size,
            last_used_gen: self.lru_gen,
        });
        self.metrics.buffer_misses += 1;
        buffer_lease_from_entry(id, &self.buffers[id])
    }

    /// Releases a buffer entry back to the matching-key free list.
    pub fn release_buffer(&mut self, id: usize) {
        if let Some(entry) = self.buffers.get(id) {
            self.free_buffers.entry(entry.key).or_default().push(id);
        }
    }

    /// Releases entries that have not been used for more than `max_age` generations.
    pub fn gc_tick(&mut self, max_age: u64) {
        let current = self.lru_gen;
        let mut texture_alive = vec![false; self.textures.len()];
        for (idx, entry) in self.textures.iter().enumerate() {
            texture_alive[idx] = current.saturating_sub(entry.last_used_gen) <= max_age;
        }
        let mut buffer_alive = vec![false; self.buffers.len()];
        for (idx, entry) in self.buffers.iter().enumerate() {
            buffer_alive[idx] = current.saturating_sub(entry.last_used_gen) <= max_age;
        }
        for list in self.free_textures.values_mut() {
            list.retain(|&id| texture_alive.get(id).copied().unwrap_or(false));
        }
        for list in self.free_buffers.values_mut() {
            list.retain(|&id| buffer_alive.get(id).copied().unwrap_or(false));
        }
        self.metrics.retained_textures = texture_alive.into_iter().filter(|alive| *alive).count();
        self.metrics.retained_buffers = buffer_alive.into_iter().filter(|alive| *alive).count();
    }

    /// Returns current metrics.
    pub fn metrics(&self) -> TransientPoolMetrics {
        TransientPoolMetrics {
            retained_textures: self.textures.len(),
            retained_buffers: self.buffers.len(),
            ..self.metrics
        }
    }
}

fn texture_lease_from_entry(id: usize, entry: &PooledTexture) -> PooledTextureLease {
    PooledTextureLease {
        pool_id: id,
        texture: entry
            .texture
            .as_ref()
            .expect("runtime texture entry has texture")
            .clone(),
        view: entry
            .view
            .as_ref()
            .expect("runtime texture entry has view")
            .clone(),
    }
}

fn buffer_lease_from_entry(id: usize, entry: &PooledBuffer) -> PooledBufferLease {
    PooledBufferLease {
        pool_id: id,
        buffer: entry
            .buffer
            .as_ref()
            .expect("runtime buffer entry has buffer")
            .clone(),
        size: entry.size,
    }
}

fn create_texture_and_view(
    device: &wgpu::Device,
    key: TextureKey,
    label: &'static str,
    usage: wgpu::TextureUsages,
) -> (wgpu::Texture, wgpu::TextureView) {
    let (width, height, layers) = match key.extent {
        TransientExtent::Backbuffer => (1, 1, key.array_layers),
        TransientExtent::Custom { width, height } => (width, height, key.array_layers),
        TransientExtent::MultiLayer {
            width,
            height,
            layers,
        } => (width, height, layers),
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: layers.max(1),
        },
        mip_level_count: key.mip_levels.max(1),
        sample_count: key.sample_count.max(1),
        dimension: key.dimension,
        format: key.format,
        usage,
        view_formats: &[],
    });
    let view = if key.dimension == wgpu::TextureDimension::D2 && layers > 1 {
        texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(label),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            array_layer_count: Some(layers.max(1)),
            ..Default::default()
        })
    } else {
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    };
    (texture, view)
}

fn create_buffer(
    device: &wgpu::Device,
    label: &'static str,
    usage: wgpu::BufferUsages,
    size: u64,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(1),
        usage,
        mapped_at_creation: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tex_key(usage: wgpu::TextureUsages, width: u32, height: u32) -> TextureKey {
        TextureKey {
            format: wgpu::TextureFormat::Rgba8Unorm,
            extent: TransientExtent::Custom { width, height },
            mip_levels: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            array_layers: 1,
            usage_bits: usage.bits() as u64,
        }
    }

    fn buf_key(usage: wgpu::BufferUsages) -> BufferKey {
        BufferKey {
            size_policy: BufferSizePolicy::Fixed(256),
            usage_bits: usage.bits() as u64,
        }
    }

    #[test]
    fn matching_descriptor_reuses_physical_texture() {
        let mut pool = TransientPool::new();
        let key = tex_key(wgpu::TextureUsages::RENDER_ATTACHMENT, 64, 64);
        let first = pool.acquire_texture(key);
        pool.release_texture(first);
        let second = pool.acquire_texture(key);
        assert_eq!(first, second);
        assert_eq!(pool.metrics().texture_hits, 1);
    }

    #[test]
    fn different_usages_produce_different_slots() {
        let mut pool = TransientPool::new();
        let a = pool.acquire_texture(tex_key(wgpu::TextureUsages::RENDER_ATTACHMENT, 64, 64));
        pool.release_texture(a);
        let b = pool.acquire_texture(tex_key(wgpu::TextureUsages::TEXTURE_BINDING, 64, 64));
        assert_ne!(a, b);
    }

    #[test]
    fn gc_tick_releases_after_threshold_generations() {
        let mut pool = TransientPool::new();
        let key = tex_key(wgpu::TextureUsages::RENDER_ATTACHMENT, 64, 64);
        let id = pool.acquire_texture(key);
        pool.release_texture(id);
        pool.begin_generation();
        pool.begin_generation();
        pool.gc_tick(0);
        let next = pool.acquire_texture(key);
        assert_ne!(id, next);
    }

    #[test]
    fn viewport_resize_invalidates_backbuffer_sized_slots() {
        let mut pool = TransientPool::new();
        let a_key = tex_key(wgpu::TextureUsages::RENDER_ATTACHMENT, 64, 64);
        let b_key = tex_key(wgpu::TextureUsages::RENDER_ATTACHMENT, 128, 64);
        let a = pool.acquire_texture(a_key);
        pool.release_texture(a);
        let b = pool.acquire_texture(b_key);
        assert_ne!(a, b);
    }

    #[test]
    fn pool_metrics_count_hits_and_misses() {
        let mut pool = TransientPool::new();
        let t_key = tex_key(wgpu::TextureUsages::RENDER_ATTACHMENT, 64, 64);
        let b_key = buf_key(wgpu::BufferUsages::COPY_DST);
        let t = pool.acquire_texture(t_key);
        pool.release_texture(t);
        let _ = pool.acquire_texture(t_key);
        let b = pool.acquire_buffer(b_key);
        pool.release_buffer(b);
        let _ = pool.acquire_buffer(b_key);
        let metrics = pool.metrics();
        assert_eq!(metrics.texture_misses, 1);
        assert_eq!(metrics.texture_hits, 1);
        assert_eq!(metrics.buffer_misses, 1);
        assert_eq!(metrics.buffer_hits, 1);
    }
}
