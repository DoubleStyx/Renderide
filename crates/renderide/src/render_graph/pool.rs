//! Transient resource pool: keying, hit/miss accounting, free-list reuse, and GC.
//!
//! [`TransientPool`] is the public face used by the executor. Internally it composes two
//! instances of the generic [`Pool`] -- one keyed by [`TextureKey`] (storing
//! [`policy::TextureSlotValue`]) and one keyed by [`BufferKey`] (storing
//! [`policy::BufferSlotValue`]). The shared LRU + free-list bookkeeping lives on [`Pool`]; the
//! GPU-resource construction, validation against device limits, and lease wrapping live on
//! [`TransientPool`] methods.

mod leases;
mod metrics;
mod policy;
mod validate;

pub use metrics::TransientPoolMetrics;
pub use policy::{BufferKey, TextureKey};

use leases::{buffer_lease_from_entry, saturating_usize, texture_lease_from_entry};
use validate::validate_texture_key;

use hashbrown::HashMap;

use crate::gpu_resource::{CacheCounters, CacheStats, TextureViewCache};
use std::sync::Arc;

use policy::{
    BufferKind, BufferSlotValue, PoolKind, TextureKind, TextureSlotValue, create_buffer,
    create_texture_and_view,
};

/// Failure to build a [`PooledTextureLease`] or [`PooledBufferLease`] from pool entries.
#[derive(Debug, thiserror::Error)]
pub enum TransientPoolError {
    /// GPU texture or view was not attached before leasing.
    #[error("transient pool texture entry {pool_id} missing GPU texture or view")]
    MissingTextureResources {
        /// Pool index.
        pool_id: usize,
    },
    /// GPU buffer was not attached before leasing.
    #[error("transient pool buffer entry {pool_id} missing GPU buffer")]
    MissingBuffer {
        /// Pool index.
        pool_id: usize,
    },
    /// Requested texture dimensions exceed device limits.
    #[error(
        "transient texture {label} {width}x{height}x{layers} mips={mip_levels} exceeds device limits"
    )]
    TextureExceedsLimits {
        /// Texture label.
        label: &'static str,
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
        /// Array layer count (or depth for 3D).
        layers: u32,
        /// Mip level count.
        mip_levels: u32,
    },
    /// Requested texture format/usage combination is unsupported by this device.
    #[error(
        "transient texture {label} format {format:?} usage {usage:?} is unsupported by this device"
    )]
    TextureUnsupportedUsage {
        /// Texture label.
        label: &'static str,
        /// Texture format.
        format: wgpu::TextureFormat,
        /// Requested usage bits.
        usage: wgpu::TextureUsages,
    },
    /// Requested texture sample count is unsupported by this device for the texture format.
    #[error(
        "transient texture {label} format {format:?} sample_count={sample_count} is unsupported by this device"
    )]
    TextureUnsupportedSampleCount {
        /// Texture label.
        label: &'static str,
        /// Texture format.
        format: wgpu::TextureFormat,
        /// Requested sample count.
        sample_count: u32,
    },
    /// Requested buffer size exceeds device limits.
    #[error("transient buffer {label} size={size} exceeds device limits (max_buffer_size={max})")]
    BufferExceedsLimits {
        /// Buffer label.
        label: &'static str,
        /// Requested size in bytes.
        size: u64,
        /// Device cap (`max_buffer_size`).
        max: u64,
    },
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
    /// Compatible derived-view cache owned by the transient pool slot.
    pub view_cache: Arc<TextureViewCache>,
    /// Resource generation for the backing texture.
    pub resource_generation: u64,
}

/// Runtime buffer borrowed from the transient pool by handle clone.
#[derive(Debug)]
pub struct PooledBufferLease {
    /// Pool entry id to release after the frame.
    pub pool_id: usize,
    /// Buffer handle.
    pub _buffer: wgpu::Buffer,
    /// Buffer size in bytes.
    pub _size: u64,
}

/// One pool slot: the alias key, the cached value, and the generation when it was last used.
#[derive(Debug)]
struct Entry<P: PoolKind> {
    key: P::Key,
    value: P::Value,
    last_used_generation: u64,
}

/// Generic LRU + free-list pool keyed by `P::Key` over values of type `P::Value`.
///
/// On [`Self::acquire`] the pool prefers an existing free entry whose key matches; on a miss it
/// pushes a new entry built by the caller. [`Self::gc`] drops cached values for entries that
/// have been inactive for more than `max_age` generations.
#[derive(Debug)]
struct Pool<P: PoolKind> {
    entries: Vec<Entry<P>>,
    free: HashMap<P::Key, Vec<usize>>,
    counters: CacheCounters,
}

impl<P: PoolKind> Default for Pool<P> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            free: HashMap::new(),
            counters: CacheCounters::default(),
        }
    }
}

impl<P: PoolKind> Pool<P> {
    /// Returns an entry id for `key`, reusing a free slot when one matches and is fresh enough.
    /// `refresh_if_stale` is called on a hit so the caller can re-attach GPU resources that the
    /// last GC tick dropped or that no longer match the desired allocation parameters.
    /// `create_value` is called on a miss to populate a fresh slot.
    fn acquire(
        &mut self,
        key: P::Key,
        generation: u64,
        mut refresh_if_stale: impl FnMut(&mut P::Value),
        create_value: impl FnOnce() -> P::Value,
    ) -> usize {
        if let Some(list) = self.free.get_mut(&key)
            && let Some(id) = list.pop()
        {
            self.counters.note_hit();
            self.entries[id].last_used_generation = generation;
            refresh_if_stale(&mut self.entries[id].value);
            return id;
        }
        let id = self.entries.len();
        self.entries.push(Entry {
            key,
            value: create_value(),
            last_used_generation: generation,
        });
        self.counters.note_miss();
        self.counters.note_insertion();
        id
    }

    /// Acquires an id without touching the cached value (key-only acquisition for tests / dry runs).
    #[cfg(test)]
    fn acquire_key_only(&mut self, key: P::Key, generation: u64) -> usize {
        self.acquire(key, generation, |_| {}, P::Value::default)
    }

    /// Returns `id` to the matching-key free list.
    fn release(&mut self, id: usize) {
        if let Some(entry) = self.entries.get(id) {
            self.free.entry(entry.key).or_default().push(id);
        }
    }

    /// Drops cached values for entries that have not been touched within `max_age` generations.
    fn gc(
        &mut self,
        current_generation: u64,
        max_age: u64,
        mut clear_value: impl FnMut(&mut P::Value),
    ) {
        let alive: Vec<bool> = self
            .entries
            .iter()
            .map(|e| current_generation.saturating_sub(e.last_used_generation) <= max_age)
            .collect();
        for list in self.free.values_mut() {
            list.retain(|&id| alive.get(id).copied().unwrap_or(false));
        }
        for (idx, &is_alive) in alive.iter().enumerate() {
            if !is_alive && let Some(entry) = self.entries.get_mut(idx) {
                clear_value(&mut entry.value);
            }
        }
    }

    /// Drops cached values for free-list entries whose key matches `pred`.
    fn evict_keys_where(
        &mut self,
        mut pred: impl FnMut(&P::Key) -> bool,
        mut clear_value: impl FnMut(&mut P::Value),
    ) {
        let keys: Vec<P::Key> = self.free.keys().copied().filter(|k| pred(k)).collect();
        for key in keys {
            let Some(ids) = self.free.remove(&key) else {
                continue;
            };
            for id in ids {
                if let Some(entry) = self.entries.get_mut(id) {
                    self.counters.note_eviction();
                    clear_value(&mut entry.value);
                }
            }
        }
    }

    /// Counts entries whose value satisfies `is_present`.
    fn retained_count(&self, is_present: impl Fn(&P::Value) -> bool) -> usize {
        self.entries.iter().filter(|e| is_present(&e.value)).count()
    }

    /// Returns the cached value at `id` for read-only access.
    fn value(&self, id: usize) -> &P::Value {
        &self.entries[id].value
    }
}

/// Transient pool metadata. Actual GPU allocation is layered on top of this key map.
#[derive(Debug, Default)]
pub struct TransientPool {
    textures: Pool<TextureKind>,
    buffers: Pool<BufferKind>,
    lru_gen: u64,
    metrics: TransientPoolMetrics,
}

impl TransientPool {
    /// Creates an empty pool.
    pub fn new() -> Self {
        Self::default()
    }

    /// Marks a new frame/generation.
    pub fn begin_generation(&mut self) {
        self.lru_gen = self.lru_gen.saturating_add(1);
    }

    /// Acquires a texture entry id for `key`, reusing a matching free entry when available.
    #[cfg(test)]
    pub fn acquire_texture(&mut self, key: TextureKey) -> usize {
        self.textures.acquire_key_only(key, self.lru_gen)
    }

    /// Acquires a real GPU texture entry for `key`, allocating on a miss.
    pub fn acquire_texture_resource(
        &mut self,
        device: &wgpu::Device,
        limits: &crate::gpu::GpuLimits,
        key: TextureKey,
        label: &'static str,
        usage: wgpu::TextureUsages,
    ) -> Result<PooledTextureLease, TransientPoolError> {
        validate_texture_key(limits, key, label, usage)?;
        let id = self.textures.acquire(
            key,
            self.lru_gen,
            |slot| {
                if slot.texture.is_none() {
                    let (texture, view) = create_texture_and_view(device, key, label, usage);
                    slot.attach(texture, view);
                }
            },
            || {
                let (texture, view) = create_texture_and_view(device, key, label, usage);
                let mut slot = TextureSlotValue::default();
                slot.attach(texture, view);
                slot
            },
        );
        texture_lease_from_entry(id, self.textures.value(id))
    }

    /// Releases a texture entry back to the matching-key free list.
    pub fn release_texture(&mut self, id: usize) {
        self.textures.release(id);
    }

    /// Acquires a buffer entry id for `key`, reusing a matching free entry when available.
    #[cfg(test)]
    pub fn acquire_buffer(&mut self, key: BufferKey) -> usize {
        self.buffers.acquire_key_only(key, self.lru_gen)
    }

    /// Acquires a real GPU buffer entry for `key`, allocating on a miss.
    pub fn acquire_buffer_resource(
        &mut self,
        device: &wgpu::Device,
        limits: &crate::gpu::GpuLimits,
        key: BufferKey,
        label: &'static str,
        usage: wgpu::BufferUsages,
        size: u64,
    ) -> Result<PooledBufferLease, TransientPoolError> {
        if !limits.buffer_size_fits(size) {
            return Err(TransientPoolError::BufferExceedsLimits {
                label,
                size,
                max: limits.max_buffer_size(),
            });
        }
        let id = self.buffers.acquire(
            key,
            self.lru_gen,
            |slot| {
                if slot.buffer.is_none() || slot.size != size {
                    slot.buffer = Some(create_buffer(device, label, usage, size));
                    slot.size = size;
                }
            },
            || BufferSlotValue {
                buffer: Some(create_buffer(device, label, usage, size)),
                size,
            },
        );
        buffer_lease_from_entry(id, self.buffers.value(id))
    }

    /// Releases a buffer entry back to the matching-key free list.
    pub fn release_buffer(&mut self, id: usize) {
        self.buffers.release(id);
    }

    /// Releases entries that have not been used for more than `max_age` generations.
    pub fn gc_tick(&mut self, max_age: u64) {
        let current = self.lru_gen;
        self.textures.gc(current, max_age, TextureSlotValue::clear);
        self.buffers.gc(current, max_age, BufferSlotValue::clear);
        self.refresh_retained_counts();
    }

    /// Drops GPU resources for free-list entries whose [`TextureKey`] matches `pred` (e.g. stale MSAA
    /// sample counts after [`crate::gpu::GpuContext::swapchain_msaa_effective`] changes).
    pub fn evict_texture_keys_where(&mut self, pred: impl FnMut(&TextureKey) -> bool) {
        self.textures
            .evict_keys_where(pred, TextureSlotValue::clear);
        self.refresh_retained_counts();
    }

    fn refresh_retained_counts(&mut self) {
        self.metrics.retained_textures = self.textures.retained_count(TextureSlotValue::is_present);
        self.metrics.retained_buffers = self.buffers.retained_count(BufferSlotValue::is_present);
    }

    /// Returns current metrics.
    pub fn metrics(&self) -> TransientPoolMetrics {
        let texture_stats = self.textures.counters.snapshot();
        let texture_view_cache = self.texture_view_cache_stats();
        let texture_view_cache_entries = self.texture_view_cache_entries();
        let buffer_stats = self.buffers.counters.snapshot();
        TransientPoolMetrics {
            texture_cache: texture_stats,
            texture_view_cache,
            texture_view_cache_entries,
            texture_hits: saturating_usize(texture_stats.hits),
            texture_misses: saturating_usize(texture_stats.misses),
            buffer_cache: buffer_stats,
            buffer_hits: saturating_usize(buffer_stats.hits),
            buffer_misses: saturating_usize(buffer_stats.misses),
            retained_textures: self.textures.retained_count(TextureSlotValue::is_present),
            retained_buffers: self.buffers.retained_count(BufferSlotValue::is_present),
        }
    }

    fn texture_view_cache_stats(&self) -> CacheStats {
        self.textures
            .entries
            .iter()
            .map(|entry| entry.value.view_cache.stats())
            .fold(CacheStats::default(), add_cache_stats)
    }

    fn texture_view_cache_entries(&self) -> usize {
        self.textures
            .entries
            .iter()
            .map(|entry| entry.value.view_cache.len())
            .sum()
    }
}

fn add_cache_stats(a: CacheStats, b: CacheStats) -> CacheStats {
    CacheStats {
        hits: a.hits.saturating_add(b.hits),
        misses: a.misses.saturating_add(b.misses),
        insertions: a.insertions.saturating_add(b.insertions),
        evictions: a.evictions.saturating_add(b.evictions),
    }
}

#[cfg(test)]
mod tests;
