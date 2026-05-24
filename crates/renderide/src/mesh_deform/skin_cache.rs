//! GPU skin cache: persistent `STORAGE | VERTEX` arenas with per-instance byte ranges.
//!
//! Used by mesh deform compute (writes with base offsets) and world mesh forward (binds
//! [`wgpu::Buffer::slice`] per draw).

mod arenas;
mod entry;
mod key;

use hashbrown::HashMap;

use self::arenas::{EntryRanges, SkinArenas};
use self::entry::{
    bytes_for_vertices, entry_layout_matches, lookup_current_entry, lru_evictable_key,
};

pub use self::entry::{DeformSignature, SkinCacheEntry, SkinCacheFrameStats};
pub use self::key::{EntryNeed, SkinCacheKey, SkinCacheRendererKind};

/// Arenas for deform outputs; ranges are tracked by [`crate::mesh_deform::range_alloc::RangeAllocator`].
pub struct GpuSkinCache {
    arenas: SkinArenas,
    entries: HashMap<SkinCacheKey, SkinCacheEntry>,
    /// Incremented each winit tick ([`crate::backend::FrameResourceManager::reset_light_prep_for_tick`]).
    frame_counter: u64,
    /// Counters reset by [`Self::advance_frame`] and plotted after the deform pass.
    stats: SkinCacheFrameStats,
}

impl GpuSkinCache {
    /// Creates empty arenas with the default initial capacity (clamped to `max_buffer_size`).
    pub fn new(device: &wgpu::Device, max_buffer_size: u64) -> Self {
        Self {
            arenas: SkinArenas::new(device, max_buffer_size),
            entries: HashMap::new(),
            frame_counter: 0,
            stats: SkinCacheFrameStats::default(),
        }
    }

    /// Monotonic frame index (for LRU / stale sweep).
    #[inline]
    pub fn frame_counter(&self) -> u64 {
        self.frame_counter
    }

    /// Advance once per winit tick before deform / forward work.
    pub fn advance_frame(&mut self) {
        self.frame_counter = self.frame_counter.saturating_add(1);
        self.stats = SkinCacheFrameStats::default();
    }

    /// Current frame's cache pressure counters.
    #[inline]
    pub fn frame_stats(&self) -> SkinCacheFrameStats {
        self.stats
    }

    /// Full positions arena (`STORAGE | VERTEX`); bind [`SkinCacheEntry::positions`] byte range for draws.
    #[inline]
    pub fn positions_arena(&self) -> &wgpu::Buffer {
        self.arenas.positions()
    }

    /// Full normals arena for skinned deformed normals.
    #[inline]
    pub fn normals_arena(&self) -> &wgpu::Buffer {
        self.arenas.normals()
    }

    /// Full tangents arena for deformed tangents.
    #[inline]
    pub fn tangents_arena(&self) -> &wgpu::Buffer {
        self.arenas.tangents()
    }

    /// Looks up a cache line only when mesh deform touched it in the current cache frame.
    pub fn lookup_current(&self, key: &SkinCacheKey) -> Option<&SkinCacheEntry> {
        lookup_current_entry(&self.entries, key, self.frame_counter)
    }

    /// Removes entries not touched since before `stale_before_frame` (exclusive).
    pub fn sweep_stale(&mut self, stale_before_frame: u64) {
        let keys: Vec<SkinCacheKey> = self
            .entries
            .iter()
            .filter(|(_, e)| e.last_touched_frame < stale_before_frame)
            .map(|(k, _)| *k)
            .collect();
        for k in keys {
            self.remove_entry(k);
        }
    }

    /// Removes all entries owned by `space_id` and returns the number removed.
    pub fn remove_space(&mut self, space_id: crate::scene::RenderSpaceId) -> usize {
        let keys: Vec<SkinCacheKey> = self
            .entries
            .keys()
            .copied()
            .filter(|key| key.space_id == space_id)
            .collect();
        let removed = keys.len();
        for key in keys {
            self.remove_entry(key);
        }
        removed
    }

    /// Allocates or reuses ranges and returns arena buffers for encode passes (single borrow).
    pub fn get_or_alloc_with_arenas(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        profiler: Option<&crate::profiling::GpuProfilerHandle>,
        key: SkinCacheKey,
        need: EntryNeed,
        vertex_count: u32,
    ) -> Option<(
        &SkinCacheEntry,
        &wgpu::Buffer,
        &wgpu::Buffer,
        &wgpu::Buffer,
        &wgpu::Buffer,
    )> {
        if vertex_count == 0 {
            return None;
        }

        if self.try_reuse_existing(key, need, vertex_count) {
            return self.entry_and_arenas(&key);
        }

        if self.entries.contains_key(&key) {
            self.remove_entry(key);
        }

        let bytes = bytes_for_vertices(vertex_count);
        loop {
            if let Some(ranges) = self.arenas.try_alloc_layout(need, bytes) {
                self.commit_entry(key, ranges, vertex_count);
                self.stats.allocations = self.stats.allocations.saturating_add(1);
                return self.entry_and_arenas(&key);
            }
            if self.arenas.grow_all(device, encoder, profiler) {
                self.stats.grows = self.stats.grows.saturating_add(1);
                continue;
            }
            if self.evict_lru_before_current_frame() {
                self.stats.evictions = self.stats.evictions.saturating_add(1);
                continue;
            }
            self.stats.current_frame_eviction_refusals =
                self.stats.current_frame_eviction_refusals.saturating_add(1);
            logger::error!(
                "GpuSkinCache: could not allocate {} bytes for deform (arena cap {}, current-frame entries protected)",
                bytes,
                self.arenas.capacity_cap_bytes()
            );
            return None;
        }
    }

    /// Allocates or reuses a cache line and marks it current for this frame.
    #[inline]
    pub fn prepare_entry(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        profiler: Option<&crate::profiling::GpuProfilerHandle>,
        key: SkinCacheKey,
        need: EntryNeed,
        vertex_count: u32,
    ) -> bool {
        self.get_or_alloc_with_arenas(device, encoder, profiler, key, need, vertex_count)
            .is_some()
    }

    /// Returns a current entry with arena buffers after allocation/growth has completed.
    pub fn lookup_current_with_arenas(
        &self,
        key: &SkinCacheKey,
    ) -> Option<(
        &SkinCacheEntry,
        &wgpu::Buffer,
        &wgpu::Buffer,
        &wgpu::Buffer,
        &wgpu::Buffer,
    )> {
        let entry = lookup_current_entry(&self.entries, key, self.frame_counter)?;
        let (p, n, t, tmp) = self.arena_buffers();
        Some((entry, p, n, t, tmp))
    }

    /// Last deform signature recorded for `key`.
    #[inline]
    pub fn entry_deform_signature(&self, key: &SkinCacheKey) -> Option<DeformSignature> {
        self.entries.get(key).and_then(|e| e.last_deform_signature)
    }

    /// Stores the deform signature for a current cache entry.
    pub fn set_entry_deform_signature(&mut self, key: SkinCacheKey, signature: DeformSignature) {
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.last_deform_signature = Some(signature);
        }
    }

    fn try_reuse_existing(
        &mut self,
        key: SkinCacheKey,
        need: EntryNeed,
        vertex_count: u32,
    ) -> bool {
        let Some(existing) = self.entries.get(&key) else {
            return false;
        };
        if existing.vertex_count != vertex_count || !entry_layout_matches(existing, need) {
            return false;
        }
        let touch = self.frame_counter;
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.last_touched_frame = touch;
        }
        self.stats.reuses = self.stats.reuses.saturating_add(1);
        true
    }

    fn commit_entry(&mut self, key: SkinCacheKey, ranges: EntryRanges, vertex_count: u32) {
        self.entries.insert(
            key,
            SkinCacheEntry::from_ranges(ranges, vertex_count, self.frame_counter),
        );
    }

    fn entry_and_arenas(
        &self,
        key: &SkinCacheKey,
    ) -> Option<(
        &SkinCacheEntry,
        &wgpu::Buffer,
        &wgpu::Buffer,
        &wgpu::Buffer,
        &wgpu::Buffer,
    )> {
        let entry = self.entries.get(key)?;
        let (p, n, t, tmp) = self.arena_buffers();
        Some((entry, p, n, t, tmp))
    }

    /// Borrowed references to the four arena buffers (positions, normals, tangents, temp).
    #[inline]
    fn arena_buffers(&self) -> (&wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer) {
        (
            self.arenas.positions(),
            self.arenas.normals(),
            self.arenas.tangents(),
            self.arenas.temp(),
        )
    }

    fn evict_lru_before_current_frame(&mut self) -> bool {
        let Some(key) = lru_evictable_key(&self.entries, self.frame_counter) else {
            return false;
        };
        self.remove_entry(key);
        true
    }

    fn remove_entry(&mut self, key: SkinCacheKey) {
        let Some(entry) = self.entries.remove(&key) else {
            return;
        };
        self.arenas.free_entry(&entry);
    }
}
