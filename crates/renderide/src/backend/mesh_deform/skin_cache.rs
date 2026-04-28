//! GPU skin cache: persistent `STORAGE | VERTEX` arenas with per-instance byte ranges.
//!
//! Used by mesh deform compute (writes with base offsets) and world mesh forward (binds
//! [`wgpu::Buffer::slice`] per draw).

use std::collections::HashMap;

use crate::scene::{MeshRendererInstanceId, RenderSpaceId};

use super::range_alloc::{Range, RangeAllocator};

/// Source renderer list for a deformable mesh instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SkinCacheRendererKind {
    /// Static mesh renderer table.
    Static,
    /// Skinned mesh renderer table.
    Skinned,
}

/// Stable key for a deformable mesh instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SkinCacheKey {
    /// Render space that owns the renderer.
    pub space_id: RenderSpaceId,
    /// Renderer table selected by this key.
    pub renderer_kind: SkinCacheRendererKind,
    /// Renderer-local identity that survives dense table reindexing.
    pub instance_id: MeshRendererInstanceId,
}

impl SkinCacheKey {
    /// Builds a skin-cache key from draw/deform identity fields.
    pub fn new(
        space_id: RenderSpaceId,
        renderer_kind: SkinCacheRendererKind,
        instance_id: MeshRendererInstanceId,
    ) -> Self {
        Self {
            space_id,
            renderer_kind,
            instance_id,
        }
    }

    /// Builds a skin-cache key from a draw's `skinned` flag.
    pub fn from_draw_parts(
        space_id: RenderSpaceId,
        skinned: bool,
        instance_id: MeshRendererInstanceId,
    ) -> Self {
        let renderer_kind = if skinned {
            SkinCacheRendererKind::Skinned
        } else {
            SkinCacheRendererKind::Static
        };
        Self::new(space_id, renderer_kind, instance_id)
    }
}

/// Whether blendshape and/or skinning compute runs for this instance (drives arena layout).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EntryNeed {
    /// Sparse blendshape scatter runs.
    pub needs_blend: bool,
    /// Linear blend skinning runs.
    pub needs_skin: bool,
}

/// One resident cache line: sub-ranges inside the global arenas.
#[derive(Debug)]
pub struct SkinCacheEntry {
    /// Final position stream (`vec4<f32>` per vertex) for forward binding.
    pub positions: Range,
    /// Deformed normals when skinning is active.
    pub normals: Option<Range>,
    /// Intermediate positions after blendshape when both blend and skin run.
    pub temp: Option<Range>,
    /// Vertex count for this cache line (matches mesh deform snapshot).
    pub vertex_count: u32,
    /// Last [`GpuSkinCache::frame_counter`] that touched this entry.
    pub last_touched_frame: u64,
}

/// Per-frame cache pressure counters for diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SkinCacheFrameStats {
    /// Cache entries reused without reallocating.
    pub reuses: u64,
    /// Cache entries allocated this frame.
    pub allocations: u64,
    /// Arena growth operations performed this frame.
    pub grows: u64,
    /// Prior-frame entries evicted to make room.
    pub evictions: u64,
    /// Allocation attempts skipped because all entries were current-frame entries.
    pub current_frame_eviction_refusals: u64,
}

/// Three arenas for deform outputs; ranges are tracked by [`RangeAllocator`].
pub struct GpuSkinCache {
    positions_arena: wgpu::Buffer,
    normals_arena: wgpu::Buffer,
    temp_arena: wgpu::Buffer,
    pos_alloc: RangeAllocator,
    nrm_alloc: RangeAllocator,
    tmp_alloc: RangeAllocator,
    entries: HashMap<SkinCacheKey, SkinCacheEntry>,
    /// Incremented each winit tick ([`crate::backend::FrameResourceManager::reset_light_prep_for_tick`]).
    frame_counter: u64,
    capacity_cap_bytes: u64,
    /// Counters reset by [`Self::advance_frame`] and plotted after the deform pass.
    stats: SkinCacheFrameStats,
}

const ARENA_ALIGN: u64 = 256;
/// Default initial arena size per stream (bytes).
const DEFAULT_INITIAL_ARENA_BYTES: u64 = 8 * 1024 * 1024;
/// Default maximum arena size per stream (bytes).
const DEFAULT_MAX_ARENA_BYTES: u64 = 256 * 1024 * 1024;

fn arena_usage() -> wgpu::BufferUsages {
    wgpu::BufferUsages::STORAGE
        | wgpu::BufferUsages::VERTEX
        | wgpu::BufferUsages::COPY_DST
        | wgpu::BufferUsages::COPY_SRC
}

fn bytes_for_vertices(vertex_count: u32) -> u64 {
    (vertex_count as u64).saturating_mul(16).max(16)
}

fn entry_layout_matches(e: &SkinCacheEntry, need: EntryNeed) -> bool {
    let needs_temp = need.needs_blend && need.needs_skin;
    let needs_normals = need.needs_skin;
    (!needs_temp || e.temp.is_some()) && (!needs_normals || e.normals.is_some())
}

impl GpuSkinCache {
    /// Creates three empty arenas with `initial_bytes` capacity each (clamped to `max_bytes` and device limit).
    pub fn new(device: &wgpu::Device, max_buffer_size: u64) -> Self {
        let cap = DEFAULT_MAX_ARENA_BYTES
            .min(max_buffer_size)
            .max(ARENA_ALIGN);
        let initial = DEFAULT_INITIAL_ARENA_BYTES.min(cap).max(ARENA_ALIGN);

        let positions_arena = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_skin_cache_positions_arena"),
            size: initial,
            usage: arena_usage(),
            mapped_at_creation: false,
        });
        let normals_arena = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_skin_cache_normals_arena"),
            size: initial,
            usage: arena_usage(),
            mapped_at_creation: false,
        });
        let temp_arena = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_skin_cache_temp_arena"),
            size: initial,
            usage: arena_usage(),
            mapped_at_creation: false,
        });

        Self {
            positions_arena,
            normals_arena,
            temp_arena,
            pos_alloc: RangeAllocator::new(initial, ARENA_ALIGN),
            nrm_alloc: RangeAllocator::new(initial, ARENA_ALIGN),
            tmp_alloc: RangeAllocator::new(initial, ARENA_ALIGN),
            entries: HashMap::new(),
            frame_counter: 0,
            capacity_cap_bytes: cap,
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

    /// Total VRAM for the three arenas (bytes).
    pub fn resident_bytes(&self) -> u64 {
        self.positions_arena.size() + self.normals_arena.size() + self.temp_arena.size()
    }

    /// Full positions arena (`STORAGE | VERTEX`); bind [`SkinCacheEntry::positions`] byte range for draws.
    #[inline]
    pub fn positions_arena(&self) -> &wgpu::Buffer {
        &self.positions_arena
    }

    /// Full normals arena for skinned deformed normals.
    #[inline]
    pub fn normals_arena(&self) -> &wgpu::Buffer {
        &self.normals_arena
    }

    /// Blendshape → skin intermediate positions when both passes run.
    #[inline]
    pub fn temp_arena(&self) -> &wgpu::Buffer {
        &self.temp_arena
    }

    /// Looks up a cache line without allocating.
    pub fn lookup(&self, key: &SkinCacheKey) -> Option<&SkinCacheEntry> {
        self.entries.get(key)
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

    /// Allocates or reuses ranges for `key`. On failure, logs and returns `None`.
    pub fn get_or_alloc(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        key: SkinCacheKey,
        need: EntryNeed,
        vertex_count: u32,
    ) -> Option<&SkinCacheEntry> {
        self.get_or_alloc_with_arenas(device, encoder, key, need, vertex_count)
            .map(|(e, _, _, _)| e)
    }

    /// Like [`Self::get_or_alloc`], also returns arena buffers for encode passes (single borrow).
    pub fn get_or_alloc_with_arenas(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        key: SkinCacheKey,
        need: EntryNeed,
        vertex_count: u32,
    ) -> Option<(&SkinCacheEntry, &wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer)> {
        if vertex_count == 0 {
            return None;
        }
        let touch = self.frame_counter;
        if let Some(existing) = self.entries.get(&key) {
            if existing.vertex_count == vertex_count && entry_layout_matches(existing, need) {
                if let Some(e) = self.entries.get_mut(&key) {
                    e.last_touched_frame = touch;
                }
                self.stats.reuses = self.stats.reuses.saturating_add(1);
                let entry = self.entries.get(&key)?;
                return Some((
                    entry,
                    &self.positions_arena,
                    &self.normals_arena,
                    &self.temp_arena,
                ));
            }
        }
        if self.entries.contains_key(&key) {
            self.remove_entry(key);
        }

        let b = bytes_for_vertices(vertex_count);
        loop {
            if self
                .try_insert_entry(key, need, vertex_count, touch, b)
                .is_ok()
            {
                self.stats.allocations = self.stats.allocations.saturating_add(1);
                let entry = self.entries.get(&key)?;
                return Some((
                    entry,
                    &self.positions_arena,
                    &self.normals_arena,
                    &self.temp_arena,
                ));
            }
            if self.grow_all(device, encoder) {
                continue;
            }
            if self.evict_lru_before_current_frame() {
                continue;
            }
            self.stats.current_frame_eviction_refusals =
                self.stats.current_frame_eviction_refusals.saturating_add(1);
            logger::error!(
                "GpuSkinCache: could not allocate {} bytes for deform (arena cap {}, current-frame entries protected)",
                b, self.capacity_cap_bytes
            );
            return None;
        }
    }

    fn try_insert_entry(
        &mut self,
        key: SkinCacheKey,
        need: EntryNeed,
        vertex_count: u32,
        touch: u64,
        b: u64,
    ) -> Result<(), ()> {
        let Some(pos) = self.pos_alloc.allocate(b) else {
            return Err(());
        };

        let normals = if need.needs_skin {
            match self.nrm_alloc.allocate(b) {
                Some(n) => Some(n),
                None => {
                    self.pos_alloc.free(pos);
                    return Err(());
                }
            }
        } else {
            None
        };

        let temp = if need.needs_blend && need.needs_skin {
            match self.tmp_alloc.allocate(b) {
                Some(t) => Some(t),
                None => {
                    self.pos_alloc.free(pos);
                    if let Some(n) = normals {
                        self.nrm_alloc.free(n);
                    }
                    return Err(());
                }
            }
        } else {
            None
        };

        self.entries.insert(
            key,
            SkinCacheEntry {
                positions: pos,
                normals,
                temp,
                vertex_count,
                last_touched_frame: touch,
            },
        );
        Ok(())
    }

    fn grow_all(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) -> bool {
        let next = self
            .pos_alloc
            .capacity()
            .saturating_mul(2)
            .min(self.capacity_cap_bytes);
        if next <= self.pos_alloc.capacity() {
            return false;
        }
        grow_one_arena(
            device,
            encoder,
            &mut self.positions_arena,
            &mut self.pos_alloc,
            next,
            "gpu_skin_cache_positions_arena",
        );
        grow_one_arena(
            device,
            encoder,
            &mut self.normals_arena,
            &mut self.nrm_alloc,
            next,
            "gpu_skin_cache_normals_arena",
        );
        grow_one_arena(
            device,
            encoder,
            &mut self.temp_arena,
            &mut self.tmp_alloc,
            next,
            "gpu_skin_cache_temp_arena",
        );
        self.stats.grows = self.stats.grows.saturating_add(1);
        true
    }

    fn evict_lru_before_current_frame(&mut self) -> bool {
        let Some(key) = lru_evictable_key(&self.entries, self.frame_counter) else {
            return false;
        };
        self.remove_entry(key);
        self.stats.evictions = self.stats.evictions.saturating_add(1);
        true
    }

    fn remove_entry(&mut self, key: SkinCacheKey) {
        let Some(e) = self.entries.remove(&key) else {
            return;
        };
        self.pos_alloc.free(e.positions);
        if let Some(n) = e.normals {
            self.nrm_alloc.free(n);
        }
        if let Some(t) = e.temp {
            self.tmp_alloc.free(t);
        }
    }
}

fn lookup_current_entry<'a>(
    entries: &'a HashMap<SkinCacheKey, SkinCacheEntry>,
    key: &SkinCacheKey,
    frame_counter: u64,
) -> Option<&'a SkinCacheEntry> {
    entries
        .get(key)
        .filter(|entry| entry.last_touched_frame == frame_counter)
}

fn lru_evictable_key(
    entries: &HashMap<SkinCacheKey, SkinCacheEntry>,
    frame_counter: u64,
) -> Option<SkinCacheKey> {
    entries
        .iter()
        .filter(|(_, entry)| entry.last_touched_frame < frame_counter)
        .min_by_key(|(_, entry)| entry.last_touched_frame)
        .map(|(key, _)| *key)
}

fn grow_one_arena(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    buf: &mut wgpu::Buffer,
    alloc: &mut RangeAllocator,
    new_cap: u64,
    label: &'static str,
) {
    let old_size = buf.size();
    if new_cap <= old_size {
        return;
    }
    let new_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: new_cap,
        usage: arena_usage(),
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(buf, 0, &new_buf, 0, old_size);
    *buf = new_buf;
    alloc.grow_to(new_cap);
}

#[cfg(test)]
mod tests {
    //! CPU-only skin-cache key identity tests.

    use super::*;

    #[test]
    fn key_distinguishes_static_and_skinned_renderer_tables() {
        let instance_id = MeshRendererInstanceId(12);
        let static_key =
            SkinCacheKey::new(RenderSpaceId(7), SkinCacheRendererKind::Static, instance_id);
        let skinned_key = SkinCacheKey::new(
            RenderSpaceId(7),
            SkinCacheRendererKind::Skinned,
            instance_id,
        );

        assert_ne!(static_key, skinned_key);
    }

    #[test]
    fn key_distinguishes_two_renderers_on_the_same_transform_by_instance_id() {
        let first = SkinCacheKey::new(
            RenderSpaceId(7),
            SkinCacheRendererKind::Skinned,
            MeshRendererInstanceId(1),
        );
        let second = SkinCacheKey::new(
            RenderSpaceId(7),
            SkinCacheRendererKind::Skinned,
            MeshRendererInstanceId(2),
        );

        assert_ne!(first, second);
    }

    fn test_entry(last_touched_frame: u64) -> SkinCacheEntry {
        SkinCacheEntry {
            positions: Range {
                offset_bytes: 0,
                len_bytes: 16,
            },
            normals: None,
            temp: None,
            vertex_count: 1,
            last_touched_frame,
        }
    }

    #[test]
    fn lookup_current_rejects_prior_frame_entries() {
        let key = SkinCacheKey::new(
            RenderSpaceId(7),
            SkinCacheRendererKind::Skinned,
            MeshRendererInstanceId(1),
        );
        let mut entries = HashMap::new();
        entries.insert(key, test_entry(10));

        assert!(lookup_current_entry(&entries, &key, 11).is_none());
        assert!(lookup_current_entry(&entries, &key, 10).is_some());
    }

    #[test]
    fn lru_evictable_key_ignores_current_frame_entries() {
        let old = SkinCacheKey::new(
            RenderSpaceId(7),
            SkinCacheRendererKind::Skinned,
            MeshRendererInstanceId(1),
        );
        let current = SkinCacheKey::new(
            RenderSpaceId(7),
            SkinCacheRendererKind::Skinned,
            MeshRendererInstanceId(2),
        );
        let mut entries = HashMap::new();
        entries.insert(current, test_entry(9));
        entries.insert(old, test_entry(7));

        assert_eq!(lru_evictable_key(&entries, 9), Some(old));
        assert_eq!(lru_evictable_key(&entries, 7), None);
    }

    #[test]
    fn layout_match_accepts_extra_temp_to_avoid_viseme_churn() {
        let entry = SkinCacheEntry {
            positions: Range {
                offset_bytes: 0,
                len_bytes: 16,
            },
            normals: Some(Range {
                offset_bytes: 16,
                len_bytes: 16,
            }),
            temp: Some(Range {
                offset_bytes: 32,
                len_bytes: 16,
            }),
            vertex_count: 1,
            last_touched_frame: 1,
        };

        assert!(entry_layout_matches(
            &entry,
            EntryNeed {
                needs_blend: false,
                needs_skin: true,
            },
        ));
    }
}
