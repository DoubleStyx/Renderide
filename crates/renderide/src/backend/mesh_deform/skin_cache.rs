//! GPU skin cache: persistent `STORAGE | VERTEX` arenas with per-instance byte ranges.
//!
//! Used by mesh deform compute (writes with base offsets) and world mesh forward (binds
//! [`wgpu::Buffer::slice`] per draw).

use std::collections::HashMap;

use crate::scene::RenderSpaceId;

use super::range_alloc::{Range, RangeAllocator};

/// Stable key for a deformable mesh instance: render space + scene node id.
pub type SkinCacheKey = (RenderSpaceId, i32);

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

/// Three arenas for deform outputs; ranges are tracked by [`RangeAllocator`].
pub struct GpuSkinCache {
    positions_arena: wgpu::Buffer,
    normals_arena: wgpu::Buffer,
    temp_arena: wgpu::Buffer,
    pos_alloc: RangeAllocator,
    nrm_alloc: RangeAllocator,
    tmp_alloc: RangeAllocator,
    entries: HashMap<SkinCacheKey, SkinCacheEntry>,
    /// Incremented each winit tick ([`crate::backend::RenderBackend::reset_light_prep_for_tick`]).
    frame_counter: u64,
    capacity_cap_bytes: u64,
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
    let want_temp = need.needs_blend && need.needs_skin;
    let want_nrm = need.needs_skin;
    e.temp.is_some() == want_temp && e.normals.is_some() == want_nrm
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
            if self.evict_lru() {
                continue;
            }
            logger::error!(
                "GpuSkinCache: could not allocate {} bytes for deform (arena cap {})",
                b,
                self.capacity_cap_bytes
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
        true
    }

    fn evict_lru(&mut self) -> bool {
        let Some(key) = self
            .entries
            .iter()
            .min_by_key(|(_, e)| e.last_touched_frame)
            .map(|(k, _)| *k)
        else {
            return false;
        };
        self.remove_entry(key);
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
    use super::*;

    /// Returns a [`wgpu::Device`] plus its paired [`wgpu::Queue`].
    ///
    /// The queue **must** be kept alive for the device's lifetime: wgpu stores a weak reference
    /// internally; dropping the queue handle makes command encoder creation panic.
    fn dummy_device_and_queue() -> Option<(wgpu::Device, wgpu::Queue)> {
        pollster::block_on(async {
            let instance = wgpu::Instance::default();
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .ok()?;
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    ..Default::default()
                })
                .await
                .ok()?;
            Some((device, queue))
        })
    }

    /// Covers distinct keys and same-key reuse against a real [`wgpu::Device`].
    ///
    /// Kept as **one** `#[test]` so this module never runs two GPU initializations in parallel.
    /// The default harness still runs this test concurrently with hundreds of unrelated tests; on
    /// some stacks that can SIGSEGV during Vulkan init. Run with `--ignored` locally or in CI
    /// where the GPU stack is stable (see `materials::registry` `wgpu_cache` tests).
    #[test]
    #[ignore = "wgpu/GPU stack (may SIGSEGV vs parallel harness); run with --ignored"]
    fn gpu_skin_cache_distinct_keys_and_same_key_reuse() {
        let Some((device, _queue)) = dummy_device_and_queue() else {
            return;
        };
        let max = device.limits().max_buffer_size;
        let mut cache = GpuSkinCache::new(&device, max);
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let need_sk = EntryNeed {
            needs_blend: false,
            needs_skin: true,
        };

        let k0 = (RenderSpaceId(0), 10);
        let k1 = (RenderSpaceId(0), 11);
        let off_distinct_a = cache
            .get_or_alloc(&device, &mut enc, k0, need_sk, 64)
            .expect("a0")
            .positions
            .offset_bytes;
        let off_distinct_b = cache
            .get_or_alloc(&device, &mut enc, k1, need_sk, 64)
            .expect("a1")
            .positions
            .offset_bytes;
        assert_ne!(off_distinct_a, off_distinct_b);

        let k = (RenderSpaceId(1), 5);
        let off_reuse0 = cache
            .get_or_alloc(&device, &mut enc, k, need_sk, 32)
            .unwrap()
            .positions
            .offset_bytes;
        let off_reuse1 = cache
            .get_or_alloc(&device, &mut enc, k, need_sk, 32)
            .unwrap()
            .positions
            .offset_bytes;
        assert_eq!(off_reuse0, off_reuse1);
    }
}
