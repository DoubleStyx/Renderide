//! Arena ownership for the GPU skin cache: positions, normals, tangents, and temp streams.
//!
//! Centralizes per-arena buffer growth and the multi-arena allocation rollback that the
//! cache policy used to inline at the call site.

use super::entry::SkinCacheEntry;
use super::key::EntryNeed;
use crate::mesh_deform::range_alloc::{Range, RangeAllocator};

/// Storage offset alignment for arena suballocations (matches WebGPU `min_storage_buffer_offset_alignment`).
pub(super) const ARENA_ALIGN: u64 = 256;

/// Default initial arena size per stream (bytes).
pub(super) const DEFAULT_INITIAL_ARENA_BYTES: u64 = 8 * 1024 * 1024;

/// Default maximum arena size per stream (bytes).
pub(super) const DEFAULT_MAX_ARENA_BYTES: u64 = 256 * 1024 * 1024;

#[inline]
fn arena_usage() -> wgpu::BufferUsages {
    wgpu::BufferUsages::STORAGE
        | wgpu::BufferUsages::VERTEX
        | wgpu::BufferUsages::COPY_DST
        | wgpu::BufferUsages::COPY_SRC
}

/// Byte ranges inside [`SkinArenas`] for one cache line. Returned by [`SkinArenas::try_alloc_layout`].
#[derive(Debug, Clone, Copy)]
pub struct EntryRanges {
    /// Final positions arena range.
    pub positions: Range,
    /// Normals arena range when skinning is active.
    pub normals: Option<Range>,
    /// Tangents arena range when deformed tangent-space shading is needed.
    pub tangents: Option<Range>,
    /// Temp arena range when both blend and skin run.
    pub temp: Option<Range>,
    /// Temp normal range when blendshape normal deltas feed skinning.
    pub temp_normals: Option<Range>,
    /// Temp tangent range when blendshape tangent deltas feed skinning.
    pub temp_tangents: Option<Range>,
}

fn free_optional(allocator: &mut RangeAllocator, range: Option<Range>) {
    if let Some(range) = range {
        allocator.free(range);
    }
}

/// Pure rollback policy: allocate positions, then conditionally normals, tangents, and temp streams
/// dictated by [`EntryNeed`]. On any partial failure the prior allocations are returned to their allocators
/// before this returns `None`. Extracted from [`SkinArenas::try_alloc_layout`] so the policy can
/// be unit-tested without a `wgpu::Device`.
fn try_alloc_stream_arenas(
    pos: &mut RangeAllocator,
    nrm: &mut RangeAllocator,
    tan: &mut RangeAllocator,
    tmp: &mut RangeAllocator,
    need: EntryNeed,
    bytes: u64,
) -> Option<EntryRanges> {
    let positions = pos.allocate(bytes)?;
    let normals = if need.needs_normals() {
        match nrm.allocate(bytes) {
            Some(range) => Some(range),
            None => {
                pos.free(positions);
                return None;
            }
        }
    } else {
        None
    };
    let tangents = if need.needs_tangents {
        match tan.allocate(bytes) {
            Some(range) => Some(range),
            None => {
                pos.free(positions);
                free_optional(nrm, normals);
                return None;
            }
        }
    } else {
        None
    };
    let temp = if need.needs_temp_positions() {
        match tmp.allocate(bytes) {
            Some(range) => Some(range),
            None => {
                pos.free(positions);
                free_optional(nrm, normals);
                free_optional(tan, tangents);
                return None;
            }
        }
    } else {
        None
    };
    let temp_normals = if need.needs_temp_normals() {
        match tmp.allocate(bytes) {
            Some(range) => Some(range),
            None => {
                pos.free(positions);
                free_optional(nrm, normals);
                free_optional(tan, tangents);
                free_optional(tmp, temp);
                return None;
            }
        }
    } else {
        None
    };
    let temp_tangents = if need.needs_temp_tangents() {
        match tmp.allocate(bytes) {
            Some(range) => Some(range),
            None => {
                pos.free(positions);
                free_optional(nrm, normals);
                free_optional(tan, tangents);
                free_optional(tmp, temp);
                free_optional(tmp, temp_normals);
                return None;
            }
        }
    } else {
        None
    };

    Some(EntryRanges {
        positions,
        normals,
        tangents,
        temp,
        temp_normals,
        temp_tangents,
    })
}

/// Pairing of an arena buffer with its [`RangeAllocator`].
struct Arena {
    buffer: wgpu::Buffer,
    alloc: RangeAllocator,
    label: &'static str,
}

impl Arena {
    fn new(device: &wgpu::Device, capacity: u64, label: &'static str) -> Self {
        Self {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: capacity,
                usage: arena_usage(),
                mapped_at_creation: false,
            }),
            alloc: RangeAllocator::new(capacity, ARENA_ALIGN),
            label,
        }
    }

    fn grow_to(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, new_cap: u64) {
        let old_size = self.buffer.size();
        if new_cap <= old_size {
            return;
        }
        let new_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(self.label),
            size: new_cap,
            usage: arena_usage(),
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &new_buf, 0, old_size);
        self.buffer = new_buf;
        self.alloc.grow_to(new_cap);
    }
}

/// GPU arenas (`STORAGE | VERTEX`) plus per-arena allocators that share a growth schedule.
pub struct SkinArenas {
    positions: Arena,
    normals: Arena,
    tangents: Arena,
    temp: Arena,
    capacity_cap_bytes: u64,
}

impl SkinArenas {
    /// Creates three empty arenas with `initial_bytes` capacity each (clamped to the device limit).
    pub fn new(device: &wgpu::Device, max_buffer_size: u64) -> Self {
        let capacity_cap_bytes = DEFAULT_MAX_ARENA_BYTES
            .min(max_buffer_size)
            .max(ARENA_ALIGN);
        let initial = DEFAULT_INITIAL_ARENA_BYTES
            .min(capacity_cap_bytes)
            .max(ARENA_ALIGN);
        Self {
            positions: Arena::new(device, initial, "gpu_skin_cache_positions_arena"),
            normals: Arena::new(device, initial, "gpu_skin_cache_normals_arena"),
            tangents: Arena::new(device, initial, "gpu_skin_cache_tangents_arena"),
            temp: Arena::new(device, initial, "gpu_skin_cache_temp_arena"),
            capacity_cap_bytes,
        }
    }

    /// Full positions arena for forward draw binding.
    #[inline]
    pub fn positions(&self) -> &wgpu::Buffer {
        &self.positions.buffer
    }

    /// Full normals arena for skinned deformed normals.
    #[inline]
    pub fn normals(&self) -> &wgpu::Buffer {
        &self.normals.buffer
    }

    /// Full tangents arena for deformed tangents.
    #[inline]
    pub fn tangents(&self) -> &wgpu::Buffer {
        &self.tangents.buffer
    }

    /// Blendshape → skin intermediate positions when both passes run.
    #[inline]
    pub fn temp(&self) -> &wgpu::Buffer {
        &self.temp.buffer
    }

    /// Total VRAM for the arenas (bytes).
    #[inline]
    pub fn resident_bytes(&self) -> u64 {
        self.positions.buffer.size()
            + self.normals.buffer.size()
            + self.tangents.buffer.size()
            + self.temp.buffer.size()
    }

    /// Capacity ceiling shared by every arena (bytes).
    #[inline]
    pub fn capacity_cap_bytes(&self) -> u64 {
        self.capacity_cap_bytes
    }

    /// Allocates `bytes` from the positions arena, plus normals/temp dictated by `need`.
    ///
    /// On any partial failure all prior allocations are rolled back and `None` is returned, so the
    /// caller observes either a complete layout or no state change.
    pub fn try_alloc_layout(&mut self, need: EntryNeed, bytes: u64) -> Option<EntryRanges> {
        try_alloc_stream_arenas(
            &mut self.positions.alloc,
            &mut self.normals.alloc,
            &mut self.tangents.alloc,
            &mut self.temp.alloc,
            need,
            bytes,
        )
    }

    /// Returns `entry`'s ranges to the per-arena allocators.
    pub fn free_entry(&mut self, entry: &SkinCacheEntry) {
        self.positions.alloc.free(entry.positions);
        if let Some(n) = entry.normals {
            self.normals.alloc.free(n);
        }
        if let Some(t) = entry.tangents {
            self.tangents.alloc.free(t);
        }
        if let Some(t) = entry.temp {
            self.temp.alloc.free(t);
        }
        if let Some(t) = entry.temp_normals {
            self.temp.alloc.free(t);
        }
        if let Some(t) = entry.temp_tangents {
            self.temp.alloc.free(t);
        }
    }

    /// Doubles arena capacity (clamped to [`Self::capacity_cap_bytes`]). Returns `true` only when
    /// growth actually happened.
    pub fn grow_all(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) -> bool {
        let next = self
            .positions
            .alloc
            .capacity()
            .saturating_mul(2)
            .min(self.capacity_cap_bytes);
        if next <= self.positions.alloc.capacity() {
            return false;
        }
        self.positions.grow_to(device, encoder, next);
        self.normals.grow_to(device, encoder, next);
        self.tangents.grow_to(device, encoder, next);
        self.temp.grow_to(device, encoder, next);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::super::entry::bytes_for_vertices;
    use super::*;

    #[test]
    fn try_alloc_stream_arenas_rolls_back_when_temp_is_full() {
        let mut pos = RangeAllocator::new(1024, ARENA_ALIGN);
        let mut nrm = RangeAllocator::new(1024, ARENA_ALIGN);
        let mut tan = RangeAllocator::new(1024, ARENA_ALIGN);
        let mut tmp = RangeAllocator::new(0, ARENA_ALIGN);
        let need = EntryNeed {
            needs_blend: true,
            needs_skin: true,
            needs_blend_normals: false,
            needs_tangents: false,
            needs_blend_tangents: false,
        };
        let bytes = bytes_for_vertices(8);

        let result = try_alloc_stream_arenas(&mut pos, &mut nrm, &mut tan, &mut tmp, need, bytes);
        assert!(
            result.is_none(),
            "temp arena is empty so the call must fail"
        );

        assert!(
            pos.allocate(1024).is_some(),
            "positions arena must be fully reclaimed after rollback"
        );
        assert!(
            nrm.allocate(1024).is_some(),
            "normals arena must be fully reclaimed after rollback"
        );
    }

    #[test]
    fn try_alloc_stream_arenas_rolls_back_when_normals_is_full() {
        let mut pos = RangeAllocator::new(1024, ARENA_ALIGN);
        let mut nrm = RangeAllocator::new(0, ARENA_ALIGN);
        let mut tan = RangeAllocator::new(1024, ARENA_ALIGN);
        let mut tmp = RangeAllocator::new(1024, ARENA_ALIGN);
        let need = EntryNeed {
            needs_blend: false,
            needs_skin: true,
            needs_blend_normals: false,
            needs_tangents: false,
            needs_blend_tangents: false,
        };
        let bytes = bytes_for_vertices(8);

        let result = try_alloc_stream_arenas(&mut pos, &mut nrm, &mut tan, &mut tmp, need, bytes);
        assert!(result.is_none());

        assert!(
            pos.allocate(1024).is_some(),
            "positions arena must be fully reclaimed after rollback"
        );
    }

    #[test]
    fn try_alloc_stream_arenas_skips_optional_arenas_when_not_needed() {
        let mut pos = RangeAllocator::new(1024, ARENA_ALIGN);
        let mut nrm = RangeAllocator::new(0, ARENA_ALIGN);
        let mut tan = RangeAllocator::new(0, ARENA_ALIGN);
        let mut tmp = RangeAllocator::new(0, ARENA_ALIGN);
        let need = EntryNeed {
            needs_blend: false,
            needs_skin: false,
            needs_blend_normals: false,
            needs_tangents: false,
            needs_blend_tangents: false,
        };

        let ranges = try_alloc_stream_arenas(&mut pos, &mut nrm, &mut tan, &mut tmp, need, 256)
            .expect("positions-only allocation should succeed");
        assert_eq!(ranges.positions.len_bytes, 256);
        assert!(ranges.normals.is_none());
        assert!(ranges.tangents.is_none());
        assert!(ranges.temp.is_none());
    }

    #[test]
    fn try_alloc_stream_arenas_allocates_tangent_and_temp_tangent_ranges() {
        let mut pos = RangeAllocator::new(2048, ARENA_ALIGN);
        let mut nrm = RangeAllocator::new(2048, ARENA_ALIGN);
        let mut tan = RangeAllocator::new(2048, ARENA_ALIGN);
        let mut tmp = RangeAllocator::new(4096, ARENA_ALIGN);
        let need = EntryNeed {
            needs_blend: true,
            needs_skin: true,
            needs_blend_normals: true,
            needs_tangents: true,
            needs_blend_tangents: true,
        };

        let ranges = try_alloc_stream_arenas(&mut pos, &mut nrm, &mut tan, &mut tmp, need, 256)
            .expect("all requested stream ranges should fit");
        assert!(ranges.normals.is_some());
        assert!(ranges.tangents.is_some());
        assert!(ranges.temp.is_some());
        assert!(ranges.temp_normals.is_some());
        assert!(ranges.temp_tangents.is_some());
    }
}
