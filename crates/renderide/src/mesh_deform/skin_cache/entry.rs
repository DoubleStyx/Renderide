//! Resident cache line, frame stats, and pure helpers used by the GPU skin cache.

use hashbrown::HashMap;

use super::key::{EntryNeed, SkinCacheKey};
use crate::mesh_deform::range_alloc::Range;

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
    /// Last [`super::GpuSkinCache::frame_counter`] that touched this entry.
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

/// Bytes required to hold `vertex_count` `vec4<f32>` deform outputs (clamped to one stride).
#[inline]
pub fn bytes_for_vertices(vertex_count: u32) -> u64 {
    u64::from(vertex_count).saturating_mul(16).max(16)
}

/// Whether `entry`'s arena layout already covers the new `need`.
#[inline]
pub fn entry_layout_matches(entry: &SkinCacheEntry, need: EntryNeed) -> bool {
    let needs_temp = need.needs_blend && need.needs_skin;
    let needs_normals = need.needs_skin;
    (!needs_temp || entry.temp.is_some()) && (!needs_normals || entry.normals.is_some())
}

/// Returns an entry only when it was last touched in the current frame (no LRU eligibility).
#[inline]
pub fn lookup_current_entry<'a>(
    entries: &'a HashMap<SkinCacheKey, SkinCacheEntry>,
    key: &SkinCacheKey,
    frame_counter: u64,
) -> Option<&'a SkinCacheEntry> {
    entries
        .get(key)
        .filter(|entry| entry.last_touched_frame == frame_counter)
}

/// Returns the LRU key strictly older than `frame_counter`, protecting current-frame entries.
#[inline]
pub fn lru_evictable_key(
    entries: &HashMap<SkinCacheKey, SkinCacheEntry>,
    frame_counter: u64,
) -> Option<SkinCacheKey> {
    entries
        .iter()
        .filter(|(_, entry)| entry.last_touched_frame < frame_counter)
        .min_by_key(|(_, entry)| entry.last_touched_frame)
        .map(|(key, _)| *key)
}

#[cfg(test)]
mod tests {
    use super::super::key::SkinCacheRendererKind;
    use super::*;
    use crate::scene::{MeshRendererInstanceId, RenderSpaceId};

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
