//! Stable identifiers for render spaces and dense transform indices.

/// Host `RenderSpaceUpdate.id` -- distinct root for a transform hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct RenderSpaceId(pub i32);

/// Dense transform index from the host (`transform_id` in `TransformsUpdate` batches).
///
/// Invariant: for a space with `N` transforms, valid indices are `0..N` (as `usize`). Parent
/// references use `i32` with `-1` meaning root; a parent must be either `-1` or another valid index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct TransformIndex(pub i32);

impl TransformIndex {
    /// Host terminator for removal / update streams (`id < 0` stops iteration).
    pub fn is_stream_end(self) -> bool {
        self.0 < 0
    }

    /// `Some(usize)` when this index is in range for `len` nodes.
    pub fn to_usize(self, len: usize) -> Option<usize> {
        if self.0 < 0 {
            return None;
        }
        let u = self.0 as usize;
        (u < len).then_some(u)
    }
}

#[cfg(test)]
mod tests {
    use super::{RenderSpaceId, TransformIndex};

    /// [`TransformIndex::is_stream_end`] triggers for any negative value, including [`i32::MIN`],
    /// because host removal / update streams use the first negative index as a terminator.
    #[test]
    fn transform_index_stream_end_covers_all_negative_values() {
        assert!(TransformIndex(-1).is_stream_end());
        assert!(TransformIndex(i32::MIN).is_stream_end());
        assert!(!TransformIndex(0).is_stream_end());
        assert!(!TransformIndex(i32::MAX).is_stream_end());
    }

    /// [`TransformIndex::to_usize`] rejects negative values and out-of-range values; the final
    /// valid index is `len - 1`, not `len`.
    #[test]
    fn transform_index_to_usize_bounds() {
        assert_eq!(TransformIndex(-1).to_usize(4), None);
        assert_eq!(TransformIndex(0).to_usize(4), Some(0));
        assert_eq!(TransformIndex(3).to_usize(4), Some(3));
        assert_eq!(TransformIndex(4).to_usize(4), None);
        assert_eq!(TransformIndex(0).to_usize(0), None);
        assert_eq!(TransformIndex(i32::MAX).to_usize(4), None);
    }

    /// [`RenderSpaceId`] is a transparent newtype and must be `Copy`, `Eq`, and `Hash` so it can be
    /// used as a key in scene-level maps; the `#[repr(transparent)]` also lets it share layout with
    /// the raw `i32` host field.
    #[test]
    fn render_space_id_is_trivially_copy_and_comparable() {
        let a = RenderSpaceId(7);
        let b = a;
        assert_eq!(a, b);
        assert_ne!(a, RenderSpaceId(8));
        assert_eq!(size_of::<RenderSpaceId>(), size_of::<i32>());
    }
}
