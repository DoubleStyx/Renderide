//! Stable identifiers for render spaces and dense transform indices.

/// Host `RenderSpaceUpdate.id` — distinct root for a transform hierarchy.
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
        if u < len {
            Some(u)
        } else {
            None
        }
    }
}
