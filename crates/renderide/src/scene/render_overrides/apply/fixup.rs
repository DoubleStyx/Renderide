//! Generic transform-removal id fixup for render-override entry tables.
//!
//! Both [`crate::scene::render_overrides::types::RenderTransformOverrideEntry`] and
//! [`crate::scene::render_overrides::types::RenderMaterialOverrideEntry`] carry a single
//! `node_id: i32` that must roll forward through swap-removed transform indices in the same
//! way mesh / layer / camera renderables do. The trait pulls the field reference out so the
//! shared helper handles both entry types with one implementation.

use crate::scene::dense_update::for_each_row_with_par_dispatch;
use crate::scene::render_overrides::types::{
    RenderMaterialOverrideEntry, RenderTransformOverrideEntry,
};
use crate::scene::transforms_apply::TransformRemovalEvent;
use crate::scene::world::fixup_transform_id;

/// Override-entry rows that hold a single mutable `node_id` index participating in transform
/// removal fixups.
pub(super) trait OverrideNodeRef {
    /// Returns a mutable reference to the entry's transform `node_id` field.
    fn node_id_mut(&mut self) -> &mut i32;
}

impl OverrideNodeRef for RenderTransformOverrideEntry {
    #[inline]
    fn node_id_mut(&mut self) -> &mut i32 {
        &mut self.node_id
    }
}

impl OverrideNodeRef for RenderMaterialOverrideEntry {
    #[inline]
    fn node_id_mut(&mut self) -> &mut i32 {
        &mut self.node_id
    }
}

/// Rolls each entry's `node_id` forward through every swap-removal in `removals`, fanning out to
/// rayon when the row count crosses [`crate::scene::dense_update::FIXUP_PARALLEL_MIN`].
pub(super) fn fixup_override_nodes_for_transform_removals<T>(
    rows: &mut [T],
    removals: &[TransformRemovalEvent],
) where
    T: OverrideNodeRef + Send,
{
    for removal in removals {
        for_each_row_with_par_dispatch(rows, |row| {
            let id = row.node_id_mut();
            *id = fixup_transform_id(*id, removal.removed_index, removal.last_index_before_swap);
        });
    }
}
