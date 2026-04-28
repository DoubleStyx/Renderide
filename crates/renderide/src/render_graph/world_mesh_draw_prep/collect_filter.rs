//! Per-space transform filter mask construction for world-mesh draw collection.

use hashbrown::HashMap;

use crate::scene::RenderSpaceId;

use super::DrawCollectionContext;

/// Builds per-space `Vec<bool>` masks from [`DrawCollectionContext::transform_filter`].
///
/// Returns an empty map when no transform filter was provided.
pub(super) fn build_per_space_filter_masks(
    space_ids: &[RenderSpaceId],
    ctx: &DrawCollectionContext<'_>,
) -> HashMap<RenderSpaceId, Vec<bool>> {
    if ctx.transform_filter.is_some() {
        space_ids
            .iter()
            .copied()
            .filter_map(|sid| {
                let mask = ctx.transform_filter?.build_pass_mask(ctx.scene, sid)?;
                Some((sid, mask))
            })
            .collect()
    } else {
        HashMap::new()
    }
}
