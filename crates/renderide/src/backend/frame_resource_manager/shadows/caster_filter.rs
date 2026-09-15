//! Shadow-caster filtering for a source view's shared draw list.

use std::sync::Arc;

use crate::materials::shadow_caster_policy_for_pipeline;
use crate::shared::ShadowCastMode;
use crate::world_mesh::{WorldMeshDrawItem, WorldMeshDrawList};

/// Whether this draw contributes depth to shadow maps.
fn shadow_draw_casts(item: &WorldMeshDrawItem) -> bool {
    item.shadow_cast_mode != ShadowCastMode::Off
        && shadow_caster_policy_for_pipeline(&item.batch_key.pipeline).casts()
}

/// Narrows a view's draws to the shadow casters, sharing the source array when they all cast.
pub(super) fn filter_shadow_caster_draws(items: &WorldMeshDrawList) -> WorldMeshDrawList {
    profiling::scope!("render::prepare_shadow_frame::filter_casters");
    // Scan once to the first non-caster. If there is none, the shared source is already the answer;
    // otherwise, the prefix is known good and can be copied in bulk.
    let Some(first_skipped) = items.iter().position(|item| !shadow_draw_casts(item)) else {
        return Arc::clone(items);
    };
    let mut kept = Vec::with_capacity(items.len().saturating_sub(1));
    kept.extend_from_slice(&items[..first_skipped]);
    kept.extend(
        items[first_skipped.saturating_add(1)..]
            .iter()
            .filter(|item| shadow_draw_casts(item))
            .cloned(),
    );
    kept.into()
}
