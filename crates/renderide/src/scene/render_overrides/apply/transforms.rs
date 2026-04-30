//! Shared-memory extraction and dense apply for [`RenderTransformOverridesUpdate`].

use crate::ipc::SharedMemoryAccessor;
use crate::scene::dense_update::swap_remove_dense_indices;
use crate::scene::error::SceneError;
use crate::scene::render_overrides::types::RenderTransformOverrideEntry;
use crate::scene::render_space::RenderSpaceState;
use crate::scene::transforms_apply::TransformRemovalEvent;
use crate::shared::{
    RENDER_TRANSFORM_OVERRIDE_STATE_HOST_ROW_BYTES, RenderTransformOverrideState,
    RenderTransformOverridesUpdate,
};

use super::fixup::fixup_override_nodes_for_transform_removals;

/// Owned per-space transform-override payload extracted from shared memory.
#[derive(Default, Debug)]
pub struct ExtractedRenderTransformOverridesUpdate {
    /// Dense override-entry removal indices (terminated by `< 0`).
    pub removals: Vec<i32>,
    /// New override-entry node ids (terminated by `< 0`).
    pub additions: Vec<i32>,
    /// Per-entry override state rows (terminated by `renderable_index < 0`).
    pub states: Vec<RenderTransformOverrideState>,
    /// Skinned-mesh renderer index slab keyed positionally by `skinned_mesh_renderer_count`.
    pub skinned_mesh_renderers_indexes: Vec<i32>,
}

/// Reads every shared-memory buffer referenced by [`RenderTransformOverridesUpdate`] into owned vectors.
pub(crate) fn extract_render_transform_overrides_update(
    shm: &mut SharedMemoryAccessor,
    update: &RenderTransformOverridesUpdate,
    scene_id: i32,
) -> Result<ExtractedRenderTransformOverridesUpdate, SceneError> {
    let mut out = ExtractedRenderTransformOverridesUpdate::default();
    if update.removals.length > 0 {
        let ctx = format!("render transform override removals scene_id={scene_id}");
        out.removals = shm
            .access_copy_diagnostic_with_context::<i32>(&update.removals, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    if update.additions.length > 0 {
        let ctx = format!("render transform override additions scene_id={scene_id}");
        out.additions = shm
            .access_copy_diagnostic_with_context::<i32>(&update.additions, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    if update.states.length > 0 {
        let ctx = format!("render transform override states scene_id={scene_id}");
        out.states = shm
            .access_copy_memory_packable_rows::<RenderTransformOverrideState>(
                &update.states,
                RENDER_TRANSFORM_OVERRIDE_STATE_HOST_ROW_BYTES,
                Some(&ctx),
            )
            .map_err(SceneError::SharedMemoryAccess)?;
        if update.skinned_mesh_renderers_indexes.length > 0 {
            let ctx = format!("render transform override skinned mesh indexes scene_id={scene_id}");
            out.skinned_mesh_renderers_indexes = shm
                .access_copy_diagnostic_with_context::<i32>(
                    &update.skinned_mesh_renderers_indexes,
                    Some(&ctx),
                )
                .map_err(SceneError::SharedMemoryAccess)?;
        }
    }
    Ok(out)
}

/// Mutates [`RenderSpaceState::render_transform_overrides`] using pre-extracted payloads.
///
/// Pre-runs the transform-removal id fixup so removed slots roll forward to the swapped index.
pub(crate) fn apply_render_transform_overrides_update_extracted(
    space: &mut RenderSpaceState,
    extracted: &ExtractedRenderTransformOverridesUpdate,
    transform_removals: &[TransformRemovalEvent],
) {
    profiling::scope!("scene::apply_render_transform_overrides");
    fixup_override_nodes_for_transform_removals(
        &mut space.render_transform_overrides,
        transform_removals,
    );

    swap_remove_dense_indices(&mut space.render_transform_overrides, &extracted.removals);

    for &node_id in extracted.additions.iter().take_while(|&&id| id >= 0) {
        space
            .render_transform_overrides
            .push(RenderTransformOverrideEntry {
                node_id,
                ..Default::default()
            });
    }

    let skinned_indices = &extracted.skinned_mesh_renderers_indexes;
    let mut skinned_cursor = 0usize;
    for state in &extracted.states {
        if state.renderable_index < 0 {
            break;
        }
        let idx = state.renderable_index as usize;
        let Some(entry) = space.render_transform_overrides.get_mut(idx) else {
            continue;
        };
        entry.context = state.context;
        entry.position_override =
            ((state.override_flags & 0b001) != 0).then_some(state.position_override);
        entry.rotation_override =
            ((state.override_flags & 0b010) != 0).then_some(state.rotation_override);
        entry.scale_override =
            ((state.override_flags & 0b100) != 0).then_some(state.scale_override);
        let count = state.skinned_mesh_renderer_count.max(0) as usize;
        entry.skinned_mesh_renderer_indices.clear();
        if count > 0 {
            let end = skinned_cursor
                .saturating_add(count)
                .min(skinned_indices.len());
            entry
                .skinned_mesh_renderer_indices
                .extend_from_slice(&skinned_indices[skinned_cursor..end]);
            skinned_cursor = end;
        }
    }
}
