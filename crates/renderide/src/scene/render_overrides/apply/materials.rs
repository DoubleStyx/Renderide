//! Shared-memory extraction and dense apply for [`RenderMaterialOverridesUpdate`].

use crate::ipc::SharedMemoryAccessor;
use crate::scene::dense_update::swap_remove_dense_indices;
use crate::scene::error::SceneError;
use crate::scene::render_overrides::types::{
    MaterialOverrideBinding, RenderMaterialOverrideEntry, decode_packed_mesh_renderer_target,
};
use crate::scene::render_space::RenderSpaceState;
use crate::scene::transforms_apply::TransformRemovalEvent;
use crate::shared::{
    MaterialOverrideState, RENDER_MATERIAL_OVERRIDE_STATE_HOST_ROW_BYTES,
    RenderMaterialOverrideState, RenderMaterialOverridesUpdate,
};

use super::fixup::fixup_override_nodes_for_transform_removals;

/// Owned per-space material-override payload extracted from shared memory.
#[derive(Default, Debug)]
pub struct ExtractedRenderMaterialOverridesUpdate {
    /// Dense override-entry removal indices (terminated by `< 0`).
    pub removals: Vec<i32>,
    /// New override-entry node ids (terminated by `< 0`).
    pub additions: Vec<i32>,
    /// Per-entry override state rows (terminated by `renderable_index < 0`).
    pub states: Vec<RenderMaterialOverrideState>,
    /// Material-override row slab keyed positionally by `materrial_override_count`.
    pub material_override_states: Vec<MaterialOverrideState>,
}

/// Reads every shared-memory buffer referenced by [`RenderMaterialOverridesUpdate`] into owned vectors.
pub(crate) fn extract_render_material_overrides_update(
    shm: &mut SharedMemoryAccessor,
    update: &RenderMaterialOverridesUpdate,
    scene_id: i32,
) -> Result<ExtractedRenderMaterialOverridesUpdate, SceneError> {
    let mut out = ExtractedRenderMaterialOverridesUpdate::default();
    if update.removals.length > 0 {
        let ctx = format!("render material override removals scene_id={scene_id}");
        out.removals = shm
            .access_copy_diagnostic_with_context::<i32>(&update.removals, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    if update.additions.length > 0 {
        let ctx = format!("render material override additions scene_id={scene_id}");
        out.additions = shm
            .access_copy_diagnostic_with_context::<i32>(&update.additions, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    if update.states.length > 0 {
        let ctx = format!("render material override states scene_id={scene_id}");
        out.states = shm
            .access_copy_memory_packable_rows::<RenderMaterialOverrideState>(
                &update.states,
                RENDER_MATERIAL_OVERRIDE_STATE_HOST_ROW_BYTES,
                Some(&ctx),
            )
            .map_err(SceneError::SharedMemoryAccess)?;
        if update.material_override_states.length > 0 {
            let ctx = format!("render material override rows scene_id={scene_id}");
            out.material_override_states = shm
                .access_copy_diagnostic_with_context::<MaterialOverrideState>(
                    &update.material_override_states,
                    Some(&ctx),
                )
                .map_err(SceneError::SharedMemoryAccess)?;
        }
    }
    Ok(out)
}

/// Mutates [`RenderSpaceState::render_material_overrides`] using pre-extracted payloads.
pub(crate) fn apply_render_material_overrides_update_extracted(
    space: &mut RenderSpaceState,
    extracted: &ExtractedRenderMaterialOverridesUpdate,
    transform_removals: &[TransformRemovalEvent],
) {
    profiling::scope!("scene::apply_render_material_overrides");
    fixup_override_nodes_for_transform_removals(
        &mut space.render_material_overrides,
        transform_removals,
    );

    swap_remove_dense_indices(&mut space.render_material_overrides, &extracted.removals);

    for &node_id in extracted.additions.iter().take_while(|&&id| id >= 0) {
        space
            .render_material_overrides
            .push(RenderMaterialOverrideEntry {
                node_id,
                ..Default::default()
            });
    }

    let materials = &extracted.material_override_states;
    let mut material_cursor = 0usize;
    for state in &extracted.states {
        if state.renderable_index < 0 {
            break;
        }
        let idx = state.renderable_index as usize;
        let Some(entry) = space.render_material_overrides.get_mut(idx) else {
            continue;
        };
        entry.context = state.context;
        entry.target = decode_packed_mesh_renderer_target(state.packed_mesh_renderer_index);
        let count = state.materrial_override_count.max(0) as usize;
        entry.material_overrides.clear();
        if count > 0 {
            let end = material_cursor.saturating_add(count).min(materials.len());
            entry
                .material_overrides
                .extend(materials[material_cursor..end].iter().map(|row| {
                    MaterialOverrideBinding {
                        material_slot_index: row.material_slot_index,
                        material_asset_id: row.material_asset_id,
                    }
                }));
            material_cursor = end;
        }
    }
}
