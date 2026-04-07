//! Render material override updates from host.

use crate::ipc::shared_memory::SharedMemoryAccessor;
use crate::scene::Scene;
use crate::shared::{RenderMaterialOverrideState, RenderMaterialOverridesUpdate};

use super::super::error::SceneError;
use super::super::pods::MaterialOverrideStatePod;

/// Applies render material overrides from host. Maps each renderable to its first material
/// override's material_asset_id for later stencil lookup. Uses material_override_states and
/// states buffers; states are laid out contiguously per renderable (materrial_override_count
/// entries each).
pub(crate) fn apply_render_material_overrides_update(
    scene: &mut Scene,
    shm: &mut SharedMemoryAccessor,
    update: &RenderMaterialOverridesUpdate,
) -> Result<(), SceneError> {
    if update.material_override_states.length <= 0 || update.states.length <= 0 {
        return Ok(());
    }
    let ctx_override = format!(
        "material_overrides material_override_states scene_id={}",
        scene.id
    );
    let override_states = shm
        .access_with_context::<RenderMaterialOverrideState>(
            &update.material_override_states,
            &ctx_override,
        )
        .map_err(SceneError::SharedMemoryAccess)?;
    let ctx_states = format!("material_overrides states scene_id={}", scene.id);
    let states = shm
        .access_with_context::<MaterialOverrideStatePod>(&update.states, &ctx_states)
        .map_err(SceneError::SharedMemoryAccess)?;

    let mesh_count = scene.drawables.len();
    let skinned_count = scene.skinned_drawables.len();
    let total_renderables = mesh_count + skinned_count;

    let mut state_offset = 0;
    for ov in &override_states {
        if ov.renderable_index < 0 {
            break;
        }
        let count = ov.materrial_override_count.max(0) as usize;
        if state_offset + count > states.len() {
            break;
        }
        let material_asset_id = if count > 0 {
            Some(states[state_offset].material_asset_id)
        } else {
            None
        };
        state_offset += count;

        let idx = ov.renderable_index as usize;
        if idx >= total_renderables {
            continue;
        }
        if idx < mesh_count {
            scene.drawables[idx].material_override_block_id = material_asset_id;
        } else {
            scene.skinned_drawables[idx - mesh_count].material_override_block_id =
                material_asset_id;
        }
    }
    Ok(())
}
