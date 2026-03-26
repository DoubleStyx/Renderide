//! Mesh renderable updates from host.
//!
//! Removal indices are processed in **buffer order** (like Unity `RenderableManager.HandleUpdate`),
//! using swap-with-last per entry. Do not sort removals descending.

use crate::ipc::shared_memory::SharedMemoryAccessor;
use crate::scene::{Drawable, Scene};
use crate::shared::{LayerType, MeshRenderablesUpdate, ShadowCastMode};

use super::super::error::SceneError;
use super::super::pods::MeshRendererStatePod;
use super::mesh_material_slots::apply_mesh_renderer_state_row;

/// Applies mesh renderable updates: removals, additions, mesh states.
pub(crate) fn apply_mesh_renderables_update(
    scene: &mut Scene,
    shm: &mut SharedMemoryAccessor,
    update: &MeshRenderablesUpdate,
    _frame_index: i32,
) -> Result<(), SceneError> {
    if update.removals.length > 0 {
        let ctx = format!("mesh removals scene_id={}", scene.id);
        let removals = shm
            .access_with_context::<i32>(&update.removals, &ctx)
            .map_err(SceneError::SharedMemoryAccess)?;
        for &raw in removals.iter().take_while(|&&i| i >= 0) {
            let idx = raw as usize;
            if idx < scene.drawables.len() {
                scene.drawables.swap_remove(idx);
            }
        }
    }
    if update.additions.length > 0 {
        let ctx = format!("mesh additions scene_id={}", scene.id);
        let additions = shm
            .access_with_context::<i32>(&update.additions, &ctx)
            .map_err(SceneError::SharedMemoryAccess)?;
        let added_node_ids: Vec<i32> = additions.iter().take_while(|&&i| i >= 0).copied().collect();
        for &node_id in &added_node_ids {
            let layer = scene
                .layer_assignments
                .get(&node_id)
                .copied()
                .unwrap_or(LayerType::overlay);
            scene.drawables.push(Drawable {
                node_id,
                layer,
                mesh_handle: -1,
                material_handle: None,
                sort_key: 0,
                is_skinned: false,
                bone_transform_ids: None,
                root_bone_transform_id: None,
                blend_shape_weights: None,
                stencil_state: None,
                material_override_block_id: None,
                mesh_renderer_property_block_slot0_id: None,
                material_slots: Vec::new(),
                render_transform_override: None,
                shadow_cast_mode: ShadowCastMode::on,
            });
        }
    }
    if update.mesh_states.length > 0 {
        let ctx = format!("mesh mesh_states scene_id={}", scene.id);
        let states = shm
            .access_with_context::<MeshRendererStatePod>(&update.mesh_states, &ctx)
            .map_err(SceneError::SharedMemoryAccess)?;
        let packed_ids = if update.mesh_materials_and_property_blocks.length > 0 {
            let ctx_m = format!(
                "mesh mesh_materials_and_property_blocks scene_id={}",
                scene.id
            );
            Some(
                shm.access_with_context::<i32>(&update.mesh_materials_and_property_blocks, &ctx_m)
                    .map_err(SceneError::SharedMemoryAccess)?,
            )
        } else {
            None
        };
        let packed_ref = packed_ids.as_deref();
        let mut packed_cursor = 0usize;
        for state in states {
            if state.renderable_index < 0 {
                break;
            }
            let idx = state.renderable_index as usize;
            let drawable = scene.drawables.get_mut(idx);
            apply_mesh_renderer_state_row(drawable, &state, packed_ref, &mut packed_cursor);
        }
    }
    Ok(())
}
