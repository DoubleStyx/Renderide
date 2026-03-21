//! Mesh renderable updates from host.
//!
//! Removal indices are processed in **buffer order** (like Unity `RenderableManager.HandleUpdate`),
//! using swap-with-last per entry. Do not sort removals descending.

use crate::ipc::shared_memory::SharedMemoryAccessor;
use crate::scene::{Drawable, Scene};
use crate::shared::enum_repr::EnumRepr;
use crate::shared::{LayerType, MeshRenderablesUpdate, ShadowCastMode};

use super::super::error::SceneError;
use super::super::pods::MeshRendererStatePod;

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
        for state in states {
            if state.renderable_index < 0 {
                break;
            }
            let idx = state.renderable_index as usize;
            if idx < scene.drawables.len() {
                scene.drawables[idx].mesh_handle = state.mesh_asset_id;
                scene.drawables[idx].sort_key = state.sorting_order;
                scene.drawables[idx].material_handle = if state.material_count > 0 {
                    Some(-1)
                } else {
                    None
                };
                scene.drawables[idx].shadow_cast_mode =
                    ShadowCastMode::from_i32(state.shadow_cast_mode as i32);
            }
        }
    }
    Ok(())
}
