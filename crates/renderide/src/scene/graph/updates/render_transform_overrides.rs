//! Render transform override updates from host.
//!
//! Applies per-skinned-mesh-renderer transform overrides (e.g. cloud-spawned avatars).
//! When set, the override replaces the node's world matrix when rendering.

use crate::ipc::shared_memory::SharedMemoryAccessor;
use crate::scene::Scene;
use crate::shared::{
    RenderTransform, RenderTransformOverrideState, RenderTransformOverridesUpdate,
};

use super::super::error::SceneError;

/// Applies render transform overrides from host.
///
/// Sets `render_transform_override` on skinned drawables when the host sends override states.
/// Clears overrides for renderables in the removals list.
pub(crate) fn apply_render_transform_overrides_update(
    scene: &mut Scene,
    shm: &mut SharedMemoryAccessor,
    update: &RenderTransformOverridesUpdate,
) -> Result<(), SceneError> {
    let i32_size = std::mem::size_of::<i32>() as i32;
    let state_size = std::mem::size_of::<RenderTransformOverrideState>() as i32;

    if update.removals.length >= i32_size {
        let ctx = format!("render_transform_overrides removals scene_id={}", scene.id);
        let removals = shm
            .access_copy_diagnostic_with_context::<i32>(&update.removals, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
        for &idx in removals.iter().take_while(|&&i| i >= 0) {
            let idx = idx as usize;
            if idx < scene.skinned_drawables.len() {
                scene.skinned_drawables[idx].render_transform_override = None;
            }
        }
    }

    if update.states.length >= state_size {
        let ctx = format!("render_transform_overrides states scene_id={}", scene.id);
        let states = shm
            .access_copy_diagnostic_with_context::<RenderTransformOverrideState>(
                &update.states,
                Some(&ctx),
            )
            .map_err(SceneError::SharedMemoryAccess)?;
        for state in &states {
            if state.renderable_index < 0 {
                break;
            }
            let idx = state.renderable_index as usize;
            if idx < scene.skinned_drawables.len() {
                scene.skinned_drawables[idx].render_transform_override = Some(RenderTransform {
                    position: state.position_override,
                    scale: state.scale_override,
                    rotation: state.rotation_override,
                });
            }
        }
    }

    Ok(())
}
