//! Layer assignment updates from host.

use crate::ipc::shared_memory::SharedMemoryAccessor;
use crate::scene::Scene;
use crate::shared::packing::enum_repr::EnumRepr;
use crate::shared::{LayerType, LayerUpdate};

use super::super::error::SceneError;
use super::super::pods::LayerAssignmentPod;

/// Applies layer updates from LayerUpdate. Parses layer_assignments (transform_id, layer_type)
/// pairs, removals (transform_ids to clear), and additions (transform_ids to add with default).
pub(crate) fn apply_layers_update(
    scene: &mut Scene,
    shm: &mut SharedMemoryAccessor,
    update: &LayerUpdate,
) -> Result<(), SceneError> {
    if update.removals.length > 0 {
        let ctx = format!("layers removals scene_id={}", scene.id);
        let removals = shm
            .access_with_context::<i32>(&update.removals, &ctx)
            .map_err(SceneError::SharedMemoryAccess)?;
        for &tid in removals.iter().take_while(|&&i| i >= 0) {
            scene.layer_assignments.remove(&tid);
        }
    }
    if update.additions.length > 0 {
        let ctx = format!("layers additions scene_id={}", scene.id);
        let additions = shm
            .access_with_context::<i32>(&update.additions, &ctx)
            .map_err(SceneError::SharedMemoryAccess)?;
        for &tid in additions.iter().take_while(|&&i| i >= 0) {
            scene
                .layer_assignments
                .entry(tid)
                .or_insert(LayerType::overlay);
        }
    }
    if update.layer_assignments.length > 0 {
        let ctx = format!("layers layer_assignments scene_id={}", scene.id);
        let assignments = shm
            .access_with_context::<LayerAssignmentPod>(&update.layer_assignments, &ctx)
            .map_err(SceneError::SharedMemoryAccess)?;
        for pod in assignments {
            if pod.transform_id < 0 {
                break;
            }
            let layer = LayerType::from_i32(pod.layer_type as i32);
            scene.layer_assignments.insert(pod.transform_id, layer);
        }
    }
    sync_drawable_layers(scene);
    Ok(())
}

/// Syncs drawable.layer from scene.layer_assignments. Call after applying layer updates.
pub(crate) fn sync_drawable_layers(scene: &mut Scene) {
    let default_layer = LayerType::overlay;
    for d in &mut scene.drawables {
        d.layer = scene
            .layer_assignments
            .get(&d.node_id)
            .copied()
            .unwrap_or(default_layer);
    }
    for d in &mut scene.skinned_drawables {
        d.layer = scene
            .layer_assignments
            .get(&d.node_id)
            .copied()
            .unwrap_or(default_layer);
    }
}
