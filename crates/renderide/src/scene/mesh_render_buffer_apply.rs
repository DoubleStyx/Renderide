//! Mesh-render-buffer state ingestion (`RenderSpaceUpdate.mesh_render_buffers_update`).

use crate::ipc::SharedMemoryAccessor;
use crate::shared::{
    MESH_RENDER_BUFFER_STATE_HOST_ROW_BYTES, MeshRenderBufferState, MeshRenderBufferUpdate,
};

use super::dense_update::{push_dense_additions, swap_remove_dense_indices};
use super::error::SceneError;
use super::render_space::RenderSpaceState;
use super::transforms_apply::TransformRemovalEvent;
use super::world::fixup_transform_id;

/// Owned per-space mesh-render-buffer payload extracted from shared memory.
#[derive(Default, Debug)]
pub struct ExtractedMeshRenderBufferUpdate {
    /// Dense row removals (terminated by `< 0`).
    pub removals: Vec<i32>,
    /// Dense row additions (terminated by `< 0`).
    pub additions: Vec<i32>,
    /// Dense mesh-render-buffer rows.
    pub states: Vec<MeshRenderBufferState>,
}

/// Reads every shared-memory buffer referenced by [`MeshRenderBufferUpdate`] into owned vectors.
pub(crate) fn extract_mesh_render_buffer_update(
    shm: &mut SharedMemoryAccessor,
    update: &MeshRenderBufferUpdate,
    scene_id: i32,
) -> Result<ExtractedMeshRenderBufferUpdate, SceneError> {
    let mut out = ExtractedMeshRenderBufferUpdate::default();
    if update.removals.length > 0 {
        let ctx = format!("mesh render buffer removals scene_id={scene_id}");
        out.removals = shm
            .access_copy_diagnostic_with_context::<i32>(&update.removals, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    if update.additions.length > 0 {
        let ctx = format!("mesh render buffer additions scene_id={scene_id}");
        out.additions = shm
            .access_copy_diagnostic_with_context::<i32>(&update.additions, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    if update.states.length > 0 {
        let ctx = format!("mesh render buffer states scene_id={scene_id}");
        out.states = shm
            .access_copy_memory_packable_rows::<MeshRenderBufferState>(
                &update.states,
                MESH_RENDER_BUFFER_STATE_HOST_ROW_BYTES,
                Some(&ctx),
            )
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    Ok(out)
}

/// Mutates [`RenderSpaceState::mesh_render_buffers`] from pre-extracted rows.
pub(crate) fn apply_mesh_render_buffer_update_extracted(
    space: &mut RenderSpaceState,
    extracted: &ExtractedMeshRenderBufferUpdate,
) {
    swap_remove_dense_indices(&mut space.mesh_render_buffers, &extracted.removals);
    push_dense_additions(
        &mut space.mesh_render_buffers,
        &extracted.additions,
        |_id| MeshRenderBufferState::default(),
    );
    for (idx, state) in extracted.states.iter().copied().enumerate() {
        let Some(slot) = space.mesh_render_buffers.get_mut(idx) else {
            continue;
        };
        *slot = state;
    }
}

/// Rewrites transform ids in mesh-render-buffer states after transform swap-removals.
pub(crate) fn fixup_mesh_render_buffers_for_transform_removals(
    space: &mut RenderSpaceState,
    removals: &[TransformRemovalEvent],
) {
    if removals.is_empty() {
        return;
    }
    for removal in removals {
        for state in &mut space.mesh_render_buffers {
            state.renderable_index = fixup_transform_id(
                state.renderable_index,
                removal.removed_index,
                removal.last_index_before_swap,
            );
        }
        space
            .mesh_render_buffers
            .retain(|state| state.renderable_index >= 0);
    }
}
