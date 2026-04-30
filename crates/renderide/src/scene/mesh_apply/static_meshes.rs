//! Static mesh renderable updates: shared-memory extraction, dense apply, and tests.

use crate::ipc::SharedMemoryAccessor;
use crate::scene::dense_update::{non_negative_i32s, swap_remove_dense_indices};
use crate::scene::error::SceneError;
use crate::scene::mesh_material_row::apply_mesh_renderer_state_row;
use crate::scene::mesh_renderable::StaticMeshRenderer;
use crate::scene::render_space::RenderSpaceState;
use crate::shared::{
    LayerType, MESH_RENDERER_STATE_HOST_ROW_BYTES, MeshRenderablesUpdate, MeshRendererState,
};

use super::diagnostics::{STATIC_MESH_OOB_WARNED_SCENES, warn_oob_renderable_index_once};

/// Owned per-space static mesh-renderable update payload extracted from shared memory.
#[derive(Default, Debug)]
pub struct ExtractedMeshRenderablesUpdate {
    /// Static-mesh renderable removal indices (terminated by `< 0`).
    pub removals: Vec<i32>,
    /// New static-mesh renderable transform ids (terminated by `< 0`).
    pub additions: Vec<i32>,
    /// Per-renderer mesh state rows (terminated by `renderable_index < 0`).
    pub mesh_states: Vec<MeshRendererState>,
    /// Optional packed material/property-block id slab (`None` when host omitted the buffer).
    pub mesh_materials_and_property_blocks: Option<Vec<i32>>,
}

/// Reads every shared-memory buffer referenced by [`MeshRenderablesUpdate`] into owned vectors.
pub(crate) fn extract_mesh_renderables_update(
    shm: &mut SharedMemoryAccessor,
    update: &MeshRenderablesUpdate,
    scene_id: i32,
) -> Result<ExtractedMeshRenderablesUpdate, SceneError> {
    let mut out = ExtractedMeshRenderablesUpdate::default();
    if update.removals.length > 0 {
        let ctx = format!("mesh removals scene_id={scene_id}");
        out.removals = shm
            .access_copy_diagnostic_with_context::<i32>(&update.removals, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    if update.additions.length > 0 {
        let ctx = format!("mesh additions scene_id={scene_id}");
        out.additions = shm
            .access_copy_diagnostic_with_context::<i32>(&update.additions, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    if update.mesh_states.length > 0 {
        let ctx = format!("mesh mesh_states scene_id={scene_id}");
        out.mesh_states = shm
            .access_copy_memory_packable_rows::<MeshRendererState>(
                &update.mesh_states,
                MESH_RENDERER_STATE_HOST_ROW_BYTES,
                Some(&ctx),
            )
            .map_err(SceneError::SharedMemoryAccess)?;
        if update.mesh_materials_and_property_blocks.length > 0 {
            let ctx_m = format!("mesh mesh_materials_and_property_blocks scene_id={scene_id}");
            out.mesh_materials_and_property_blocks = Some(
                shm.access_copy_diagnostic_with_context::<i32>(
                    &update.mesh_materials_and_property_blocks,
                    Some(&ctx_m),
                )
                .map_err(SceneError::SharedMemoryAccess)?,
            );
        }
    }
    Ok(out)
}

/// Mutates [`RenderSpaceState::static_mesh_renderers`] using a pre-extracted payload.
pub(crate) fn apply_mesh_renderables_update_extracted(
    space: &mut RenderSpaceState,
    extracted: &ExtractedMeshRenderablesUpdate,
    scene_id: i32,
) {
    profiling::scope!("scene::apply_meshes");
    swap_remove_dense_indices(&mut space.static_mesh_renderers, &extracted.removals);
    for node_id in non_negative_i32s(&extracted.additions) {
        let instance_id = space.allocate_mesh_renderer_instance_id();
        space.static_mesh_renderers.push(StaticMeshRenderer {
            instance_id,
            node_id,
            layer: LayerType::Hidden,
            ..Default::default()
        });
    }
    let packed_ref = extracted.mesh_materials_and_property_blocks.as_deref();
    let mut packed_cursor = 0usize;
    let len = space.static_mesh_renderers.len();
    for state in &extracted.mesh_states {
        if state.renderable_index < 0 {
            break;
        }
        let idx = state.renderable_index as usize;
        let drawable = space.static_mesh_renderers.get_mut(idx);
        if drawable.is_none() {
            warn_oob_renderable_index_once(
                scene_id,
                "static",
                idx,
                len,
                &STATIC_MESH_OOB_WARNED_SCENES,
            );
        }
        apply_mesh_renderer_state_row(drawable, state, packed_ref, &mut packed_cursor);
    }
}

#[cfg(test)]
mod tests {
    use crate::scene::mesh_renderable::MeshRendererInstanceId;
    use crate::scene::render_space::RenderSpaceState;

    use super::{ExtractedMeshRenderablesUpdate, apply_mesh_renderables_update_extracted};

    #[test]
    fn static_instance_ids_are_fresh_and_survive_swap_remove() {
        let mut space = RenderSpaceState::default();
        apply_mesh_renderables_update_extracted(
            &mut space,
            &ExtractedMeshRenderablesUpdate {
                additions: vec![10, 11, 12, -1],
                ..Default::default()
            },
            1,
        );
        let ids: Vec<_> = space
            .static_mesh_renderers
            .iter()
            .map(|renderer| renderer.instance_id)
            .collect();
        assert_eq!(
            ids,
            vec![
                MeshRendererInstanceId(1),
                MeshRendererInstanceId(2),
                MeshRendererInstanceId(3),
            ]
        );

        apply_mesh_renderables_update_extracted(
            &mut space,
            &ExtractedMeshRenderablesUpdate {
                removals: vec![1, -1],
                additions: vec![13, -1],
                ..Default::default()
            },
            1,
        );
        let ids: Vec<_> = space
            .static_mesh_renderers
            .iter()
            .map(|renderer| renderer.instance_id)
            .collect();
        assert_eq!(
            ids,
            vec![
                MeshRendererInstanceId(1),
                MeshRendererInstanceId(3),
                MeshRendererInstanceId(4),
            ]
        );
    }
}
