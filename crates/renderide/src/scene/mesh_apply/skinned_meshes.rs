//! Skinned mesh renderable updates: extraction, dense apply orchestration, and the bone /
//! blendshape / bounds sub-applies that the orchestration calls.

use crate::ipc::SharedMemoryAccessor;
use crate::scene::dense_update::{non_negative_i32s, swap_remove_dense_indices};
use crate::scene::error::SceneError;
use crate::scene::mesh_material_row::apply_mesh_renderer_state_row;
use crate::scene::mesh_renderable::{SkinnedMeshRenderer, StaticMeshRenderer};
use crate::scene::render_space::RenderSpaceState;
use crate::scene::transforms_apply::TransformRemovalEvent;
use crate::shared::packing_extras::SKINNED_MESH_BOUNDS_UPDATE_HOST_ROW_BYTES;
use crate::shared::{
    BlendshapeUpdate, BlendshapeUpdateBatch, BoneAssignment, LayerType,
    MESH_RENDERER_STATE_HOST_ROW_BYTES, MeshRendererState, SkinnedMeshBoundsUpdate,
    SkinnedMeshRenderablesUpdate,
};

use super::diagnostics::{
    BONE_INDEX_EMPTY_WARNED_SCENES, SKINNED_MESH_OOB_WARNED_SCENES, warn_oob_renderable_index_once,
};
use super::fixups::fixup_skinned_bones_for_transform_removals;

/// Owned per-space skinned mesh-renderable update payload extracted from shared memory.
#[derive(Default, Debug)]
pub struct ExtractedSkinnedMeshRenderablesUpdate {
    /// Skinned-mesh renderable removal indices (terminated by `< 0`).
    pub removals: Vec<i32>,
    /// New skinned-mesh renderable transform ids (terminated by `< 0`).
    pub additions: Vec<i32>,
    /// Per-renderer mesh state rows (terminated by `renderable_index < 0`).
    pub mesh_states: Vec<MeshRendererState>,
    /// Optional packed material/property-block id slab (`None` when host omitted the buffer).
    pub mesh_materials_and_property_blocks: Option<Vec<i32>>,
    /// Per-renderer bone-assignment row (terminated by `renderable_index < 0`).
    pub bone_assignments: Vec<BoneAssignment>,
    /// Bone transform-index slab keyed by [`BoneAssignment::bone_count`].
    pub bone_transform_indexes: Vec<i32>,
    /// Per-renderer blendshape batch row (terminated by `renderable_index < 0`).
    pub blendshape_update_batches: Vec<BlendshapeUpdateBatch>,
    /// Blendshape weight delta slab keyed by [`BlendshapeUpdateBatch::blendshape_update_count`].
    pub blendshape_updates: Vec<BlendshapeUpdate>,
    /// Per-renderer posed object-space AABB rows from the host's
    /// [`SkinnedMeshRenderablesUpdate::bounds_updates`] buffer (terminated by
    /// `renderable_index < 0`). Each row carries the tight per-frame AABB computed by the host's
    /// animation evaluation and is used verbatim for CPU frustum / Hi-Z culling.
    pub bounds_updates: Vec<SkinnedMeshBoundsUpdate>,
}

/// Maximum blendshape index accepted from IPC blendshape weight updates.
///
/// Matches the cap enforced by [`crate::assets::mesh::layout`] when extracting blendshape
/// data; updates referencing higher indices are silently dropped to prevent attacker-driven
/// `Vec::resize` on the per-renderable weight array.
const MAX_BLENDSHAPE_INDEX: usize = 4096;

/// Reads every shared-memory buffer referenced by [`SkinnedMeshRenderablesUpdate`] into owned vectors.
pub(crate) fn extract_skinned_mesh_renderables_update(
    shm: &mut SharedMemoryAccessor,
    update: &SkinnedMeshRenderablesUpdate,
    scene_id: i32,
) -> Result<ExtractedSkinnedMeshRenderablesUpdate, SceneError> {
    let mut out = ExtractedSkinnedMeshRenderablesUpdate::default();
    if update.removals.length > 0 {
        let ctx = format!("skinned removals scene_id={scene_id}");
        out.removals = shm
            .access_copy_diagnostic_with_context::<i32>(&update.removals, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    if update.additions.length > 0 {
        let ctx = format!("skinned additions scene_id={scene_id}");
        out.additions = shm
            .access_copy_diagnostic_with_context::<i32>(&update.additions, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    if update.mesh_states.length > 0 {
        let ctx = format!("skinned mesh_states scene_id={scene_id}");
        out.mesh_states = shm
            .access_copy_memory_packable_rows::<MeshRendererState>(
                &update.mesh_states,
                MESH_RENDERER_STATE_HOST_ROW_BYTES,
                Some(&ctx),
            )
            .map_err(SceneError::SharedMemoryAccess)?;
        if update.mesh_materials_and_property_blocks.length > 0 {
            let ctx_m = format!("skinned mesh_materials_and_property_blocks scene_id={scene_id}");
            out.mesh_materials_and_property_blocks = Some(
                shm.access_copy_diagnostic_with_context::<i32>(
                    &update.mesh_materials_and_property_blocks,
                    Some(&ctx_m),
                )
                .map_err(SceneError::SharedMemoryAccess)?,
            );
        }
    }
    if update.bone_assignments.length > 0 {
        let ctx_assign = format!("skinned bone_assignments scene_id={scene_id}");
        out.bone_assignments = shm
            .access_copy_diagnostic_with_context::<BoneAssignment>(
                &update.bone_assignments,
                Some(&ctx_assign),
            )
            .map_err(SceneError::SharedMemoryAccess)?;
        if update.bone_transform_indexes.length > 0 {
            let ctx_idx = format!("skinned bone_transform_indexes scene_id={scene_id}");
            out.bone_transform_indexes = shm
                .access_copy_diagnostic_with_context::<i32>(
                    &update.bone_transform_indexes,
                    Some(&ctx_idx),
                )
                .map_err(SceneError::SharedMemoryAccess)?;
        }
    }
    if update.blendshape_update_batches.length > 0 && update.blendshape_updates.length > 0 {
        let ctx_batch = format!("skinned blendshape_update_batches scene_id={scene_id}");
        out.blendshape_update_batches = shm
            .access_copy_diagnostic_with_context::<BlendshapeUpdateBatch>(
                &update.blendshape_update_batches,
                Some(&ctx_batch),
            )
            .map_err(SceneError::SharedMemoryAccess)?;
        let ctx_upd = format!("skinned blendshape_updates scene_id={scene_id}");
        out.blendshape_updates = shm
            .access_copy_diagnostic_with_context::<BlendshapeUpdate>(
                &update.blendshape_updates,
                Some(&ctx_upd),
            )
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    if update.bounds_updates.length > 0 {
        let ctx_bounds = format!("skinned bounds_updates scene_id={scene_id}");
        out.bounds_updates = shm
            .access_copy_memory_packable_rows::<SkinnedMeshBoundsUpdate>(
                &update.bounds_updates,
                SKINNED_MESH_BOUNDS_UPDATE_HOST_ROW_BYTES,
                Some(&ctx_bounds),
            )
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    Ok(out)
}

/// Skinned renderable removals and additive spawn (dense indices).
fn apply_skinned_removals_and_additions_extracted(
    space: &mut RenderSpaceState,
    extracted: &ExtractedSkinnedMeshRenderablesUpdate,
) {
    profiling::scope!("scene::apply_skinned_removals_additions");
    swap_remove_dense_indices(&mut space.skinned_mesh_renderers, &extracted.removals);
    for node_id in non_negative_i32s(&extracted.additions) {
        let instance_id = space.allocate_mesh_renderer_instance_id();
        space.skinned_mesh_renderers.push(SkinnedMeshRenderer {
            base: StaticMeshRenderer {
                instance_id,
                node_id,
                layer: LayerType::Hidden,
                ..Default::default()
            },
            ..Default::default()
        });
    }
}

/// Applies per-skinned-renderable [`MeshRendererState`] rows and optional packed material lists.
fn apply_skinned_mesh_state_rows_extracted(
    space: &mut RenderSpaceState,
    extracted: &ExtractedSkinnedMeshRenderablesUpdate,
    scene_id: i32,
) {
    profiling::scope!("scene::apply_skinned_state_rows");
    if extracted.mesh_states.is_empty() {
        return;
    }
    let packed_ref = extracted.mesh_materials_and_property_blocks.as_deref();
    let mut packed_cursor = 0usize;
    let len = space.skinned_mesh_renderers.len();
    for state in &extracted.mesh_states {
        if state.renderable_index < 0 {
            break;
        }
        let idx = state.renderable_index as usize;
        let drawable = space.skinned_mesh_renderers.get_mut(idx);
        if drawable.is_none() {
            warn_oob_renderable_index_once(
                scene_id,
                "skinned",
                idx,
                len,
                &SKINNED_MESH_OOB_WARNED_SCENES,
            );
        }
        apply_mesh_renderer_state_row(drawable, state, packed_ref, &mut packed_cursor);
    }
}

/// Writes bone index lists from paired assignment / index buffers.
fn apply_skinned_bone_index_buffers_extracted(
    space: &mut RenderSpaceState,
    extracted: &ExtractedSkinnedMeshRenderablesUpdate,
    scene_id: i32,
) {
    profiling::scope!("scene::apply_skinned_bone_indices");
    if extracted.bone_assignments.is_empty() {
        return;
    }
    if extracted.bone_transform_indexes.is_empty() {
        let should_warn = BONE_INDEX_EMPTY_WARNED_SCENES.lock().insert(scene_id);
        if should_warn {
            logger::warn!(
                "Skinned update: bone_assignments present but bone_transform_indexes empty (scene_id={scene_id}); skipping bone index application"
            );
        }
        return;
    }
    let indexes = &extracted.bone_transform_indexes;
    let mut index_offset = 0usize;
    for assignment in &extracted.bone_assignments {
        if assignment.renderable_index < 0 {
            break;
        }
        let idx = assignment.renderable_index as usize;
        let bone_count = assignment.bone_count.max(0) as usize;
        let Some(end) = index_offset.checked_add(bone_count) else {
            break;
        };
        if idx < space.skinned_mesh_renderers.len() && end <= indexes.len() {
            let ids: Vec<i32> = indexes[index_offset..end].to_vec();
            space.skinned_mesh_renderers[idx].bone_transform_indices = ids;
            space.skinned_mesh_renderers[idx].root_bone_transform_id =
                (assignment.root_bone_transform_id >= 0)
                    .then_some(assignment.root_bone_transform_id);
        }
        index_offset = end;
    }
}

/// Applies batched blendshape weight deltas into per-renderable weight vectors.
fn apply_skinned_blendshape_weight_batches_extracted(
    space: &mut RenderSpaceState,
    extracted: &ExtractedSkinnedMeshRenderablesUpdate,
) {
    profiling::scope!("scene::apply_skinned_blendshape_weights");
    if extracted.blendshape_update_batches.is_empty() || extracted.blendshape_updates.is_empty() {
        return;
    }
    let updates = &extracted.blendshape_updates;
    let mut update_offset = 0usize;
    for batch in &extracted.blendshape_update_batches {
        if batch.renderable_index < 0 {
            break;
        }
        let idx = batch.renderable_index as usize;
        let count = batch.blendshape_update_count.max(0) as usize;
        let Some(end) = update_offset.checked_add(count) else {
            break;
        };
        if idx < space.skinned_mesh_renderers.len() && end <= updates.len() {
            let weights = &mut space.skinned_mesh_renderers[idx].base.blend_shape_weights;
            for upd in &updates[update_offset..end] {
                let bi = upd.blendshape_index.max(0) as usize;
                if bi >= MAX_BLENDSHAPE_INDEX {
                    continue;
                }
                let needed = bi + 1;
                if weights.len() < needed {
                    weights.resize(needed, 0.0);
                }
                weights[bi] = upd.weight;
            }
        }
        update_offset = end;
    }
}

/// Stores host-computed posed object-space bounds onto skinned renderables for culling.
///
/// The host emits one row per renderable whose `ComputedBounds` changed since the previous
/// frame; unchanged renderables retain their last posted bound. Rows are terminated by the
/// first entry with `renderable_index < 0`.
fn apply_skinned_posed_bounds_extracted(
    space: &mut RenderSpaceState,
    extracted: &ExtractedSkinnedMeshRenderablesUpdate,
) {
    profiling::scope!("scene::apply_skinned_posed_bounds");
    for row in &extracted.bounds_updates {
        if row.renderable_index < 0 {
            break;
        }
        let idx = row.renderable_index as usize;
        if let Some(entry) = space.skinned_mesh_renderers.get_mut(idx) {
            entry.posed_object_bounds = Some(row.local_bounds);
        }
    }
}

/// Mutates [`RenderSpaceState`] using a pre-extracted [`ExtractedSkinnedMeshRenderablesUpdate`].
pub(crate) fn apply_skinned_mesh_renderables_update_extracted(
    space: &mut RenderSpaceState,
    extracted: &ExtractedSkinnedMeshRenderablesUpdate,
    transform_removals: &[TransformRemovalEvent],
    scene_id: i32,
) {
    profiling::scope!("scene::apply_skinned_meshes");
    fixup_skinned_bones_for_transform_removals(space, transform_removals);
    apply_skinned_removals_and_additions_extracted(space, extracted);
    apply_skinned_mesh_state_rows_extracted(space, extracted, scene_id);
    apply_skinned_bone_index_buffers_extracted(space, extracted, scene_id);
    apply_skinned_blendshape_weight_batches_extracted(space, extracted);
    apply_skinned_posed_bounds_extracted(space, extracted);
}

#[cfg(test)]
mod posed_bounds_tests {
    //! [`apply_skinned_posed_bounds_extracted`] writes per-renderable posed bounds onto
    //! [`SkinnedMeshRenderer::posed_object_bounds`] and honours the `renderable_index < 0`
    //! terminator used by the host.

    use glam::Vec3;

    use crate::scene::mesh_renderable::SkinnedMeshRenderer;
    use crate::scene::render_space::RenderSpaceState;
    use crate::shared::{RenderBoundingBox, SkinnedMeshBoundsUpdate};

    use super::{ExtractedSkinnedMeshRenderablesUpdate, apply_skinned_posed_bounds_extracted};

    fn make_space_with(n: usize) -> RenderSpaceState {
        let mut space = RenderSpaceState::default();
        for _ in 0..n {
            space
                .skinned_mesh_renderers
                .push(SkinnedMeshRenderer::default());
        }
        space
    }

    fn bounds(cx: f32, hx: f32) -> RenderBoundingBox {
        RenderBoundingBox {
            center: Vec3::new(cx, 0.0, 0.0),
            extents: Vec3::new(hx, hx, hx),
        }
    }

    fn extracted_with_rows(
        rows: Vec<SkinnedMeshBoundsUpdate>,
    ) -> ExtractedSkinnedMeshRenderablesUpdate {
        ExtractedSkinnedMeshRenderablesUpdate {
            bounds_updates: rows,
            ..Default::default()
        }
    }

    #[test]
    fn posed_bounds_are_stored_per_renderable() {
        let mut space = make_space_with(3);
        let extracted = extracted_with_rows(vec![
            SkinnedMeshBoundsUpdate {
                renderable_index: 0,
                local_bounds: bounds(1.0, 0.5),
            },
            SkinnedMeshBoundsUpdate {
                renderable_index: 2,
                local_bounds: bounds(2.0, 0.25),
            },
        ]);
        apply_skinned_posed_bounds_extracted(&mut space, &extracted);
        assert_eq!(
            space.skinned_mesh_renderers[0]
                .posed_object_bounds
                .unwrap()
                .center,
            Vec3::new(1.0, 0.0, 0.0)
        );
        assert!(
            space.skinned_mesh_renderers[1]
                .posed_object_bounds
                .is_none()
        );
        assert_eq!(
            space.skinned_mesh_renderers[2]
                .posed_object_bounds
                .unwrap()
                .extents,
            Vec3::new(0.25, 0.25, 0.25)
        );
    }

    #[test]
    fn negative_renderable_index_terminates_rows() {
        let mut space = make_space_with(2);
        let extracted = extracted_with_rows(vec![
            SkinnedMeshBoundsUpdate {
                renderable_index: 0,
                local_bounds: bounds(1.0, 0.5),
            },
            SkinnedMeshBoundsUpdate {
                renderable_index: -1,
                local_bounds: bounds(99.0, 99.0),
            },
            SkinnedMeshBoundsUpdate {
                renderable_index: 1,
                local_bounds: bounds(2.0, 0.5),
            },
        ]);
        apply_skinned_posed_bounds_extracted(&mut space, &extracted);
        assert!(
            space.skinned_mesh_renderers[0]
                .posed_object_bounds
                .is_some()
        );
        assert!(
            space.skinned_mesh_renderers[1]
                .posed_object_bounds
                .is_none()
        );
    }

    #[test]
    fn out_of_range_index_is_ignored() {
        let mut space = make_space_with(1);
        let extracted = extracted_with_rows(vec![SkinnedMeshBoundsUpdate {
            renderable_index: 99,
            local_bounds: bounds(1.0, 0.5),
        }]);
        apply_skinned_posed_bounds_extracted(&mut space, &extracted);
        assert!(
            space.skinned_mesh_renderers[0]
                .posed_object_bounds
                .is_none()
        );
    }
}

#[cfg(test)]
mod renderer_instance_id_tests {
    //! Regression coverage for skinned-renderer instance identity across dense table reindexing.

    use crate::scene::mesh_renderable::MeshRendererInstanceId;
    use crate::scene::render_space::RenderSpaceState;

    use super::{
        ExtractedSkinnedMeshRenderablesUpdate, apply_skinned_removals_and_additions_extracted,
    };

    #[test]
    fn skinned_instance_ids_are_fresh_and_survive_swap_remove() {
        let mut space = RenderSpaceState::default();
        apply_skinned_removals_and_additions_extracted(
            &mut space,
            &ExtractedSkinnedMeshRenderablesUpdate {
                additions: vec![20, 21, 22, -1],
                ..Default::default()
            },
        );
        let ids: Vec<_> = space
            .skinned_mesh_renderers
            .iter()
            .map(|renderer| renderer.base.instance_id)
            .collect();
        assert_eq!(
            ids,
            vec![
                MeshRendererInstanceId(1),
                MeshRendererInstanceId(2),
                MeshRendererInstanceId(3),
            ]
        );

        apply_skinned_removals_and_additions_extracted(
            &mut space,
            &ExtractedSkinnedMeshRenderablesUpdate {
                removals: vec![1, -1],
                additions: vec![23, -1],
                ..Default::default()
            },
        );
        let ids: Vec<_> = space
            .skinned_mesh_renderers
            .iter()
            .map(|renderer| renderer.base.instance_id)
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
