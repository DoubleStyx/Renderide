//! Static and skinned mesh renderable updates from shared memory (Unity `RenderableManager` parity).

use std::collections::HashSet;
use std::sync::LazyLock;

use parking_lot::Mutex;

use crate::ipc::SharedMemoryAccessor;
use crate::shared::packing_extras::SKINNED_MESH_BOUNDS_UPDATE_HOST_ROW_BYTES;
use crate::shared::{
    BlendshapeUpdate, BlendshapeUpdateBatch, BoneAssignment, LayerType, MeshRenderablesUpdate,
    MeshRendererState, SkinnedMeshBoundsUpdate, SkinnedMeshRenderablesUpdate,
    MESH_RENDERER_STATE_HOST_ROW_BYTES,
};

use super::error::SceneError;
use super::mesh_material_row::apply_mesh_renderer_state_row;
use super::mesh_renderable::{SkinnedMeshRenderer, StaticMeshRenderer};
use super::render_space::RenderSpaceState;
use super::transforms_apply::TransformRemovalEvent;
use super::world::fixup_transform_id;

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

static BONE_INDEX_EMPTY_WARNED_SCENES: LazyLock<Mutex<HashSet<i32>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

/// Once-per-scene dedup for [`apply_mesh_renderables_update_extracted`] out-of-range
/// `renderable_index` warnings.
///
/// The host's [`crate::shared::MeshRendererState::renderable_index`] should always be in range
/// `[0, static_mesh_renderers.len())` when state rows are applied. An out-of-range row indicates
/// a host-renderer protocol drift (e.g. an addition row was dropped, or a previous removals
/// batch was skipped) — silently ignoring the row leaves the renderable invisible until the
/// host re-emits, which is exactly the failure mode the `instance_changed_buffer` fix targets.
static STATIC_MESH_OOB_WARNED_SCENES: LazyLock<Mutex<HashSet<i32>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

/// Same as [`STATIC_MESH_OOB_WARNED_SCENES`] but for skinned mesh state rows.
static SKINNED_MESH_OOB_WARNED_SCENES: LazyLock<Mutex<HashSet<i32>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

fn warn_oob_renderable_index_once(
    scene_id: i32,
    kind: &'static str,
    bad_index: usize,
    len: usize,
    warned: &Mutex<HashSet<i32>>,
) {
    let mut w = warned.lock();
    if w.insert(scene_id) {
        logger::warn!(
            "{kind} mesh state: renderable_index {bad_index} out of range (len={len}) in scene_id={scene_id}; row dropped silently. Suggests host-renderer protocol drift; subsequent occurrences in this scene are suppressed."
        );
    }
}

/// Skinned renderer count above which the inner fixup loop fans out to the rayon pool.
///
/// Each entry touches its own `bone_transform_indices` slab (often hundreds of ids), so the
/// removals × renderers × bones product grows quickly. Above this size rayon amortizes its
/// overhead; below it the serial path avoids the dispatch cost.
const SKINNED_FIXUP_PARALLEL_MIN: usize = 128;

/// Static renderer count above which the fixup fans out to the rayon pool (matches
/// [`SKINNED_FIXUP_PARALLEL_MIN`] / the layer-fixup threshold; a single `node_id` write per
/// entry is cheap, but large scenes push the removals × renderers product into the tens of
/// thousands during bulk teardowns).
const STATIC_FIXUP_PARALLEL_MIN: usize = 128;

/// Rolls each [`StaticMeshRenderer::node_id`] forward through this frame's transform
/// swap-removals so existing entries follow their transform when it was swap-moved into a
/// freed slot (host-side `RenderTransformManager.RemoveRenderTransform`). Must run before
/// [`apply_mesh_renderables_update_extracted`] so any new state rows land on correctly
/// reindexed entries.
pub(crate) fn fixup_static_meshes_for_transform_removals(
    space: &mut RenderSpaceState,
    removals: &[TransformRemovalEvent],
) {
    if removals.is_empty() || space.static_mesh_renderers.is_empty() {
        return;
    }
    let use_parallel = space.static_mesh_renderers.len() >= STATIC_FIXUP_PARALLEL_MIN;
    for ev in removals {
        let removed_id = ev.removed_index;
        let last_index = ev.last_index_before_swap;
        if use_parallel {
            use rayon::prelude::*;
            space.static_mesh_renderers.par_iter_mut().for_each(|m| {
                m.node_id = fixup_transform_id(m.node_id, removed_id, last_index);
            });
        } else {
            for m in &mut space.static_mesh_renderers {
                m.node_id = fixup_transform_id(m.node_id, removed_id, last_index);
            }
        }
    }
}

fn fixup_skinned_bones_for_transform_removals(
    space: &mut RenderSpaceState,
    removals: &[TransformRemovalEvent],
) {
    let use_parallel = space.skinned_mesh_renderers.len() >= SKINNED_FIXUP_PARALLEL_MIN;
    for ev in removals {
        let removed_id = ev.removed_index;
        let last_index = ev.last_index_before_swap;
        if use_parallel {
            use rayon::prelude::*;
            space
                .skinned_mesh_renderers
                .par_iter_mut()
                .for_each(|entry| {
                    entry.base.node_id =
                        fixup_transform_id(entry.base.node_id, removed_id, last_index);
                    for id in &mut entry.bone_transform_indices {
                        *id = fixup_transform_id(*id, removed_id, last_index);
                    }
                    if let Some(rid) = entry.root_bone_transform_id {
                        entry.root_bone_transform_id =
                            Some(fixup_transform_id(rid, removed_id, last_index));
                    }
                });
        } else {
            for entry in &mut space.skinned_mesh_renderers {
                entry.base.node_id = fixup_transform_id(entry.base.node_id, removed_id, last_index);
                for id in &mut entry.bone_transform_indices {
                    *id = fixup_transform_id(*id, removed_id, last_index);
                }
                if let Some(rid) = entry.root_bone_transform_id {
                    entry.root_bone_transform_id =
                        Some(fixup_transform_id(rid, removed_id, last_index));
                }
            }
        }
    }
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
    for &raw in extracted.removals.iter().take_while(|&&i| i >= 0) {
        let idx = raw as usize;
        if idx < space.static_mesh_renderers.len() {
            space.static_mesh_renderers.swap_remove(idx);
        }
    }
    for &node_id in extracted.additions.iter().take_while(|&&i| i >= 0) {
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
    for &raw in extracted.removals.iter().take_while(|&&i| i >= 0) {
        let idx = raw as usize;
        if idx < space.skinned_mesh_renderers.len() {
            space.skinned_mesh_renderers.swap_remove(idx);
        }
    }
    for &node_id in extracted.additions.iter().take_while(|&&i| i >= 0) {
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
        // `index_offset + bone_count` could wrap on 32-bit usize (release builds wrap on
        // arithmetic overflow); use `checked_add` so a corrupted host cannot bypass the bounds
        // check via wraparound.
        let Some(end) = index_offset.checked_add(bone_count) else {
            break;
        };
        if idx < space.skinned_mesh_renderers.len() && end <= indexes.len() {
            let ids: Vec<i32> = indexes[index_offset..end].to_vec();
            space.skinned_mesh_renderers[idx].bone_transform_indices = ids;
            space.skinned_mesh_renderers[idx].root_bone_transform_id =
                if assignment.root_bone_transform_id >= 0 {
                    Some(assignment.root_bone_transform_id)
                } else {
                    None
                };
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
        // `update_offset + count` may wrap on 32-bit usize; use `checked_add`.
        let Some(end) = update_offset.checked_add(count) else {
            break;
        };
        if idx < space.skinned_mesh_renderers.len() && end <= updates.len() {
            let weights = &mut space.skinned_mesh_renderers[idx].base.blend_shape_weights;
            for upd in &updates[update_offset..end] {
                let bi = upd.blendshape_index.max(0) as usize;
                // Cap the blendshape index so a corrupted host cannot drive a multi-gigabyte
                // `weights.resize` via an attacker-chosen index. `MAX_BLENDSHAPE_INDEX` matches
                // the existing renderer-side cap in `assets::mesh::layout` (4096 shapes per
                // mesh). Out-of-range entries are dropped.
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

/// Maximum blendshape index accepted from IPC blendshape weight updates.
///
/// Matches the cap enforced by [`crate::assets::mesh::layout`] when extracting blendshape
/// data; updates referencing higher indices are silently dropped to prevent attacker-driven
/// `Vec::resize` on the per-renderable weight array.
const MAX_BLENDSHAPE_INDEX: usize = 4096;

/// Stores host-computed posed object-space bounds onto skinned renderables for culling.
///
/// The host emits one row per renderable whose [`SkinnedMeshRenderer::ComputedBounds`] changed
/// since the previous frame; unchanged renderables retain their last posted bound. Rows are
/// terminated by the first entry with `renderable_index < 0`.
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

    use super::{apply_skinned_posed_bounds_extracted, ExtractedSkinnedMeshRenderablesUpdate};

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
        assert!(space.skinned_mesh_renderers[1]
            .posed_object_bounds
            .is_none());
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
        assert!(space.skinned_mesh_renderers[0]
            .posed_object_bounds
            .is_some());
        // The terminator row must prevent the third entry from reaching renderable 1.
        assert!(space.skinned_mesh_renderers[1]
            .posed_object_bounds
            .is_none());
    }

    #[test]
    fn out_of_range_index_is_ignored() {
        let mut space = make_space_with(1);
        let extracted = extracted_with_rows(vec![SkinnedMeshBoundsUpdate {
            renderable_index: 99,
            local_bounds: bounds(1.0, 0.5),
        }]);
        apply_skinned_posed_bounds_extracted(&mut space, &extracted);
        assert!(space.skinned_mesh_renderers[0]
            .posed_object_bounds
            .is_none());
    }
}

#[cfg(test)]
mod renderer_instance_id_tests {
    //! Regression coverage for renderer-local identity across dense table reindexing.

    use crate::scene::mesh_renderable::MeshRendererInstanceId;
    use crate::scene::render_space::RenderSpaceState;

    use super::{
        apply_mesh_renderables_update_extracted, apply_skinned_removals_and_additions_extracted,
        ExtractedMeshRenderablesUpdate, ExtractedSkinnedMeshRenderablesUpdate,
    };

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

#[cfg(test)]
mod static_fixup_tests {
    //! Regression coverage for [`fixup_static_meshes_for_transform_removals`]: ensures
    //! [`StaticMeshRenderer::node_id`] follows a transform when the host swap-moves it into a
    //! freed slot (same contract as the layer / skinned-mesh / override fixups).

    use crate::scene::mesh_renderable::StaticMeshRenderer;
    use crate::scene::render_space::RenderSpaceState;
    use crate::scene::transforms_apply::TransformRemovalEvent;

    use super::fixup_static_meshes_for_transform_removals;

    fn space_with_static_meshes(node_ids: &[i32]) -> RenderSpaceState {
        let mut space = RenderSpaceState::default();
        for &node_id in node_ids {
            space.static_mesh_renderers.push(StaticMeshRenderer {
                node_id,
                ..Default::default()
            });
        }
        space
    }

    #[test]
    fn static_mesh_node_id_follows_swap_remove() {
        let mut space = space_with_static_meshes(&[5, 42, 7]);
        fixup_static_meshes_for_transform_removals(
            &mut space,
            &[TransformRemovalEvent {
                removed_index: 5,
                last_index_before_swap: 42,
            }],
        );
        // Mesh that was at the removed transform collapses to -1 (its renderable will be removed
        // by the host in the same frame — we verify only the index fixup here).
        assert_eq!(space.static_mesh_renderers[0].node_id, -1);
        // Mesh whose transform was swap-moved from 42 into slot 5 follows to 5.
        assert_eq!(space.static_mesh_renderers[1].node_id, 5);
        // Unrelated mesh is untouched.
        assert_eq!(space.static_mesh_renderers[2].node_id, 7);
    }

    #[test]
    fn static_mesh_fixup_no_op_when_no_removals() {
        let mut space = space_with_static_meshes(&[1, 2, 3]);
        fixup_static_meshes_for_transform_removals(&mut space, &[]);
        assert_eq!(
            space
                .static_mesh_renderers
                .iter()
                .map(|m| m.node_id)
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn static_mesh_node_id_cleared_when_mesh_was_on_removed_transform() {
        let mut space = space_with_static_meshes(&[1]);
        fixup_static_meshes_for_transform_removals(
            &mut space,
            &[TransformRemovalEvent {
                removed_index: 1,
                last_index_before_swap: 1,
            }],
        );
        assert_eq!(space.static_mesh_renderers[0].node_id, -1);
    }

    /// Regression for the duplicate-hides-original bug: when host swap-removes a transform whose
    /// `last_index_before_swap` is the slot referenced by an existing static mesh (e.g. the most
    /// recently spawned object that just got duplicated), the orchestrated chain must remap that
    /// mesh's `node_id` to the freed slot exactly once. The inline `fixup_transform_id` loop that
    /// previously sat inside [`apply_transform_removals_ordered`] caused a second remap that turned
    /// the now-valid `node_id` into `-1`, hiding the original renderable.
    #[test]
    fn static_mesh_survives_transform_removal_when_swapped_into_freed_slot() {
        use glam::{Quat, Vec3};

        use crate::scene::transforms_apply::{
            apply_transforms_update_extracted, ExtractedTransformsUpdate,
        };
        use crate::scene::world::WorldTransformCache;
        use crate::shared::RenderTransform;

        let identity = RenderTransform {
            position: Vec3::ZERO,
            scale: Vec3::ONE,
            rotation: Quat::IDENTITY,
        };

        // Two transforms ([helper, original]); a static mesh anchored to the original at index 1.
        let mut space = RenderSpaceState::default();
        space.nodes.push(identity);
        space.nodes.push(identity);
        space.node_parents.push(-1);
        space.node_parents.push(-1);
        space
            .static_mesh_renderers
            .push(StaticMeshRenderer::default());
        space.static_mesh_renderers[0].node_id = 1;

        let mut cache = WorldTransformCache {
            world_matrices: vec![glam::Mat4::IDENTITY; 2],
            computed: vec![false; 2],
            local_matrices: vec![glam::Mat4::IDENTITY; 2],
            local_dirty: vec![true; 2],
            degenerate_scales: vec![false; 2],
            visit_epoch: vec![0; 2],
            walk_epoch: 0,
            children: Vec::new(),
            children_dirty: true,
        };

        let extracted = ExtractedTransformsUpdate {
            removals: vec![0, -1],
            target_transform_count: 1,
            ..Default::default()
        };
        let mut removal_events = Vec::new();
        apply_transforms_update_extracted(
            &mut space,
            &mut cache,
            crate::scene::ids::RenderSpaceId(0),
            &extracted,
            &mut removal_events,
        );

        fixup_static_meshes_for_transform_removals(&mut space, &removal_events);

        assert_eq!(space.nodes.len(), 1);
        assert_eq!(
            space.static_mesh_renderers[0].node_id, 0,
            "original's transform was swap-moved from slot 1 into slot 0; mesh must follow"
        );
    }
}
