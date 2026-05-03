//! [`CameraRenderablesUpdate`] ingestion from shared memory (FrooxEngine `CamerasManager` parity).

use crate::ipc::SharedMemoryAccessor;
use crate::shared::{CAMERA_STATE_HOST_ROW_BYTES, CameraRenderablesUpdate, CameraState};

use super::dense_update::{
    push_dense_additions, retain_live_transform_ids, swap_remove_dense_indices,
};
use super::error::SceneError;
use super::render_space::RenderSpaceState;
use super::transforms_apply::TransformRemovalEvent;
use super::world::fixup_transform_id;

/// Owned per-space camera-update payload extracted from shared memory.
///
/// Produced by [`extract_camera_renderables_update`] in the serial pre-extract phase so the
/// per-space apply work (see [`apply_camera_renderables_update_extracted`]) can run on a rayon
/// worker without holding a mutable borrow on the [`SharedMemoryAccessor`].
#[derive(Default, Debug)]
pub struct ExtractedCameraRenderablesUpdate {
    /// Dense camera-renderable removal indices (terminated by `< 0`).
    pub removals: Vec<i32>,
    /// Camera-renderable additions (host transform indices, terminated by `< 0`).
    pub additions: Vec<i32>,
    /// Per-camera state rows (terminated by `renderable_index < 0`).
    pub states: Vec<CameraState>,
    /// Optional selective / exclude transform-id slab (`None` when host omitted the buffer).
    pub transform_ids: Option<Vec<i32>>,
}

/// One host camera renderable in a render space (dense table; `renderable_index` <-> row in host state buffer).
#[derive(Debug, Clone)]
pub struct CameraRenderableEntry {
    /// Dense index in [`RenderSpaceState::cameras`] (matches [`CameraState::renderable_index`]).
    pub renderable_index: i32,
    /// Node / transform index for the camera component.
    pub transform_id: i32,
    /// Latest packed state from shared memory.
    pub state: CameraState,
    /// When non-empty, only these transform indices are drawn (Unity selective list).
    pub selective_transform_ids: Vec<i32>,
    /// Transform indices excluded from drawing when selective is empty.
    pub exclude_transform_ids: Vec<i32>,
}

/// Reads every shared-memory buffer referenced by [`CameraRenderablesUpdate`] into owned vectors.
///
/// Pre-extracting payloads here lets the per-space apply step run on a rayon worker without
/// holding a mutable borrow on [`SharedMemoryAccessor`].
pub(crate) fn extract_camera_renderables_update(
    shm: &mut SharedMemoryAccessor,
    update: &CameraRenderablesUpdate,
    scene_id: i32,
) -> Result<ExtractedCameraRenderablesUpdate, SceneError> {
    let mut out = ExtractedCameraRenderablesUpdate::default();
    if update.removals.length > 0 {
        let ctx = format!("camera removals scene_id={scene_id}");
        out.removals = shm
            .access_copy_diagnostic_with_context::<i32>(&update.removals, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    if update.additions.length > 0 {
        let ctx = format!("camera additions scene_id={scene_id}");
        out.additions = shm
            .access_copy_diagnostic_with_context::<i32>(&update.additions, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    if update.states.length > 0 {
        let ctx = format!("camera states scene_id={scene_id}");
        out.states = shm
            .access_copy_memory_packable_rows::<CameraState>(
                &update.states,
                CAMERA_STATE_HOST_ROW_BYTES,
                Some(&ctx),
            )
            .map_err(SceneError::SharedMemoryAccess)?;
        if update.transform_ids.length > 0 {
            let ctx_t = format!("camera transform_ids scene_id={scene_id}");
            out.transform_ids = Some(
                shm.access_copy_diagnostic_with_context::<i32>(&update.transform_ids, Some(&ctx_t))
                    .map_err(SceneError::SharedMemoryAccess)?,
            );
        }
    }
    Ok(out)
}

/// Mutates [`RenderSpaceState`] using a pre-extracted [`ExtractedCameraRenderablesUpdate`].
///
/// Single-threaded for one space; safe to call concurrently across distinct spaces.
pub(crate) fn apply_camera_renderables_update_extracted(
    space: &mut RenderSpaceState,
    extracted: &ExtractedCameraRenderablesUpdate,
) {
    profiling::scope!("scene::apply_cameras");
    swap_remove_dense_indices(&mut space.cameras, &extracted.removals);
    push_dense_additions(&mut space.cameras, &extracted.additions, |node_id| {
        CameraRenderableEntry {
            renderable_index: -1,
            transform_id: node_id,
            state: CameraState::default(),
            selective_transform_ids: Vec::new(),
            exclude_transform_ids: Vec::new(),
        }
    });
    let transform_ids = extracted.transform_ids.as_deref();
    let mut tid_cursor = 0usize;
    for state in &extracted.states {
        if state.renderable_index < 0 {
            break;
        }
        let idx = state.renderable_index as usize;
        let Some(entry) = space.cameras.get_mut(idx) else {
            continue;
        };
        entry.renderable_index = state.renderable_index;
        entry.state = *state;
        let sel = state.selective_render_count.max(0) as usize;
        let excl = state.exclude_render_count.max(0) as usize;
        let need = sel.saturating_add(excl);
        if let Some(slice) = transform_ids {
            if tid_cursor.saturating_add(need) <= slice.len() {
                if sel > 0 {
                    entry.selective_transform_ids = slice[tid_cursor..tid_cursor + sel].to_vec();
                    tid_cursor += sel;
                } else {
                    entry.selective_transform_ids.clear();
                }
                if excl > 0 {
                    entry.exclude_transform_ids = slice[tid_cursor..tid_cursor + excl].to_vec();
                    tid_cursor += excl;
                } else {
                    entry.exclude_transform_ids.clear();
                }
            } else {
                logger::warn!(
                    "camera state renderable_index={}: transform_ids buffer too short (need {need} after {tid_cursor}, len {})",
                    state.renderable_index,
                    slice.len()
                );
                entry.selective_transform_ids.clear();
                entry.exclude_transform_ids.clear();
            }
        } else {
            entry.selective_transform_ids.clear();
            entry.exclude_transform_ids.clear();
        }
    }
}

/// Rolls each camera's cached transform indices forward through this frame's transform
/// swap-removals, matching the host-side `RenderableIndex` reindexing done by
/// `RenderTransformManager.RemoveRenderTransform`. Must run before
/// [`apply_camera_renderables_update_extracted`] so new state rows land against correctly
/// reindexed cameras.
pub(crate) fn fixup_cameras_for_transform_removals(
    space: &mut RenderSpaceState,
    removals: &[TransformRemovalEvent],
) {
    if removals.is_empty() || space.cameras.is_empty() {
        return;
    }
    for removal in removals {
        for cam in &mut space.cameras {
            cam.transform_id = fixup_transform_id(
                cam.transform_id,
                removal.removed_index,
                removal.last_index_before_swap,
            );
            for id in &mut cam.selective_transform_ids {
                *id =
                    fixup_transform_id(*id, removal.removed_index, removal.last_index_before_swap);
            }
            for id in &mut cam.exclude_transform_ids {
                *id =
                    fixup_transform_id(*id, removal.removed_index, removal.last_index_before_swap);
            }
        }
    }
    // Selective/exclude lists treat a collapsed (-1) entry as a dead filter slot; drop them so
    // the per-frame apply doesn't use a stale index for rendering decisions.
    for cam in &mut space.cameras {
        retain_live_transform_ids(&mut cam.selective_transform_ids);
        retain_live_transform_ids(&mut cam.exclude_transform_ids);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::ids::RenderSpaceId;
    use crate::scene::render_space::RenderSpaceState;
    use crate::scene::transforms_apply::TransformRemovalEvent;

    fn space_with_camera(
        transform_id: i32,
        selective: Vec<i32>,
        exclude: Vec<i32>,
    ) -> RenderSpaceState {
        let mut space = RenderSpaceState {
            id: RenderSpaceId(0),
            ..Default::default()
        };
        space.cameras.push(CameraRenderableEntry {
            renderable_index: 0,
            transform_id,
            state: CameraState::default(),
            selective_transform_ids: selective,
            exclude_transform_ids: exclude,
        });
        space
    }

    #[test]
    fn camera_transform_id_follows_swap_remove() {
        let mut space = space_with_camera(42, vec![10, 42], vec![5, 42]);
        fixup_cameras_for_transform_removals(
            &mut space,
            &[TransformRemovalEvent {
                removed_index: 10,
                last_index_before_swap: 42,
            }],
        );
        let cam = &space.cameras[0];
        // The camera's transform was at the last index (42) and is now at the freed slot (10).
        assert_eq!(cam.transform_id, 10);
        // Selective list: the `10` entry was the removed transform -> dropped. The `42` entry
        // was the swapped-in last -> rewritten to `10`.
        assert_eq!(cam.selective_transform_ids, vec![10]);
        // Exclude list: `5` unaffected, `42` -> `10`.
        assert_eq!(cam.exclude_transform_ids, vec![5, 10]);
    }

    #[test]
    fn camera_fixup_ignores_unrelated_removals() {
        let mut space = space_with_camera(7, vec![1, 2, 3], vec![4]);
        fixup_cameras_for_transform_removals(
            &mut space,
            &[TransformRemovalEvent {
                removed_index: 100,
                last_index_before_swap: 200,
            }],
        );
        let cam = &space.cameras[0];
        assert_eq!(cam.transform_id, 7);
        assert_eq!(cam.selective_transform_ids, vec![1, 2, 3]);
        assert_eq!(cam.exclude_transform_ids, vec![4]);
    }
}
