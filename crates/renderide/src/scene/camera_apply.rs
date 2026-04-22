//! [`CameraRenderablesUpdate`] ingestion from shared memory (FrooxEngine `CamerasManager` parity).

use crate::ipc::SharedMemoryAccessor;
use crate::shared::{CameraRenderablesUpdate, CameraState, CAMERA_STATE_HOST_ROW_BYTES};

use super::error::SceneError;
use super::render_space::RenderSpaceState;

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

/// One host camera renderable in a render space (dense table; `renderable_index` ↔ row in host state buffer).
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
    for &raw in extracted.removals.iter().take_while(|&&i| i >= 0) {
        let idx = raw as usize;
        if idx < space.cameras.len() {
            space.cameras.swap_remove(idx);
        }
    }
    for &node_id in extracted.additions.iter().take_while(|&&i| i >= 0) {
        space.cameras.push(CameraRenderableEntry {
            renderable_index: -1,
            transform_id: node_id,
            state: CameraState::default(),
            selective_transform_ids: Vec::new(),
            exclude_transform_ids: Vec::new(),
        });
    }
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
