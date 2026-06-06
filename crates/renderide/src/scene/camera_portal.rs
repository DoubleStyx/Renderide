//! Camera-portal renderable state mirrored from host updates.

use crate::ipc::SharedMemoryAccessor;
use crate::shared::{
    CAMERA_PORTAL_STATE_HOST_ROW_BYTES, CameraPortalState, CameraPortalsRenderablesUpdate,
};

use super::dense_update::{push_dense_additions, swap_remove_dense_indices_with_update};
use super::error::SceneError;
use super::render_space::RenderSpaceState;
use super::transforms::TransformRemovalEvent;
use super::world::fixup_transform_id;

const CAMERA_PORTAL_HAS_FAR_CLIP_VALUE_BIT: i32 = 1 << 0;
const CAMERA_PORTAL_HAS_CAMERA_CLEAR_MODE_BIT: i32 = 1 << 1;
const CAMERA_PORTAL_DISABLE_PER_PIXEL_LIGHTS_BIT: i32 = 1 << 2;
const CAMERA_PORTAL_DISABLE_SHADOWS_BIT: i32 = 1 << 3;
const CAMERA_PORTAL_PORTAL_MODE_BIT: i32 = 1 << 4;

/// One dense camera-portal renderable entry inside a render space.
#[derive(Debug, Clone)]
pub struct CameraPortalEntry {
    /// Dense renderable index assigned by the host.
    pub renderable_index: i32,
    /// Dense transform index that owns the camera portal component.
    pub transform_id: i32,
    /// Latest portal state row sent by the host.
    pub state: CameraPortalState,
}

/// Owned camera-portal update extracted from shared memory.
#[derive(Default, Debug)]
pub struct ExtractedCameraPortalRenderablesUpdate {
    /// Dense renderable removal indices terminated by a negative entry.
    pub removals: Vec<i32>,
    /// Added portal transform indices terminated by a negative entry.
    pub additions: Vec<i32>,
    /// Portal state rows terminated by `renderable_index < 0`.
    pub states: Vec<CameraPortalState>,
}

/// Returns whether the portal state carries an override far clip value.
#[inline]
pub fn camera_portal_has_far_clip_value(flags: i32) -> bool {
    flags & CAMERA_PORTAL_HAS_FAR_CLIP_VALUE_BIT != 0
}

/// Returns whether the portal state carries an override clear mode.
#[inline]
pub fn camera_portal_has_camera_clear_mode(flags: i32) -> bool {
    flags & CAMERA_PORTAL_HAS_CAMERA_CLEAR_MODE_BIT != 0
}

/// Returns whether per-pixel lights should be disabled for this portal render.
#[inline]
pub fn camera_portal_disable_per_pixel_lights(flags: i32) -> bool {
    flags & CAMERA_PORTAL_DISABLE_PER_PIXEL_LIGHTS_BIT != 0
}

/// Returns whether shadow rendering should be disabled for this portal render.
#[inline]
pub fn camera_portal_disable_shadows(flags: i32) -> bool {
    flags & CAMERA_PORTAL_DISABLE_SHADOWS_BIT != 0
}

/// Returns whether the state represents portal mode rather than mirror mode.
#[inline]
pub fn camera_portal_portal_mode(flags: i32) -> bool {
    flags & CAMERA_PORTAL_PORTAL_MODE_BIT != 0
}

/// Reads every camera-portal shared-memory buffer for one render-space update.
pub(crate) fn extract_camera_portal_renderables_update(
    shm: &mut SharedMemoryAccessor,
    update: &CameraPortalsRenderablesUpdate,
    scene_id: i32,
) -> Result<ExtractedCameraPortalRenderablesUpdate, SceneError> {
    let mut out = ExtractedCameraPortalRenderablesUpdate::default();
    if update.removals.length > 0 {
        let ctx = format!("camera portal removals scene_id={scene_id}");
        out.removals = shm
            .access_copy_diagnostic_with_context::<i32>(&update.removals, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    if update.additions.length > 0 {
        let ctx = format!("camera portal additions scene_id={scene_id}");
        out.additions = shm
            .access_copy_diagnostic_with_context::<i32>(&update.additions, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    if update.states.length > 0 {
        let ctx = format!("camera portal states scene_id={scene_id}");
        out.states = shm
            .access_copy_memory_packable_rows::<CameraPortalState>(
                &update.states,
                CAMERA_PORTAL_STATE_HOST_ROW_BYTES,
                Some(&ctx),
            )
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    Ok(out)
}

fn update_moved_camera_portal(portal: &mut CameraPortalEntry, index: i32) {
    portal.renderable_index = index;
}

fn build_added_camera_portal(transform_id: i32, renderable_index: i32) -> CameraPortalEntry {
    CameraPortalEntry {
        renderable_index,
        transform_id,
        state: CameraPortalState::default(),
    }
}

/// Applies a pre-extracted camera-portal update to one render space.
pub(crate) fn apply_camera_portal_renderables_update_extracted(
    space: &mut RenderSpaceState,
    extracted: &ExtractedCameraPortalRenderablesUpdate,
) {
    profiling::scope!("scene::apply_camera_portals");
    swap_remove_dense_indices_with_update(
        &mut space.camera_portals,
        &extracted.removals,
        update_moved_camera_portal,
    );
    push_dense_additions(
        &mut space.camera_portals,
        &extracted.additions,
        &build_added_camera_portal,
    );
    for state in &extracted.states {
        if state.renderable_index < 0 {
            break;
        }
        let idx = state.renderable_index as usize;
        let Some(entry) = space.camera_portals.get_mut(idx) else {
            continue;
        };
        entry.renderable_index = state.renderable_index;
        entry.state = *state;
    }
}

/// Updates cached portal transform indices after dense transform swap-removals.
pub(crate) fn fixup_camera_portals_for_transform_removals(
    space: &mut RenderSpaceState,
    removals: &[TransformRemovalEvent],
) {
    if removals.is_empty() || space.camera_portals.is_empty() {
        return;
    }
    for removal in removals {
        for portal in &mut space.camera_portals {
            portal.transform_id = fixup_transform_id(
                portal.transform_id,
                removal.removed_index,
                removal.last_index_before_swap,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CameraPortalEntry, ExtractedCameraPortalRenderablesUpdate,
        apply_camera_portal_renderables_update_extracted, camera_portal_disable_per_pixel_lights,
        camera_portal_disable_shadows, camera_portal_has_camera_clear_mode,
        camera_portal_has_far_clip_value, camera_portal_portal_mode,
    };
    use crate::scene::render_space::RenderSpaceState;
    use crate::shared::CameraPortalState;

    #[test]
    fn flag_helpers_match_host_bits() {
        let flags = 0b1_1111;
        assert!(camera_portal_has_far_clip_value(flags));
        assert!(camera_portal_has_camera_clear_mode(flags));
        assert!(camera_portal_disable_per_pixel_lights(flags));
        assert!(camera_portal_disable_shadows(flags));
        assert!(camera_portal_portal_mode(flags));

        assert!(!camera_portal_has_far_clip_value(0));
        assert!(!camera_portal_has_camera_clear_mode(0));
        assert!(!camera_portal_disable_per_pixel_lights(0));
        assert!(!camera_portal_disable_shadows(0));
        assert!(!camera_portal_portal_mode(0));
    }

    #[test]
    fn apply_adds_and_updates_camera_portal_state() {
        let mut space = RenderSpaceState::default();
        let update = ExtractedCameraPortalRenderablesUpdate {
            additions: vec![9, -1],
            states: vec![CameraPortalState {
                renderable_index: 0,
                mesh_renderer_index: 4,
                render_texture_id: 77,
                flags: 0b1_0000,
                ..CameraPortalState::default()
            }],
            ..ExtractedCameraPortalRenderablesUpdate::default()
        };

        apply_camera_portal_renderables_update_extracted(&mut space, &update);

        assert_eq!(space.camera_portals.len(), 1);
        assert_eq!(space.camera_portals[0].renderable_index, 0);
        assert_eq!(space.camera_portals[0].transform_id, 9);
        assert_eq!(space.camera_portals[0].state.mesh_renderer_index, 4);
        assert_eq!(space.camera_portals[0].state.render_texture_id, 77);
        assert!(camera_portal_portal_mode(
            space.camera_portals[0].state.flags
        ));
    }

    #[test]
    fn apply_removes_with_swap_update_and_stops_at_sentinel() {
        let mut space = RenderSpaceState {
            camera_portals: vec![
                CameraPortalEntry {
                    renderable_index: 0,
                    transform_id: 10,
                    state: CameraPortalState::default(),
                },
                CameraPortalEntry {
                    renderable_index: 1,
                    transform_id: 11,
                    state: CameraPortalState::default(),
                },
                CameraPortalEntry {
                    renderable_index: 2,
                    transform_id: 12,
                    state: CameraPortalState::default(),
                },
            ],
            ..RenderSpaceState::default()
        };
        let update = ExtractedCameraPortalRenderablesUpdate {
            removals: vec![1, -1, 0],
            ..ExtractedCameraPortalRenderablesUpdate::default()
        };

        apply_camera_portal_renderables_update_extracted(&mut space, &update);

        assert_eq!(space.camera_portals.len(), 2);
        assert_eq!(space.camera_portals[0].renderable_index, 0);
        assert_eq!(space.camera_portals[1].renderable_index, 1);
        assert_eq!(space.camera_portals[1].transform_id, 12);
    }
}
