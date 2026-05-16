//! Per-render-space `BlitToDisplay` renderable state mirrored from
//! [`crate::shared::BlitToDisplayRenderablesUpdate`].
//!
//! Stores one [`BlitToDisplayEntry`] per renderable, indexed densely by host
//! `renderable_index`.

use crate::color_space::srgb_vec4_rgb_to_linear;
use crate::ipc::SharedMemoryAccessor;
use crate::scene::dense_update::{push_dense_additions, swap_remove_dense_indices};
use crate::scene::error::SceneError;
use crate::scene::render_space::RenderSpaceState;
use crate::shared::{
    BLIT_TO_DISPLAY_STATE_HOST_ROW_BYTES, BlitToDisplayRenderablesUpdate, BlitToDisplayState,
};

/// One host `BlitToDisplay` renderable mirrored on the renderer side.
///
/// `state.renderable_index < 0` indicates a freshly-added entry whose first state row has not
/// arrived yet; the present path skips entries in that condition.
#[derive(Clone, Copy, Debug, Default)]
pub struct BlitToDisplayEntry {
    /// Latest [`BlitToDisplayState`] for this renderable, populated from the
    /// `BlitToDisplayRenderablesUpdate.states` slab. `background_color` is normalized to linear
    /// RGB on apply.
    pub state: BlitToDisplayState,
    /// `true` once a host state row has been applied to [`Self::state`]. Skipped when `false`.
    pub state_initialized: bool,
}

/// Owned per-space `BlitToDisplay` payload extracted from shared memory.
#[derive(Default, Debug)]
pub struct ExtractedBlitToDisplayUpdate {
    /// Dense renderable removal indices (terminated by `< 0`).
    pub removals: Vec<i32>,
    /// New renderable host ids (terminated by `< 0`); ignored other than to grow the dense table.
    pub additions: Vec<i32>,
    /// Per-renderable state rows (terminated by `renderable_index < 0`).
    pub states: Vec<BlitToDisplayState>,
}

/// Reads every shared-memory buffer referenced by [`BlitToDisplayRenderablesUpdate`].
pub(crate) fn extract_blit_to_display_update(
    shm: &mut SharedMemoryAccessor,
    update: &BlitToDisplayRenderablesUpdate,
    scene_id: i32,
) -> Result<ExtractedBlitToDisplayUpdate, SceneError> {
    let mut out = ExtractedBlitToDisplayUpdate::default();
    if update.removals.length > 0 {
        let ctx = format!("blit_to_display removals scene_id={scene_id}");
        out.removals = shm
            .access_copy_diagnostic_with_context::<i32>(&update.removals, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    if update.additions.length > 0 {
        let ctx = format!("blit_to_display additions scene_id={scene_id}");
        out.additions = shm
            .access_copy_diagnostic_with_context::<i32>(&update.additions, Some(&ctx))
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    if update.states.length > 0 {
        let ctx = format!("blit_to_display states scene_id={scene_id}");
        out.states = shm
            .access_copy_memory_packable_rows::<BlitToDisplayState>(
                &update.states,
                BLIT_TO_DISPLAY_STATE_HOST_ROW_BYTES,
                Some(&ctx),
            )
            .map_err(SceneError::SharedMemoryAccess)?;
    }
    Ok(out)
}

/// Applies an [`ExtractedBlitToDisplayUpdate`] against the dense per-space table.
pub(crate) fn apply_blit_to_display_update_extracted(
    space: &mut RenderSpaceState,
    extracted: &ExtractedBlitToDisplayUpdate,
) {
    profiling::scope!("scene::apply_blit_to_displays");

    swap_remove_dense_indices(&mut space.blit_to_displays, &extracted.removals);

    push_dense_additions(&mut space.blit_to_displays, &extracted.additions, |_id| {
        BlitToDisplayEntry::default()
    });

    for state in &extracted.states {
        if state.renderable_index < 0 {
            break;
        }
        let idx = state.renderable_index as usize;
        let Some(entry) = space.blit_to_displays.get_mut(idx) else {
            continue;
        };
        let mut state = *state;
        state.background_color = srgb_vec4_rgb_to_linear(state.background_color);
        entry.state = state;
        entry.state_initialized = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec4;

    fn state_for(renderable_index: i32, display_index: i16, texture_id: i32) -> BlitToDisplayState {
        BlitToDisplayState {
            renderable_index,
            texture_id,
            background_color: Vec4::new(0.0, 0.0, 0.0, 1.0),
            display_index,
            flags: 0,
            _padding: [0; 1],
        }
    }

    #[test]
    fn additions_grow_dense_table() {
        let mut space = RenderSpaceState::default();
        let extracted = ExtractedBlitToDisplayUpdate {
            removals: vec![-1],
            additions: vec![100, 101, -1],
            states: vec![],
        };
        apply_blit_to_display_update_extracted(&mut space, &extracted);
        assert_eq!(space.blit_to_displays.len(), 2);
        assert!(!space.blit_to_displays[0].state_initialized);
        assert!(!space.blit_to_displays[1].state_initialized);
    }

    #[test]
    fn states_populate_existing_entries_in_order() {
        let mut space = RenderSpaceState::default();
        space
            .blit_to_displays
            .extend([BlitToDisplayEntry::default(), BlitToDisplayEntry::default()]);
        let extracted = ExtractedBlitToDisplayUpdate {
            removals: vec![-1],
            additions: vec![-1],
            states: vec![state_for(0, 0, 7), state_for(1, 2, 11)],
        };
        apply_blit_to_display_update_extracted(&mut space, &extracted);
        assert!(space.blit_to_displays[0].state_initialized);
        assert_eq!(space.blit_to_displays[0].state.texture_id, 7);
        assert_eq!(space.blit_to_displays[0].state.display_index, 0);
        assert_eq!(space.blit_to_displays[1].state.texture_id, 11);
        assert_eq!(space.blit_to_displays[1].state.display_index, 2);
    }

    #[test]
    fn removals_swap_remove_dense_entries() {
        let mut space = RenderSpaceState::default();
        space.blit_to_displays.extend([
            BlitToDisplayEntry {
                state: state_for(0, 0, 1),
                state_initialized: true,
            },
            BlitToDisplayEntry {
                state: state_for(1, 1, 2),
                state_initialized: true,
            },
            BlitToDisplayEntry {
                state: state_for(2, 2, 3),
                state_initialized: true,
            },
        ]);
        let extracted = ExtractedBlitToDisplayUpdate {
            removals: vec![0, -1],
            additions: vec![-1],
            states: vec![],
        };
        apply_blit_to_display_update_extracted(&mut space, &extracted);
        assert_eq!(space.blit_to_displays.len(), 2);
        assert_eq!(space.blit_to_displays[0].state.texture_id, 3);
        assert_eq!(space.blit_to_displays[1].state.texture_id, 2);
    }

    #[test]
    fn states_linearize_background_color() {
        let mut space = RenderSpaceState::default();
        space.blit_to_displays.push(BlitToDisplayEntry::default());
        let mut state = state_for(0, 0, 7);
        state.background_color = Vec4::new(0.5, 0.04045, 1.25, 0.33);
        let extracted = ExtractedBlitToDisplayUpdate {
            removals: vec![-1],
            additions: vec![-1],
            states: vec![state],
        };

        apply_blit_to_display_update_extracted(&mut space, &extracted);

        let color = space.blit_to_displays[0].state.background_color;
        assert!((color.x - 0.214_041_14).abs() < 0.000_001);
        assert!((color.y - (0.04045 / 12.92)).abs() < 0.000_001);
        assert!((color.z - 1.665_939_9).abs() < 0.000_001);
        assert_eq!(color.w, 0.33);
    }
}
