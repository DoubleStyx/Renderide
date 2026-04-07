//! Helpers for processing FrameSubmitData in Session.

use crate::shared::{FrameSubmitData, RenderTransform};

/// Primary view selection from frame data.
#[derive(Debug, Clone)]
pub struct PrimaryViewSelection {
    pub space_id: i32,
    pub override_view_position: bool,
    pub view_position_is_external: bool,
    pub root_transform: RenderTransform,
    pub view_transform: RenderTransform,
}

/// Applies clip planes, FOV, and output state from frame data to the given mutable refs.
pub fn apply_clip_and_output_state(
    data: &FrameSubmitData,
    near_clip: &mut f32,
    far_clip: &mut f32,
    desktop_fov: &mut f32,
    lock_cursor: &mut bool,
) {
    *near_clip = data.near_clip;
    *far_clip = data.far_clip;
    *desktop_fov = data.desktop_fov;
    if let Some(ref output) = data.output_state {
        *lock_cursor = output.lock_cursor;
    }
}

/// Error when frame data has multiple active non-overlay spaces.
#[derive(Debug)]
pub struct MultipleActiveNonOverlaySpaces {
    /// Number of active non-overlay spaces (must be > 1 for this error).
    pub count: usize,
}

impl std::fmt::Display for MultipleActiveNonOverlaySpaces {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "multiple active non-overlay spaces (count={}), expected at most one",
            self.count
        )
    }
}

impl std::error::Error for MultipleActiveNonOverlaySpaces {}

/// Validates that at most one active non-overlay space exists.
pub fn validate_active_non_overlay(
    data: &FrameSubmitData,
) -> Result<(), MultipleActiveNonOverlaySpaces> {
    let active_non_overlay: Vec<_> = data
        .render_spaces
        .iter()
        .filter(|u| u.is_active && !u.is_overlay)
        .collect();
    if active_non_overlay.len() > 1 {
        return Err(MultipleActiveNonOverlaySpaces {
            count: active_non_overlay.len(),
        });
    }
    Ok(())
}

/// Selects the primary view from frame data.
/// View selection: override (freecam) → overriden_view_transform; else → root_transform.
/// When view_position_is_external is true (e.g. VR/third-person), view may need to come from
/// input/head state; we use root for now since the host does not send a separate head pose.
pub fn select_primary_view(data: &FrameSubmitData) -> Option<PrimaryViewSelection> {
    let active_non_overlay: Vec<_> = data
        .render_spaces
        .iter()
        .filter(|u| u.is_active && !u.is_overlay)
        .collect();

    if let Some(update) = active_non_overlay.first() {
        let view_transform = if update.override_view_position {
            update.overriden_view_transform
        } else {
            update.root_transform
        };
        return Some(PrimaryViewSelection {
            space_id: update.id,
            override_view_position: update.override_view_position,
            view_position_is_external: update.view_position_is_external,
            root_transform: update.root_transform,
            view_transform,
        });
    }

    data.render_spaces
        .first()
        .map(|first| PrimaryViewSelection {
            space_id: first.id,
            override_view_position: first.override_view_position,
            view_position_is_external: first.view_position_is_external,
            root_transform: first.root_transform,
            view_transform: first.root_transform,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_render_space(
        id: i32,
        is_active: bool,
        is_overlay: bool,
    ) -> crate::shared::RenderSpaceUpdate {
        crate::shared::RenderSpaceUpdate {
            id,
            is_active,
            is_overlay,
            ..Default::default()
        }
    }

    #[test]
    fn validate_active_non_overlay_ok_when_zero() {
        let data = FrameSubmitData {
            render_spaces: vec![
                make_render_space(0, false, false),
                make_render_space(1, true, true),
            ],
            ..Default::default()
        };
        assert!(validate_active_non_overlay(&data).is_ok());
    }

    #[test]
    fn validate_active_non_overlay_ok_when_one() {
        let data = FrameSubmitData {
            render_spaces: vec![
                make_render_space(0, true, false),
                make_render_space(1, true, true),
            ],
            ..Default::default()
        };
        assert!(validate_active_non_overlay(&data).is_ok());
    }

    #[test]
    fn validate_active_non_overlay_err_when_multiple() {
        let data = FrameSubmitData {
            render_spaces: vec![
                make_render_space(0, true, false),
                make_render_space(1, true, false),
            ],
            ..Default::default()
        };
        let err = validate_active_non_overlay(&data).unwrap_err();
        assert_eq!(err.count, 2);
    }

    #[test]
    fn select_primary_view_uses_active_non_overlay() {
        let mut space = make_render_space(5, true, false);
        space.override_view_position = true;
        space.overriden_view_transform.position = nalgebra::Vector3::new(10.0, 0.0, 0.0);
        space.root_transform.position = nalgebra::Vector3::new(0.0, 0.0, 0.0);

        let data = FrameSubmitData {
            render_spaces: vec![make_render_space(0, true, true), space],
            ..Default::default()
        };
        let sel = select_primary_view(&data).expect("should have selection");
        assert_eq!(sel.space_id, 5);
        assert!(sel.override_view_position);
        assert!((sel.view_transform.position.x - 10.0).abs() < 1e-5);
    }

    #[test]
    fn select_primary_view_fallback_to_first_when_no_non_overlay() {
        let data = FrameSubmitData {
            render_spaces: vec![
                make_render_space(3, true, true),
                make_render_space(7, true, true),
            ],
            ..Default::default()
        };
        let sel = select_primary_view(&data).expect("should fallback to first");
        assert_eq!(sel.space_id, 3);
    }
}
