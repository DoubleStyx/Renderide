//! Camera-portal view-planning data.

use glam::Mat4;

use crate::camera::{CameraPortalMode, CameraPortalSourceView, CameraRenderRect};
use crate::scene::RenderSpaceId;
use crate::shared::{CameraPortalState, RenderingContext};

/// One resolved source eye and target rect for a camera-portal render.
#[derive(Clone, Copy)]
pub(super) struct CameraPortalSourceViewPlan {
    /// Immutable source camera data used to build the portal view.
    pub(super) source: CameraPortalSourceView,
    /// Rectangle within the host render texture that receives this view.
    pub(super) render_rect: CameraRenderRect,
    /// Eye index for split stereo targets; mono targets use zero.
    pub(super) eye_index: u8,
}

/// Fixed-size set of camera-portal source plans for mono or stereo rendering.
#[derive(Clone, Copy)]
pub(super) struct CameraPortalSourceViewPlans {
    len: usize,
    plans: [CameraPortalSourceViewPlan; 2],
}

/// Scene and render-target snapshot for one camera portal task.
#[derive(Clone, Copy)]
pub(super) struct CameraPortalViewTask {
    /// Render space containing the camera portal.
    pub(super) render_space_id: RenderSpaceId,
    /// Dense portal row within the render space.
    pub(super) portal_index: usize,
    /// Host portal state copied before GPU target allocation.
    pub(super) state: CameraPortalState,
    /// Stable host renderable index for view identity.
    pub(super) renderable_index: i32,
    /// Host render texture asset id to write.
    pub(super) render_texture_id: i32,
    /// Resolved mirror or portal behavior.
    pub(super) mode: CameraPortalMode,
    /// Render context used for transform/material overrides.
    pub(super) render_context: RenderingContext,
    /// Surface matrix copied before per-eye view construction.
    pub(super) surface_world_matrix: Mat4,
}

impl CameraPortalSourceViewPlans {
    /// Builds a single full-target camera-portal source plan.
    pub(super) fn mono(
        source: CameraPortalSourceView,
        target_extent_px: (u32, u32),
    ) -> Option<Self> {
        let render_rect = CameraRenderRect {
            origin_px: (0, 0),
            extent_px: target_extent_px,
        };
        if render_rect.extent_px.0 == 0 || render_rect.extent_px.1 == 0 {
            return None;
        }
        let plan = CameraPortalSourceViewPlan {
            source,
            render_rect,
            eye_index: 0,
        };
        Some(Self {
            len: 1,
            plans: [plan, plan],
        })
    }

    /// Builds split left/right source plans for a stereo camera-portal target.
    pub(super) fn stereo(
        left: CameraPortalSourceView,
        right: CameraPortalSourceView,
        target_extent_px: (u32, u32),
    ) -> Option<Self> {
        let (left_rect, right_rect) = camera_portal_stereo_render_rects(target_extent_px)?;
        Some(Self {
            len: 2,
            plans: [
                CameraPortalSourceViewPlan {
                    source: left,
                    render_rect: left_rect,
                    eye_index: 0,
                },
                CameraPortalSourceViewPlan {
                    source: right,
                    render_rect: right_rect,
                    eye_index: 1,
                },
            ],
        })
    }

    /// Iterates the active source plans.
    pub(super) fn iter(self) -> impl Iterator<Item = CameraPortalSourceViewPlan> {
        self.plans.into_iter().take(self.len)
    }
}

/// Splits a stereo camera-portal target into left and right render rectangles.
pub(super) fn camera_portal_stereo_render_rects(
    target_extent_px: (u32, u32),
) -> Option<(CameraRenderRect, CameraRenderRect)> {
    let (width, height) = target_extent_px;
    if width < 2 || height == 0 {
        return None;
    }
    let left_width = width / 2;
    let right_width = width - left_width;
    if left_width == 0 || right_width == 0 {
        return None;
    }
    Some((
        CameraRenderRect {
            origin_px: (0, 0),
            extent_px: (left_width, height),
        },
        CameraRenderRect {
            origin_px: (left_width, 0),
            extent_px: (right_width, height),
        },
    ))
}
