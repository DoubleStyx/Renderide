//! View parameters and projection abstraction.
//!
//! Centralizes projection selection so the main view and overlay pass can later use different
//! projections (perspective vs orthographic).

use nalgebra::Matrix4;

use super::batch::SpaceDrawBatch;
use super::pass::{orthographic_projection_reverse_z, reverse_z_projection};
use crate::session::Session;

/// Projection type for a view.
#[derive(Clone, Debug)]
pub enum ViewProjection {
    /// Perspective projection with vertical FOV and aspect ratio.
    Perspective {
        /// Vertical field of view in radians.
        fov: f32,
        /// Aspect ratio (width / height).
        aspect: f32,
    },
    /// Orthographic projection with half-size and aspect ratio.
    Orthographic {
        /// Half-height of the orthographic view volume.
        half_size: f32,
        /// Aspect ratio (width / height).
        aspect: f32,
    },
}

/// View parameters: projection type, clip planes, and matrix generation.
///
/// Enables the main view and overlay pass to use different projections (e.g. perspective
/// for the scene, orthographic for UI overlays).
#[derive(Clone, Debug)]
pub struct ViewParams {
    /// Projection type (perspective or orthographic).
    pub projection: ViewProjection,
    /// Near clip plane distance.
    pub near_clip: f32,
    /// Far clip plane distance.
    pub far_clip: f32,
}

impl ViewParams {
    /// Builds viewport-based orthographic params for screen-space UI fallback.
    ///
    /// Used when overlay batches exist but the primary camera has perspective projection.
    /// half_size = 1.0 yields NDC from -1..1 in both axes; aspect matches viewport.
    pub fn orthographic_viewport_fallback(aspect: f32, near_clip: f32, far_clip: f32) -> Self {
        Self {
            projection: ViewProjection::Orthographic {
                half_size: 1.0,
                aspect,
            },
            near_clip: near_clip.max(0.01),
            far_clip,
        }
    }

    /// Computes overlay projection override for the current frame.
    ///
    /// Returns `Some(ViewParams)` when overlays should use orthographic projection:
    /// - If the primary camera task has orthographic params, uses those.
    /// - Else if any batch has `is_overlay`, uses `orthographic_viewport_fallback`.
    ///   Returns `None` when no overlay projection override is needed.
    pub fn overlay_projection_for_frame(
        session: &Session,
        draw_batches: &[SpaceDrawBatch],
        aspect: f32,
    ) -> Option<Self> {
        session
            .primary_camera_task()
            .and_then(|t| t.parameters.as_ref())
            .filter(|p| p.projection == crate::shared::CameraProjection::orthographic)
            .map(|p| Self {
                projection: ViewProjection::Orthographic {
                    half_size: p.orthographic_size,
                    aspect,
                },
                near_clip: p.near_clip.max(0.01),
                far_clip: p.far_clip,
            })
            .or_else(|| {
                let has_overlay_batches = draw_batches.iter().any(|b| b.is_overlay);
                if has_overlay_batches {
                    Some(Self::orthographic_viewport_fallback(
                        aspect,
                        session.near_clip(),
                        session.far_clip(),
                    ))
                } else {
                    None
                }
            })
    }

    /// Builds perspective view params from session and aspect ratio.
    ///
    /// Uses `session.desktop_fov`, `session.near_clip`, and `session.far_clip`.
    pub fn perspective_from_session(session: &Session, aspect: f32) -> Self {
        let near = session.near_clip().max(0.01);
        let far = session.far_clip();
        Self {
            projection: ViewProjection::Perspective {
                fov: session.desktop_fov().to_radians(),
                aspect,
            },
            near_clip: near,
            far_clip: far,
        }
    }

    /// Returns the projection matrix for this view.
    ///
    /// Uses reverse-Z depth (near → 1, far → -1 in NDC).
    pub fn to_projection_matrix(&self) -> Matrix4<f32> {
        let near = self.near_clip;
        let far = self.far_clip;

        match &self.projection {
            ViewProjection::Perspective { fov, aspect } => {
                reverse_z_projection(*aspect, *fov, near, far)
            }
            ViewProjection::Orthographic { half_size, aspect } => {
                let half_width = half_size * aspect;
                orthographic_projection_reverse_z(half_width, *half_size, near, far)
            }
        }
    }
}
