//! Stereo camera bundles for OpenXR multiview rendering.

use glam::{Mat4, Vec3};

use super::geometry::EyeView;

/// Per-eye matrices for an OpenXR stereo multiview view.
///
/// Consolidates the view-projection, view-only, projection, and eye positions so callers cannot
/// set one without the others. Present only on the HMD view; non-HMD views carry [`None`] for
/// this slot on [`crate::camera::HostCameraFrame::stereo`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StereoViewMatrices {
    /// Left-eye camera data.
    pub left: EyeView,
    /// Right-eye camera data.
    pub right: EyeView,
}

impl StereoViewMatrices {
    /// Builds a stereo bundle from left and right eye data.
    #[inline]
    pub const fn new(left: EyeView, right: EyeView) -> Self {
        Self { left, right }
    }

    /// Per-eye view-projection matrices.
    #[inline]
    pub const fn view_proj_pair(&self) -> (Mat4, Mat4) {
        (self.left.view_proj, self.right.view_proj)
    }

    /// Per-eye world-to-view matrices.
    #[inline]
    pub const fn view_pair(&self) -> (Mat4, Mat4) {
        (self.left.view, self.right.view)
    }

    /// Per-eye world-space camera positions.
    #[inline]
    pub const fn world_position_pair(&self) -> (Vec3, Vec3) {
        (self.left.world_position, self.right.world_position)
    }
}
