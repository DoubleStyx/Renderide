//! Algebraic camera primitives: clips, viewports, poses, and single-eye views.

use glam::{Mat4, Vec3};

use super::projection::{clamp_desktop_fov_degrees, reverse_z_orthographic, reverse_z_perspective};
use super::view::apply_view_handedness_fix;

/// Positive camera clip-plane distances in view space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraClipPlanes {
    /// Near clip distance.
    pub near: f32,
    /// Far clip distance.
    pub far: f32,
}

impl CameraClipPlanes {
    /// Builds a clip-plane pair.
    #[inline]
    pub const fn new(near: f32, far: f32) -> Self {
        Self { near, far }
    }
}

impl Default for CameraClipPlanes {
    #[inline]
    fn default() -> Self {
        Self {
            near: 0.01,
            far: 10_000.0,
        }
    }
}

/// Pixel extent for one camera view.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Viewport {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl Viewport {
    /// Builds a viewport from a width and height.
    #[inline]
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Builds a viewport from an existing tuple extent.
    #[inline]
    pub const fn from_tuple(value: (u32, u32)) -> Self {
        Self::new(value.0, value.1)
    }

    /// Returns `true` when either dimension is zero.
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// Aspect ratio with a minimum denominator of one pixel.
    #[inline]
    pub fn aspect(self) -> f32 {
        self.width as f32 / self.height.max(1) as f32
    }

    /// Number of tile columns needed to cover this viewport.
    #[inline]
    pub fn tile_columns(self, tile_size: u32) -> u32 {
        self.width.div_ceil(tile_size.max(1))
    }

    /// Number of tile rows needed to cover this viewport.
    #[inline]
    pub fn tile_rows(self, tile_size: u32) -> u32 {
        self.height.div_ceil(tile_size.max(1))
    }
}

/// Orthographic projection inputs for camera paths that need an overlay or secondary ortho pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrthographicProjectionSpec {
    /// Half-height of the orthographic volume in view space.
    pub half_height: f32,
    /// Clip planes for this orthographic projection.
    pub clip: CameraClipPlanes,
}

impl OrthographicProjectionSpec {
    /// Builds a projection spec, clamping the half-height away from zero.
    #[inline]
    pub fn new(half_height: f32, clip: CameraClipPlanes) -> Self {
        Self {
            half_height: half_height.max(1e-6),
            clip,
        }
    }

    /// Builds the reverse-Z orthographic projection for a viewport aspect ratio.
    #[inline]
    pub fn projection(self, viewport: Viewport) -> Mat4 {
        let half_width = self.half_height * viewport.aspect();
        reverse_z_orthographic(half_width, self.half_height, self.clip.near, self.clip.far)
    }
}

/// A camera pose represented as a world-to-view matrix and world-space position.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraPose {
    /// World-to-view matrix with the renderer handedness fix applied.
    pub world_to_view: Mat4,
    /// World-space camera position.
    pub world_position: Vec3,
}

impl CameraPose {
    /// Builds a pose from a camera world matrix, ignoring transform scale for the view matrix.
    ///
    /// Camera scale is consumed by the camera controller paths that scale orthographic size and
    /// clip planes. The view matrix itself follows camera position and rotation only.
    #[inline]
    pub fn from_world_matrix(camera_world_matrix: Mat4) -> Self {
        let (_, rotation, position) = camera_world_matrix.to_scale_rotation_translation();
        let camera_pose_matrix =
            Mat4::from_scale_rotation_translation(Vec3::ONE, rotation, position);
        Self {
            world_to_view: apply_view_handedness_fix(camera_pose_matrix.inverse()),
            world_position: position,
        }
    }
}

/// Complete single-eye camera data used by stereo and explicit secondary-camera views.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EyeView {
    /// World-to-view matrix.
    pub view: Mat4,
    /// View-to-clip projection.
    pub proj: Mat4,
    /// World-to-clip view-projection.
    pub view_proj: Mat4,
    /// World-space eye position.
    pub world_position: Vec3,
}

impl EyeView {
    /// Builds an eye view from resolved matrices and world position.
    #[inline]
    pub const fn new(view: Mat4, proj: Mat4, view_proj: Mat4, world_position: Vec3) -> Self {
        Self {
            view,
            proj,
            view_proj,
            world_position,
        }
    }

    /// Builds an eye view from a pose and projection.
    #[inline]
    pub fn from_pose_projection(pose: CameraPose, proj: Mat4) -> Self {
        Self {
            view: pose.world_to_view,
            proj,
            view_proj: proj * pose.world_to_view,
            world_position: pose.world_position,
        }
    }

    /// Builds a symmetric perspective eye from a pose and viewport.
    #[inline]
    pub fn perspective_from_pose(
        pose: CameraPose,
        viewport: Viewport,
        fov_degrees: f32,
        clip: CameraClipPlanes,
    ) -> Self {
        let proj = reverse_z_perspective(
            viewport.aspect(),
            clamp_desktop_fov_degrees(fov_degrees).to_radians(),
            clip.near,
            clip.far,
        );
        Self::from_pose_projection(pose, proj)
    }
}

#[cfg(test)]
mod tests {
    use glam::{Mat4, Quat, Vec3};

    use super::{CameraPose, Viewport};
    use crate::camera::apply_view_handedness_fix;

    #[test]
    fn viewport_aspect_and_tiles_are_stable_for_empty_height() {
        let viewport = Viewport::new(1920, 0);
        assert_eq!(viewport.aspect(), 1920.0);
        assert!(viewport.is_empty());
        assert_eq!(Viewport::new(65, 33).tile_columns(32), 3);
        assert_eq!(Viewport::new(65, 33).tile_rows(32), 2);
    }

    #[test]
    fn camera_pose_from_world_matrix_ignores_transform_scale() {
        let position = Vec3::new(4.0, 5.0, 6.0);
        let rotation = Quat::from_rotation_y(0.35);
        let camera_world =
            Mat4::from_scale_rotation_translation(Vec3::splat(2.5), rotation, position);

        let pose = CameraPose::from_world_matrix(camera_world);
        let expected_pose_matrix =
            Mat4::from_scale_rotation_translation(Vec3::ONE, rotation, position);
        let expected_view = apply_view_handedness_fix(expected_pose_matrix.inverse());

        assert!(pose.world_to_view.abs_diff_eq(expected_view, 1e-5));
        assert_eq!(pose.world_position, position);
    }
}
