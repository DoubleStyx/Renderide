//! CPU layout for `shaders/source/modules/globals.wgsl` (`FrameGlobals` at `@group(0) @binding(0)`).

use bytemuck::{Pod, Zeroable};
use glam::Mat4;

/// Uniform block matching WGSL `FrameGlobals` (80-byte size, 16-byte aligned).
///
/// Encodes camera position, per-eye coefficients for view-space Z from world position, clustered
/// grid dimensions, clip planes, light count, and viewport size for clustered forward sampling.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct FrameGpuUniforms {
    /// World-space camera position (`.w` unused).
    pub camera_world_pos: [f32; 4],
    /// Left-eye (or mono) world -> view-space Z: `dot(xyz, world) + w`.
    pub view_space_z_coeffs: [f32; 4],
    /// Right-eye world -> view-space Z. Set equal to `view_space_z_coeffs` in mono mode.
    pub view_space_z_coeffs_right: [f32; 4],
    /// Cluster grid width in tiles (X).
    pub cluster_count_x: u32,
    /// Cluster grid height in tiles (Y).
    pub cluster_count_y: u32,
    /// Depth slice count for clustered lighting (Z).
    pub cluster_count_z: u32,
    /// Camera near clip plane (view space, positive forward).
    pub near_clip: f32,
    /// Camera far clip plane (reverse-Z aware; matches shader expectations).
    pub far_clip: f32,
    /// Number of lights packed into the frame storage buffer for this pass.
    pub light_count: u32,
    /// Viewport width in pixels (physical).
    pub viewport_width: u32,
    /// Viewport height in pixels (physical).
    pub viewport_height: u32,
}

/// Inputs for [`FrameGpuUniforms::new_clustered`] (clustered forward + lighting).
#[derive(Clone, Copy, Debug)]
pub struct ClusteredFrameGlobalsParams {
    /// World-space camera position for the active view.
    pub camera_world_pos: glam::Vec3,
    /// Left-eye (or mono) view-space Z coefficients from world position.
    pub view_space_z_coeffs: [f32; 4],
    /// Right-eye view-space Z coefficients; equals `view_space_z_coeffs` in mono.
    pub view_space_z_coeffs_right: [f32; 4],
    /// Cluster grid width in tiles.
    pub cluster_count_x: u32,
    /// Cluster grid height in tiles.
    pub cluster_count_y: u32,
    /// Cluster grid depth (Z slices).
    pub cluster_count_z: u32,
    /// Near clip in view space (positive forward).
    pub near_clip: f32,
    /// Far clip (reverse-Z aware).
    pub far_clip: f32,
    /// Packed light count for the frame buffer.
    pub light_count: u32,
    /// Viewport width in physical pixels.
    pub viewport_width: u32,
    /// Viewport height in physical pixels.
    pub viewport_height: u32,
}

impl FrameGpuUniforms {
    /// Coefficients so `dot(coeffs.xyz, world) + coeffs.w` yields view-space Z for a world point.
    ///
    /// Uses the third row of the column-major world-to-view matrix (`glam` column vectors).
    pub fn view_space_z_coeffs_from_world_to_view(world_to_view: Mat4) -> [f32; 4] {
        let m = world_to_view;
        [m.x_axis.z, m.y_axis.z, m.z_axis.z, m.w_axis.z]
    }

    /// Builds per-frame uniforms for clustered forward and lighting.
    ///
    /// `params.view_space_z_coeffs_right` should equal `params.view_space_z_coeffs` in mono mode.
    pub fn new_clustered(params: ClusteredFrameGlobalsParams) -> Self {
        Self {
            camera_world_pos: [
                params.camera_world_pos.x,
                params.camera_world_pos.y,
                params.camera_world_pos.z,
                0.0,
            ],
            view_space_z_coeffs: params.view_space_z_coeffs,
            view_space_z_coeffs_right: params.view_space_z_coeffs_right,
            cluster_count_x: params.cluster_count_x,
            cluster_count_y: params.cluster_count_y,
            cluster_count_z: params.cluster_count_z,
            near_clip: params.near_clip,
            far_clip: params.far_clip,
            light_count: params.light_count,
            viewport_width: params.viewport_width,
            viewport_height: params.viewport_height,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_globals_size_80() {
        assert_eq!(std::mem::size_of::<FrameGpuUniforms>(), 80);
        assert_eq!(std::mem::size_of::<FrameGpuUniforms>() % 16, 0);
    }

    #[test]
    fn z_coeffs_extracts_third_row_for_translation_only_view() {
        // Translation-only view: world-to-view z = world.z + tz (tz from row 3, w component).
        let tz = 7.0;
        let m = Mat4::from_translation(glam::Vec3::new(0.0, 0.0, tz));
        let coeffs = FrameGpuUniforms::view_space_z_coeffs_from_world_to_view(m);
        assert_eq!(coeffs, [0.0, 0.0, 1.0, tz]);

        // Sanity: dot(coeffs.xyz, p) + coeffs.w matches (m * p).z for a sample point.
        let p = glam::Vec3::new(2.0, -3.0, 4.0);
        let view_z = (m * p.extend(1.0)).z;
        let dotted = coeffs[2].mul_add(p.z, coeffs[0].mul_add(p.x, coeffs[1] * p.y)) + coeffs[3];
        assert!((view_z - dotted).abs() < 1e-6);
    }

    #[test]
    fn z_coeffs_matches_third_component_under_yaw_rotation() {
        // Yaw should leave Z row invariant (rotation about Y keeps Z-basis).
        let m = Mat4::from_rotation_y(std::f32::consts::FRAC_PI_3);
        let coeffs = FrameGpuUniforms::view_space_z_coeffs_from_world_to_view(m);
        let p = glam::Vec3::new(1.5, -0.25, 2.0);
        let view_z = (m * p.extend(1.0)).z;
        let dotted = coeffs[2].mul_add(p.z, coeffs[0].mul_add(p.x, coeffs[1] * p.y)) + coeffs[3];
        assert!((view_z - dotted).abs() < 1e-5);
    }

    #[test]
    fn new_clustered_populates_fields_including_zero_w_for_camera_pos() {
        let u = FrameGpuUniforms::new_clustered(ClusteredFrameGlobalsParams {
            camera_world_pos: glam::Vec3::new(1.0, 2.0, 3.0),
            view_space_z_coeffs: [0.1, 0.2, 0.3, 0.4],
            view_space_z_coeffs_right: [0.5, 0.6, 0.7, 0.8],
            cluster_count_x: 16,
            cluster_count_y: 9,
            cluster_count_z: 24,
            near_clip: 0.05,
            far_clip: 1000.0,
            light_count: 42,
            viewport_width: 1920,
            viewport_height: 1080,
        });
        assert_eq!(u.camera_world_pos, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(u.view_space_z_coeffs, [0.1, 0.2, 0.3, 0.4]);
        assert_eq!(u.view_space_z_coeffs_right, [0.5, 0.6, 0.7, 0.8]);
        assert_eq!(u.cluster_count_x, 16);
        assert_eq!(u.cluster_count_y, 9);
        assert_eq!(u.cluster_count_z, 24);
        assert_eq!(u.near_clip, 0.05);
        assert_eq!(u.far_clip, 1000.0);
        assert_eq!(u.light_count, 42);
        assert_eq!(u.viewport_width, 1920);
        assert_eq!(u.viewport_height, 1080);
    }
}
